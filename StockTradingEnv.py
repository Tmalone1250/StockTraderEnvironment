import gym
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from gym import spaces
import yfinance as yf
from robinhood_trader import RobinhoodTrader
import time

def apply_ema(values, alpha=0.1):
    """Applies Exponential Moving Average (EMA) for stability."""
    result = np.zeros_like(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
    return result

class ForwardDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ForwardDynamicsModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim)
        )
    
    def forward(self, state, action):
        # Create one-hot encoded action
        action_one_hot = F.one_hot(torch.tensor([action]), num_classes=self.action_dim).float().squeeze(0)
        
        # Ensure state is a tensor and has correct shape
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        
        # Concatenate state and action
        x = torch.cat([state, action_one_hot], dim=0)
        
        # Get prediction
        return self.network(x)

    def train_step(self, current_state, action, next_state):
        action_tensor = torch.tensor([action], dtype=torch.long)
        predicted_state = self.forward(torch.FloatTensor(current_state), action_tensor)
        loss = F.mse_loss(predicted_state, torch.FloatTensor(next_state))
        loss.backward()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        optimizer.step()
        optimizer.zero_grad()

class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, live_trading=False):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.live_trading = live_trading
        self.max_leverage = 2.0
        self.margin_requirement = 0.3
        
        # Initialize Robinhood trader if live trading is enabled
        self.trader = None
        if live_trading:
            self.trader = RobinhoodTrader()
            if not self.trader.login():
                raise Exception("Failed to login to Robinhood")
            # Get actual account balance
            account_info = self.trader.get_account_info()
            self.initial_balance = float(account_info['portfolio_value'])
        
        # Risk management parameters
        self.position_limit = 0.25
        self.stop_loss_threshold = -0.15
        self.trailing_stop_loss = -0.10
        self.max_drawdown_limit = -0.25
        self.volatility_scaling = True
        
        # Action space: 0 = Strong Sell (2x short), 1 = Sell (1x short), 2 = Hold, 3 = Buy (1x long), 4 = Strong Buy (2x long)
        self.action_space = spaces.Discrete(5)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        
        self.dynamics_model = ForwardDynamicsModel(
            state_dim=self.observation_space.shape[0],
            action_dim=self.action_space.n
        )
        
        # Risk metrics tracking
        self.position_high_water_mark = 0
        self.current_drawdown = 0
        self.position_entry_price = None
        self.position_entry_value = None
        self.pending_orders = {}
        
        self.reset()
    
    def calculate_position_size(self, action, current_price):
        """Calculate position size with risk management"""
        # Get volatility scaling factor
        if self.volatility_scaling:
            vol = self.df['Close'].rolling(window=20).std().iloc[self.current_step]
            vol_scale = 0.2 / (vol + 1e-6)  # Target 20% volatility
            vol_scale = np.clip(vol_scale, 0.5, 2.0)  # Limit scaling factor
        else:
            vol_scale = 1.0
        
        # Calculate base position size
        portfolio_value = self.balance + (self.holdings * current_price)
        max_position_value = portfolio_value * self.position_limit * vol_scale
        
        # Apply leverage limits
        position_multipliers = {
            0: -2.0,  # Strong Sell
            1: -1.0,  # Sell
            2: 0.0,   # Hold
            3: 1.0,   # Buy
            4: 2.0    # Strong Buy
        }
        
        multiplier = position_multipliers[action]
        leverage_used = abs(multiplier)
        
        # Scale position size based on available margin and leverage
        if multiplier > 0:  # Long positions
            max_shares = min(
                max_position_value * leverage_used / current_price,
                (self.balance * self.max_leverage) / current_price
            )
        else:  # Short positions
            max_shares = max(
                -max_position_value * leverage_used / current_price,
                -(self.balance / (current_price * self.margin_requirement))
            )
        
        return max_shares
    
    def check_stop_loss(self, current_price):
        """Check and apply stop loss conditions"""
        if self.holdings == 0:
            return False
        
        if self.position_entry_price is None:
            self.position_entry_price = current_price
            self.position_entry_value = abs(self.holdings * current_price)
            self.position_high_water_mark = self.position_entry_value
            return False
        
        # Calculate position P&L
        current_value = abs(self.holdings * current_price)
        self.position_high_water_mark = max(self.position_high_water_mark, current_value)
        
        # Calculate drawdown from high water mark
        drawdown = (current_value - self.position_high_water_mark) / self.position_high_water_mark
        
        # Calculate loss from entry
        loss_from_entry = (current_value - self.position_entry_value) / self.position_entry_value
        
        # Check stop loss conditions
        stop_loss_triggered = (
            loss_from_entry <= self.stop_loss_threshold or  # Fixed stop loss
            drawdown <= self.trailing_stop_loss or          # Trailing stop loss
            self.current_drawdown <= self.max_drawdown_limit # Portfolio drawdown limit
        )
        
        return stop_loss_triggered
    
    def execute_trade(self, action, current_price):
        """Execute trade in live or simulation mode"""
        if not self.live_trading:
            return super().execute_trade(action, current_price)
        
        # Calculate desired position based on action
        new_holdings = self.calculate_position_size(action, current_price)
        position_change = new_holdings - self.holdings
        
        if abs(position_change) < 0.01:  # Minimum trade size
            return
        
        try:
            # Place order through Robinhood
            symbol = self.df.index.name or 'UNKNOWN'
            side = 'buy' if position_change > 0 else 'sell'
            quantity = abs(position_change)
            
            order = self.trader.place_order(
                symbol=symbol,
                quantity=quantity,
                side=side,
                order_type='market'
            )
            
            if order:
                self.pending_orders[order['id']] = {
                    'quantity': quantity,
                    'side': side,
                    'price': current_price
                }
                
                # Wait for order to fill (with timeout)
                timeout = time.time() + 60  # 60 second timeout
                while time.time() < timeout:
                    status = self.trader.get_order_status(order['id'])
                    if status['state'] == 'filled':
                        executed_price = float(status['average_price'])
                        cost = position_change * executed_price
                        
                        if side == 'buy':
                            self.balance -= cost
                            self.holdings += quantity
                        else:
                            self.balance += abs(cost)
                            self.holdings -= quantity
                        
                        break
                    time.sleep(1)
                
                # Cancel order if it didn't fill
                if time.time() >= timeout:
                    self.trader.cancel_order(order['id'])
                
        except Exception as e:
            print(f"Trade execution failed: {str(e)}")
    
    def update_portfolio_value(self):
        """Update portfolio value from live trading or simulation"""
        if not self.live_trading:
            return super().update_portfolio_value()
        
        try:
            account_info = self.trader.get_account_info()
            self.portfolio_value = float(account_info['portfolio_value'])
            self.balance = float(account_info['buying_power'])
            
            # Update holdings from positions
            positions = self.trader.get_positions()
            self.holdings = sum(pos['quantity'] for pos in positions)
            
        except Exception as e:
            print(f"Failed to update portfolio value: {str(e)}")
    
    def get_current_price(self):
        """Get current price from live trading or historical data"""
        if not self.live_trading:
            return self.df.iloc[self.current_step]['Close']
        
        symbol = self.df.index.name or 'UNKNOWN'
        try:
            return float(self.trader.get_latest_price(symbol)[0])
        except Exception as e:
            print(f"Failed to get current price: {str(e)}")
            return None
    
    def close(self):
        """Cleanup when environment is done"""
        if self.live_trading and self.trader:
            # Cancel any pending orders
            for order_id in self.pending_orders:
                self.trader.cancel_order(order_id)
            # Logout from Robinhood
            self.trader.logout()
    
    def step(self, action):
        self.current_step += 1
        current_price = self.get_current_price()
        
        # Get current state for prediction error calculation
        current_state = self._get_observation()
        
        # Check stop loss before taking new action
        if self.check_stop_loss(current_price):
            # Close position at market price
            close_cost = self.holdings * current_price
            self.balance += close_cost
            self.holdings = 0
            self.position_entry_price = None
            action = 2  # Force hold action
        
        # Calculate new position size with risk management
        new_holdings = self.calculate_position_size(action, current_price)
        
        # Apply position change
        if new_holdings != self.holdings:
            cost = (new_holdings - self.holdings) * current_price
            if (cost < 0 and self.balance >= abs(cost)) or (cost > 0):
                self.balance -= cost
                self.holdings = new_holdings
                self.position_entry_price = current_price
                self.position_entry_value = abs(new_holdings * current_price)
                self.position_high_water_mark = self.position_entry_value
        
        # Update portfolio value and metrics
        self.portfolio_value = self.balance + (self.holdings * current_price)
        self.portfolio_values.append(self.portfolio_value)  # Track portfolio value
        
        # Update drawdown
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        self.current_drawdown = (self.portfolio_value - self.max_portfolio_value) / self.max_portfolio_value
        self.max_drawdown = min(self.max_drawdown, self.current_drawdown)
        
        # Calculate reward with risk-adjusted returns
        price_change = (current_price - self.df.iloc[self.current_step-1]['Close']) / self.df.iloc[self.current_step-1]['Close']
        position_value = self.holdings * current_price
        
        # Risk-adjusted reward calculation
        sharpe_ratio = price_change / (self.df['Close'].rolling(window=20).std().iloc[self.current_step] + 1e-6)
        reward = (price_change * position_value * sharpe_ratio) / self.initial_balance
        
        # Add small risk-seeking bonus for controlled leverage
        leverage_bonus = (abs(self.holdings * current_price) / self.initial_balance) * 0.005
        reward += leverage_bonus
        
        # Penalize excessive drawdown
        if self.current_drawdown < -0.2:  # More than 20% drawdown
            reward -= abs(self.current_drawdown + 0.2) * 0.1
        
        # Predict next state and calculate prediction error
        action_tensor = torch.tensor([action], dtype=torch.long)
        predicted_state = self.dynamics_model(torch.FloatTensor(current_state), action_tensor)
        next_state = self._get_observation()
        prediction_error = F.mse_loss(predicted_state, torch.FloatTensor(next_state)).item()
        
        # Update dynamics model
        self.dynamics_model.train_step(current_state, action, next_state)
        
        done = self.current_step >= len(self.df) - 1
        
        return self._get_observation(), reward, done, {
            'portfolio_value': self.portfolio_value,
            'holdings': self.holdings,
            'balance': self.balance,
            'prediction_error': prediction_error,
            'return': (self.portfolio_value - self.initial_balance) / self.initial_balance,
            'drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = 0
        self.portfolio_value = self.initial_balance
        self.portfolio_values = [self.initial_balance]  # Track portfolio values
        self.max_portfolio_value = self.initial_balance
        self.max_drawdown = 0
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current state observation"""
        # Get the current price data
        current_step_data = self.df.iloc[self.current_step]
        
        # Create observation with basic price data and account info
        obs = np.array([
            float(current_step_data['Close']),  # Current price
            float(current_step_data['Returns']) if 'Returns' in current_step_data else 0.0,  # Returns
            float(current_step_data['Volatility']) if 'Volatility' in current_step_data else 0.0,  # Volatility
            float(current_step_data['ATR']) if 'ATR' in current_step_data else 0.0,  # ATR
            self.balance,  # Current balance
            self.holdings,  # Current holdings
            self.portfolio_value  # Current portfolio value
        ], dtype=np.float32)
        
        return obs
    
def test_scenario(env, strategy_name, risk_factor=2.0):
    """Test a trading strategy in the environment"""
    state = env.reset()
    done = False
    total_reward = 0
    prediction_errors = []
    
    while not done:
        # Simple strategy implementations
        if strategy_name == "Aggressive Trading":
            # In bear market, be more aggressive with shorts
            if state[-3] > 0:  # If we have cash balance
                action = 0  # Strong Sell
            else:
                action = 2  # Hold
                
        elif strategy_name == "Contrarian Trading":
            # Buy on significant drops, sell on any rise
            price_change = env.df.iloc[env.current_step]['Returns']
            if price_change < -0.02:  # Significant drop
                action = 4  # Strong Buy
            elif price_change > 0.01:  # Any rise
                action = 0  # Strong Sell
            else:
                action = 2  # Hold
                
        elif strategy_name == "Volatility Trading":
            # Trade based on volatility in bear market
            vol = env.df['Close'].rolling(window=5).std().iloc[env.current_step]
            avg_vol = env.df['Close'].rolling(window=20).std().iloc[env.current_step]
            
            if vol > avg_vol * 1.5:  # High volatility
                action = 0  # Strong Sell
            elif vol < avg_vol * 0.5:  # Low volatility
                action = 3  # Buy
            else:
                action = 2  # Hold
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        # Track metrics
        total_reward += reward if not np.isnan(reward) else 0
        prediction_errors.append(info['prediction_error'])
        
        # Print step information
        print(f"\nStep {env.current_step}/{len(env.df)-1}")
        print(f"Action: {['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy'][action]}")
        print(f"Reward: {reward:.2f}")
        print(f"Prediction Error: {info['prediction_error']:.6f}")
        print(f"Portfolio Value: {info['portfolio_value']:.2f}")
        print(f"Holdings: {info['holdings']}")
        print(f"Balance: {info['balance']:.2f}")
        print(f"Return: {info['return']*100:.2f}%")
        
        state = next_state
    
    # Print final results
    print("\nScenario Results:")
    print(f"Total Reward: {total_reward}")
    print(f"Average Prediction Error: {np.mean(prediction_errors):.6f}")
    print(f"Final Portfolio Value: {info['portfolio_value']:.2f}")
    print(f"Total Return: {info['return']*100:.2f}%")
    print(f"Max Drawdown: {info['drawdown']*100:.2f}%")
    
    return total_reward, np.mean(prediction_errors), info['return'], info['drawdown']

if __name__ == "__main__":
    print("Loading stock data...\n")
    # Load your stock data here (e.g., using yfinance or pandas_datareader)
    # For this example, we'll create sample data
    dates = pd.date_range(start='2020-01-01', end='2022-12-31')
    data = pd.DataFrame(index=dates)
    
    # Use APTV stock data as an example (you can change this to any stock)
    ticker = 'APTV'
    data = yf.download(ticker, start='2020-01-01', end='2022-12-31')
    
    print("Preprocessing data...\n")
    # Add technical indicators
    data['Returns'] = data['Close'].pct_change()
    data['Rolling_Returns'] = data['Returns'].rolling(window=20).sum()
    data['Volume_Change'] = data['Volume'].pct_change()
    
    # Normalize price data
    for col in ['Open', 'High', 'Low', 'Close']:
        data[col] = data[col] / data[col].max()
    
    print("Identifying bear market periods...\n")
    # Find a significant bear market period (20-day window with lowest returns)
    window_size = 20
    rolling_returns = data['Rolling_Returns'].rolling(window=window_size).sum()
    best_start_idx = rolling_returns.argmin() - window_size
    
    # Extract 30 days of bear market data
    bear_market_data = data.iloc[best_start_idx:best_start_idx+30].copy()
    
    print(f"Best performing stock for bear market: {ticker}\n")
    print("Bear Market Period:")
    print(f"Start Date: {bear_market_data.index[0].strftime('%Y-%m-%d')}")
    print(f"End Date: {bear_market_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Total Return: {bear_market_data['Returns'].sum()*100:.2f}%\n")
    
    print(f"Bear Market Data Shape: {bear_market_data.shape}\n")
    print(f"Features: {list(bear_market_data.columns)}\n")
    print("First few rows of bear market data:")
    print(bear_market_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Rolling_Returns', 'Volume_Change']].head())
    print("\nInitializing environment with bear market data...")
    
    # Create and test environment
    env = StockTradingEnv(bear_market_data)
    print("Environment initialized\n")
    
    print("Testing Aggressive Buy and Hold strategy...\n")
    
    # Test different strategies
    strategies = ["Aggressive Trading", "Contrarian Trading", "Volatility Trading"]
    results = {}
    
    for strategy in strategies:
        print(f"\n=== Testing Scenario: {strategy} ===\n")
        reward, pred_error, returns, drawdown = test_scenario(env, strategy)
        results[strategy] = {
            'reward': reward,
            'pred_error': pred_error,
            'returns': returns,
            'drawdown': drawdown
        }
    
    # Compare strategies
    print("\n=== Bear Market Strategy Comparison ===")
    for strategy, metrics in results.items():
        print(f"{strategy}:")
        print(f"- Total Reward: {metrics['reward']}")
        print(f"- Final Return: {metrics['returns']*100:.2f}%")
        print(f"- Max Drawdown: {metrics['drawdown']*100:.2f}%\n")
    
    print("=== Prediction Error Analysis in Bear Market ===")
    for strategy, metrics in results.items():
        print(f"{strategy} Avg Error: {metrics['pred_error']:.6f}")
