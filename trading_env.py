import numpy as np
import pandas as pd

class TradingEnv:
    def __init__(self, df, max_leverage=2.0):
        self.df = df
        self.max_leverage = max_leverage
        self.reset()
        
    def reset(self):
        self.current_step = 0
        self.balance = 10000.0
        self.holdings = 0
        self.portfolio_values = [self.balance]
        self.portfolio_value = self.balance
        self.max_drawdown = 0
        self.peak_value = self.balance
        return self.get_state()
        
    def get_state(self):
        current_price = self.df['Close'].iloc[self.current_step]
        returns = self.df['Close'].pct_change().iloc[self.current_step]
        volatility = self.df['Close'].pct_change().rolling(window=20).std().iloc[self.current_step]
        rsi = self.df['RSI'].iloc[self.current_step]
        portfolio_return = (self.portfolio_value - self.portfolio_values[-2]) / self.portfolio_values[-2] if len(self.portfolio_values) > 1 else 0
        
        return np.array([
            current_price,
            returns,
            volatility,
            rsi,
            portfolio_return,
            self.holdings,
            self.portfolio_value,
            self.balance
        ])
        
    def step(self, action):
        current_price = self.df['Close'].iloc[self.current_step]
        
        # Calculate position value and current leverage
        position_value = self.holdings * current_price
        current_leverage = abs(position_value) / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # Define position sizes for different actions
        position_sizes = {
            0: -0.5,  # Strong Sell: Reduce position by 50%
            1: -0.25, # Sell: Reduce position by 25%
            2: 0,     # Hold: No change
            3: 0.25,  # Buy: Increase position by 25%
            4: 0.5,   # Strong Buy: Increase position by 50%
            5: 1.0    # Maximum Leverage: Full position
        }
        
        # Calculate target position change
        position_change = position_sizes[action]
        
        if action == 5:  # Maximum leverage
            target_holdings = (self.max_leverage * self.portfolio_value) / current_price
            shares_to_trade = target_holdings - self.holdings
        else:
            # Calculate shares to trade based on portfolio value and position change
            shares_to_trade = int((self.portfolio_value * position_change) / current_price)
        
        # Apply transaction costs (0.1%)
        transaction_cost = abs(shares_to_trade * current_price * 0.001)
        
        # Update holdings and balance
        self.holdings += shares_to_trade
        self.balance -= (shares_to_trade * current_price + transaction_cost)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # Update portfolio value
        self.portfolio_value = self.balance + (self.holdings * current_price)
        self.portfolio_values.append(self.portfolio_value)
        
        # Update maximum drawdown
        self.peak_value = max(self.peak_value, self.portfolio_value)
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Calculate reward (risk-adjusted returns)
        reward = (self.portfolio_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
        reward = reward * (1 - self.max_drawdown)  # Penalize for drawdown
        
        return self.get_state(), reward, done, {}
