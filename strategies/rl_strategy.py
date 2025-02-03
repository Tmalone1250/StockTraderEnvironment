"""Reinforcement Learning Enhanced Trading Strategy"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from .base import State

class DQN(nn.Module):
    """Deep Q-Network architecture"""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.dropout1 = nn.Dropout(0.15)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.15)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    """Deep Q-Network Agent for trading"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Initialize networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_counter = 0
        
    def memorize(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, training=True):
        """Choose action based on epsilon-greedy policy"""
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            act_values = self.model(state)
            return torch.argmax(act_values).item()
        
    def replay(self, batch_size):
        """Train on batch from replay memory"""
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([i[0] for i in minibatch]).squeeze(1).to(self.device)
        actions = torch.LongTensor([i[1] for i in minibatch]).to(self.device)
        rewards = torch.FloatTensor([i[2] for i in minibatch]).to(self.device)
        next_states = torch.FloatTensor([i[3] for i in minibatch]).squeeze(1).to(self.device)
        dones = torch.FloatTensor([i[4] for i in minibatch]).to(self.device)
        
        # Current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        # Compute loss and update
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_model(self):
        """Update target network"""
        self.target_model.load_state_dict(self.model.state_dict())

class RLState(State):
    """Extended state class to handle RL agent state"""
    def __init__(self):
        super().__init__()
        self.state_size = 20  # Number of features in state
        self.action_size = 3  # Buy, Sell, Hold
        self.agent = DQNAgent(self.state_size, self.action_size)
        self.batch_size = 32
        self.update_target_every = 5
        self.min_samples_before_training = 100
        self.reward_scaling = 100  # Scale rewards for better learning
        self.current_drawdown = 0.0  # Initialize drawdown
        self.max_portfolio_value = 0.0  # Track max portfolio value
        
    def update_risk_metrics(self, current_price, portfolio_value, current_step):
        """Update risk metrics including drawdown"""
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
        if self.max_portfolio_value > 0:
            self.current_drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
        super().update_risk_metrics(current_price, portfolio_value, current_step)

    def get_state(self, df, current_step):
        """Create state representation from market data"""
        try:
            # Calculate volume momentum using rolling means
            vol_5 = df['Volume'].rolling(window=5).mean().iloc[current_step]
            vol_10 = df['Volume'].rolling(window=10).mean().iloc[current_step]
            
            state = [
                df['BB_PCT'].iloc[current_step],
                df['BB_Width'].iloc[current_step],
                df['RSI'].iloc[current_step] / 100,
                df['ROC_MA'].iloc[current_step] / 100,
                df['MACD'].iloc[current_step],
                df['Signal'].iloc[current_step],
                df['Trend_Strength'].iloc[current_step],
                df['Strong_Trend'].iloc[current_step],
                df['Volatility'].iloc[current_step],
                df['ATR'].iloc[current_step] / df['Close'].iloc[current_step],
                1 if df['MA10'].iloc[current_step] > df['MA30'].iloc[current_step] else 0,
                df['Price_Slope'].iloc[current_step] / 100,
                # Price momentum
                (df['Close'].iloc[current_step] / df['Close'].iloc[max(0, current_step - 5)]) - 1,
                (df['Close'].iloc[current_step] / df['Close'].iloc[max(0, current_step - 10)]) - 1,
                (df['Close'].iloc[current_step] / df['Close'].iloc[max(0, current_step - 20)]) - 1,
                # Volume momentum
                df['Volume'].iloc[current_step] / vol_5 - 1 if vol_5 > 0 else 0,
                df['Volume'].iloc[current_step] / vol_10 - 1 if vol_10 > 0 else 0,
                # Position features
                1.0 if self.position_size else 0.0,
                self.entry_price / df['Close'].iloc[current_step] if self.position_size else 1.0,
                # Current drawdown
                self.current_drawdown
            ]
            return np.array(state).reshape(1, -1)  # Reshape for DQN input
        except Exception as e:
            print(f"Error getting state: {str(e)}")
            # Return zero state if there's an error
            return np.zeros((1, self.state_size))
            
    def calculate_reward(self, df, current_step, action):
        """Calculate reward based on multiple factors including risk-adjusted returns"""
        try:
            # Get price changes and position
            current_price = df['Close'].iloc[current_step]
            prev_price = df['Close'].iloc[current_step - 1]
            price_change = (current_price / prev_price) - 1
            has_position = self.position_size > 0
            
            # 1. Base reward component: Risk-adjusted returns
            if action == 0:  # Buy
                base_reward = price_change * self.reward_scaling
                # Bonus for catching upward momentum
                if df['ROC_MA'].iloc[current_step] > 0:
                    base_reward *= 1.2
            elif action == 1:  # Sell
                base_reward = -price_change * self.reward_scaling
                # Bonus for avoiding downward momentum
                if df['ROC_MA'].iloc[current_step] < 0:
                    base_reward *= 1.2
            else:  # Hold
                base_reward = 0 if not has_position else price_change * self.reward_scaling
            
            # 2. Risk management component
            risk_reward = 0
            
            # Penalize high drawdown
            if self.current_drawdown > 0.05:  # 5% drawdown threshold
                risk_reward -= self.current_drawdown * 50
                
            # Reward for following volatility-based position sizing
            volatility = df['Volatility'].iloc[current_step]
            if action == 0 and volatility < df['Volatility'].iloc[current_step - 1]:
                risk_reward += 0.5  # Bonus for buying in decreasing volatility
            elif action == 1 and volatility > df['Volatility'].iloc[current_step - 1]:
                risk_reward += 0.5  # Bonus for selling in increasing volatility
                
            # 3. Market regime component
            regime_reward = 0
            
            # Trend following rewards
            ma_trend = df['MA10'].iloc[current_step] > df['MA30'].iloc[current_step]
            if action == 0 and ma_trend:  # Buying in uptrend
                regime_reward += 1.0
            elif action == 1 and not ma_trend:  # Selling in downtrend
                regime_reward += 1.0
                
            # Trend strength consideration
            if df['Strong_Trend'].iloc[current_step]:
                if (action == 0 and ma_trend) or (action == 1 and not ma_trend):
                    regime_reward *= 1.5
                    
            # 4. Technical analysis component
            tech_reward = 0
            
            # RSI signals
            rsi = df['RSI'].iloc[current_step]
            if action == 0 and rsi < 30:  # Buying oversold
                tech_reward += 1.0
            elif action == 1 and rsi > 70:  # Selling overbought
                tech_reward += 1.0
                
            # Bollinger Bands signals
            bb_pct = df['BB_PCT'].iloc[current_step]
            if action == 0 and bb_pct < -0.8:  # Buying near lower band
                tech_reward += 1.0
            elif action == 1 and bb_pct > 0.8:  # Selling near upper band
                tech_reward += 1.0
                
            # MACD signals
            macd_cross = (
                df['MACD'].iloc[current_step] > df['Signal'].iloc[current_step] and
                df['MACD'].iloc[current_step - 1] <= df['Signal'].iloc[current_step - 1]
            )
            if action == 0 and macd_cross:  # Buying on MACD cross up
                tech_reward += 1.0
            elif action == 1 and not macd_cross:  # Selling on MACD cross down
                tech_reward += 1.0
                
            # 5. Position management component
            pos_reward = 0
            
            # Reward for position consistency
            if len(self.agent.memory) > 0:
                last_action = self.agent.memory[-1][1]
                # Penalize frequent position changes
                if action != last_action:
                    pos_reward -= 0.2
                # But reward for holding good positions
                elif action == last_action and base_reward > 0:
                    pos_reward += 0.3
                    
            # Reward for proper position sizing
            if has_position:
                # Reward for having larger positions in favorable conditions
                if df['Strong_Trend'].iloc[current_step] and ma_trend:
                    pos_reward += self.position_size * 0.5
                # Penalize for having large positions in unfavorable conditions
                elif not df['Strong_Trend'].iloc[current_step] or not ma_trend:
                    pos_reward -= self.position_size * 0.3
                    
            # 6. Volume analysis component
            vol_reward = 0
            
            # Volume trend
            current_volume = df['Volume'].iloc[current_step]
            prev_volume = df['Volume'].iloc[current_step - 1]
            vol_change = current_volume / prev_volume if prev_volume > 0 else 1.0
            
            if action == 0 and vol_change > 1.2:  # Buying with increasing volume
                vol_reward += 0.5
            elif action == 1 and vol_change < 0.8:  # Selling with decreasing volume
                vol_reward += 0.5
                
            # Combine all reward components with weights
            total_reward = (
                base_reward * 1.0 +      # Primary importance
                risk_reward * 0.8 +      # High importance for risk management
                regime_reward * 0.6 +    # Medium importance for market regime
                tech_reward * 0.4 +      # Lower importance for technical signals
                pos_reward * 0.3 +       # Lower importance for position management
                vol_reward * 0.2         # Lowest importance for volume analysis
            )
            
            # Apply dynamic reward scaling based on prediction accuracy
            if len(self.agent.memory) >= 10:
                recent_rewards = [x[2] for x in self.agent.memory[-10:]]
                accuracy = np.mean([1 if r > 0 else 0 for r in recent_rewards])
                # Scale rewards based on recent performance
                total_reward *= (0.5 + accuracy)  # Scale between 0.5x and 1.5x
                
            return total_reward
            
        except Exception as e:
            print(f"Error calculating reward: {str(e)}")
            return 0

def rl_enhanced_strategy(state, df, current_step):
    """
    Reinforcement Learning Enhanced Trading Strategy
    - Uses Deep Q-Network for dynamic strategy adaptation
    - State space includes technical indicators and market conditions
    - Actions: Buy (1.0), Sell (0.0), Hold (current position)
    - Rewards based on PnL and risk-adjusted metrics
    """
    try:
        if not isinstance(state, RLState):
            state = RLState()
            
        if current_step < 20:  # Need enough data for features
            return 0.0, state
            
        # Get current indicators
        close = df['Close'].iloc[current_step]
        volatility = df['Volatility'].iloc[current_step]
        
        # Update risk metrics
        portfolio_value = close * state.position_size if state.position_size else close
        state.update_risk_metrics(close, portfolio_value, current_step)
        
        # Check if we should exit based on risk management
        if state.should_exit_trade(close, volatility):
            return 0.0, state
            
        # Get current state
        current_state = state.get_state(df, current_step)
        
        # Choose action
        action = state.agent.act(current_state, training=True)
        
        # Convert action to position size
        if action == 0:  # Buy
            final_position = state.get_position_size(1.0, volatility)
        elif action == 1:  # Sell
            final_position = 0.0
        else:  # Hold
            final_position = state.position_size if state.position_size else 0.0
            
        # Calculate reward
        reward = state.calculate_reward(df, current_step, action)
        
        # Get next state if possible
        if current_step < len(df) - 1:
            next_state = state.get_state(df, current_step + 1)
        else:
            next_state = current_state  # Use current state if at end of data
            
        # Store experience in memory
        state.agent.memorize(current_state, action, reward, next_state, current_step == len(df) - 1)
        
        # Train the network
        if len(state.agent.memory) > state.min_samples_before_training:
            state.agent.replay(state.batch_size)
            
            # Update target network periodically
            state.agent.update_target_counter += 1
            if state.agent.update_target_counter >= state.update_target_every:
                state.agent.update_target_model()
                state.agent.update_target_counter = 0
                
        # Update position size
        state.position_size = final_position
        if final_position > 0:
            state.entry_price = close
            
        return final_position, state
        
    except Exception as e:
        print(f"Error in RL strategy: {str(e)}")
        return 0.0, state
