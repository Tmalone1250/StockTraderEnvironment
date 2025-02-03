# Install required libraries
# !pip install gymnasium numpy stable-baselines3 pandas shimmy torch

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

class ForwardDynamicsModel(nn.Module):
    """Predicts next state given current state and action"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ForwardDynamicsModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        
    def _process_inputs(self, state, action):
        """Process inputs to ensure correct tensor dimensions"""
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        
        # Ensure state has batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Create one-hot encoded action tensor
        action_one_hot = torch.zeros((state.size(0), self.action_dim))
        action_one_hot[:, action] = 1
        
        return state, action_one_hot
        
    def forward(self, state, action):
        # Process inputs
        state, action_one_hot = self._process_inputs(state, action)
        
        # Concatenate along feature dimension
        x = torch.cat([state, action_one_hot], dim=1)
        return self.network(x)
    
    def update(self, state, action, next_state):
        """Train the forward dynamics model"""
        self.optimizer.zero_grad()
        
        # Process inputs
        state, action_one_hot = self._process_inputs(state, action)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.FloatTensor(next_state)
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)
        
        # Predict next state
        predicted_next_state = self.network(torch.cat([state, action_one_hot], dim=1))
        
        # Calculate loss
        loss = self.loss_fn(predicted_next_state, next_state)
        
        # Backpropagate and update
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def get_prediction_error(self, state, action, next_state):
        """Calculate prediction error (intrinsic reward)"""
        with torch.no_grad():
            # Process inputs
            state, action_one_hot = self._process_inputs(state, action)
            if not isinstance(next_state, torch.Tensor):
                next_state = torch.FloatTensor(next_state)
            if next_state.dim() == 1:
                next_state = next_state.unsqueeze(0)
            
            # Get prediction and calculate error
            predicted_next_state = self.network(torch.cat([state, action_one_hot], dim=1))
            prediction_error = F.mse_loss(predicted_next_state, next_state)
            
        return prediction_error.item()

# Step 1: Define the Stock Trading Environment
class StockTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=100000):
        print("Initializing StockTradingEnv...")
        super(StockTradingEnv, self).__init__()
        
        # Define action and observation space
        print("Setting up action and observation spaces...")
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        print(f"Action space: {self.action_space}")
        print(f"Observation space: {self.observation_space}")
        
        # Initialize forward dynamics model for curiosity-driven learning
        print("Initializing forward dynamics model...")
        self.forward_model = ForwardDynamicsModel(
            state_dim=10,  # Observation space dimension
            action_dim=3   # Action space dimension
        )
        print("Forward dynamics model initialized")
        
        # Curiosity parameters
        self.curiosity_weight = 0.1  # Weight for intrinsic reward
        self.curiosity_decay = 0.9999  # Decay factor for curiosity weight
        
        # Stock data
        print("Processing stock data...")
        self.data = data
        self.current_step = 0
        
        # Calculate daily volume moving average for volume constraints
        print("Calculating volume moving average...")
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=20).mean().bfill()
        
        # Trading parameters
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.prev_net_worth = initial_balance
        
        # Transaction costs
        self.commission_rate = 0.001  # 0.1% commission per trade
        self.slippage_rate = 0.0005   # 0.05% slippage per trade
        
        # Maximum trading volume (as a fraction of daily volume)
        self.max_volume_fraction = 0.1  # Maximum 10% of daily volume
        
        # For reward normalization
        self.returns_history = []
        self.reward_scaling = 1.0
        
        # For tracking prediction errors
        self.prediction_errors = []
        print("StockTradingEnv initialization completed")

    def calculate_transaction_costs(self, shares, price):
        """Calculate total transaction costs including commission and slippage"""
        transaction_value = shares * price
        commission = transaction_value * self.commission_rate
        slippage = transaction_value * self.slippage_rate
        return commission + slippage

    def get_max_shares_possible(self, price, daily_volume):
        """Calculate maximum shares that can be traded based on volume constraints"""
        volume_constraint = daily_volume * self.max_volume_fraction
        balance_constraint = self.balance / (price * (1 + self.commission_rate + self.slippage_rate))
        return min(volume_constraint, balance_constraint)

    def reset(self, seed=None):
        # Reset the state
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.current_step = 0
        self.returns_history = []
        self.prediction_errors = []
        return self._next_observation(), {}

    def _next_observation(self):
        # Get the current stock data
        current_data = self.data.iloc[self.current_step]
        
        # Calculate price changes for better feature representation
        price_change = current_data['Close'] / current_data['Open'] - 1
        volume_change = current_data['Volume'] / current_data['Volume_MA'] - 1
        
        obs = np.array([
            current_data['Open'],
            current_data['High'],
            current_data['Low'],
            current_data['Close'],
            current_data['Volume'],
            price_change,
            volume_change,
            float(current_data['Name']),
            self.shares_held,
            self.balance
        ], dtype=np.float32)
        return obs

    def step(self, action):
        try:
            print(f"\nExecuting step with action {action}")
            # Store current state for curiosity calculation
            current_state = self._next_observation()
            
            # Get current price and volume data
            current_data = self.data.iloc[self.current_step]
            current_price = current_data['Close']
            daily_volume = current_data['Volume']
            print(f"Current price: {current_price}, Daily volume: {daily_volume}")
            
            # Store previous net worth for reward calculation
            self.prev_net_worth = self.net_worth
            
            # Execute action
            if action == 1:  # Buy
                print("Attempting to buy...")
                max_shares = self.get_max_shares_possible(current_price, daily_volume)
                if max_shares > 0:
                    # Calculate transaction costs
                    transaction_costs = self.calculate_transaction_costs(max_shares, current_price)
                    total_cost = (max_shares * current_price) + transaction_costs
                    
                    if total_cost <= self.balance:
                        self.shares_held += max_shares
                        self.balance -= total_cost
                        print(f"Bought {max_shares:.2f} shares at {current_price:.2f}")
                    
            elif action == 2:  # Sell
                print("Attempting to sell...")
                if self.shares_held > 0:
                    # Limit sell volume
                    sell_shares = min(
                        self.shares_held,
                        daily_volume * self.max_volume_fraction
                    )
                    
                    # Calculate transaction costs
                    transaction_costs = self.calculate_transaction_costs(sell_shares, current_price)
                    
                    # Execute sell
                    self.balance += (sell_shares * current_price) - transaction_costs
                    self.shares_held -= sell_shares
                    print(f"Sold {sell_shares:.2f} shares at {current_price:.2f}")

            # Update current step and net worth
            self.current_step += 1
            self.net_worth = self.balance + self.shares_held * current_price
            print(f"Updated net worth: {self.net_worth:.2f}")
            
            # Get next state
            next_state = self._next_observation()
            
            print("Calculating prediction error...")
            # Update forward dynamics model and get intrinsic reward
            prediction_error = self.forward_model.get_prediction_error(
                current_state, action, next_state
            )
            self.forward_model.update(current_state, action, next_state)
            self.prediction_errors.append(prediction_error)
            print(f"Prediction error: {prediction_error}")
            
            # Calculate extrinsic reward (normalized returns)
            returns = (self.net_worth - self.prev_net_worth) / self.prev_net_worth
            self.returns_history.append(returns)
            
            if len(self.returns_history) > 1:
                # Normalize reward using recent returns history
                self.reward_scaling = 1.0 / (np.std(self.returns_history[-100:]) + 1e-6)
            
            # Combine extrinsic and intrinsic rewards
            extrinsic_reward = returns * self.reward_scaling
            intrinsic_reward = prediction_error * self.curiosity_weight
            reward = extrinsic_reward + intrinsic_reward
            print(f"Rewards - Extrinsic: {extrinsic_reward:.4f}, Intrinsic: {intrinsic_reward:.4f}, Total: {reward:.4f}")
            
            # Decay curiosity weight
            self.curiosity_weight *= self.curiosity_decay

            # Check if done
            done = self.current_step >= len(self.data) - 1
            truncated = False

            # Add information about the trade
            info = {
                'net_worth': self.net_worth,
                'balance': self.balance,
                'shares_held': self.shares_held,
                'returns': returns,
                'reward_scaling': self.reward_scaling,
                'extrinsic_reward': extrinsic_reward,
                'intrinsic_reward': intrinsic_reward,
                'prediction_error': prediction_error,
                'curiosity_weight': self.curiosity_weight
            }
            
            print("Step completed successfully")
            return next_state, reward, done, truncated, info
            
        except Exception as e:
            print(f"Error in step function: {str(e)}")
            print(f"Current step: {self.current_step}")
            print(f"Data shape: {self.data.shape}")
            raise e

# Step 2: Load stock data
print("\nLoading stock data...")
data = pd.read_csv('stock_data.csv')  # Replace with your stock data file
print("Data loaded successfully:")
print("\nFirst few rows of the data:")
print(data.head())
print("\nData shape:", data.shape)
print("Columns:", data.columns.tolist())

print("\nProcessing dates and calculating volume moving average...")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Encode stock names as numerical values (required for the state space)
data['Name'] = pd.factorize(data['Name'])[0]  # Convert stock names to unique integers

# Step 3: Create the environment
print("\nInitializing trading environment...")
env = StockTradingEnv(data)
print("Environment initialized successfully")

# Step 4: Initialize the RL agent
print("\nInitializing DQN agent...")
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0001,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
)
print("DQN agent initialized successfully")

# Step 5: Train the agent
print("\nStarting training...")
total_timesteps = 100000  # Total number of training steps
model.learn(total_timesteps=total_timesteps)
print("Training completed")

# Step 6: Evaluate the agent
print("\nStarting evaluation...")
eval_env = StockTradingEnv(data)
print("Getting initial observation...")
obs, _ = eval_env.reset()
print(f"Initial observation shape: {obs.shape}")
done = False
truncated = False
total_reward = 0
episode_steps = 0

print("\nStarting evaluation loop...")
try:
    while not (done or truncated):
        print(f"\nStep {episode_steps + 1}")
        print("Predicting action...")
        action, _states = model.predict(obs)
        print(f"Predicted action: {action}")
        
        print("Executing step...")
        obs, reward, done, truncated, info = eval_env.step(action)
        total_reward += reward
        episode_steps += 1
        
        print(f"Step {episode_steps} completed - Reward: {reward:.4f}, Total: {total_reward:.4f}")
        print(f"Portfolio value: {info['net_worth']:.2f}")
except Exception as e:
    print(f"Error during evaluation: {str(e)}")
    raise e

print(f"\nEvaluation Results:")
print(f"Total Steps: {episode_steps}")
print(f"Total Profit: {total_reward}")
print(f"Final Portfolio Value: {info['net_worth']}")
print(f"Final Balance: {info['balance']}")
print(f"Final Shares Held: {info['shares_held']}")