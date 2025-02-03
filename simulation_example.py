import yfinance as yf
import pandas as pd
import numpy as np
from StockTradingEnv import StockTradingEnv

def prepare_data(symbol='AAPL', start_date='2023-01-01', end_date='2023-12-31'):
    """Download and prepare stock data with technical indicators"""
    print(f"Downloading {symbol} data from {start_date} to {end_date}...")
    df = yf.download(symbol, start=start_date, end=end_date)
    
    # Calculate technical indicators
    df['Returns'] = df['Close'].pct_change()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Calculate volatility
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # Remove any rows with NaN values
    df.dropna(inplace=True)
    
    return df

def simple_trading_strategy(state):
    """
    A simple trading strategy based on state observations
    Returns: 0 (Strong Sell), 1 (Sell), 2 (Hold), 3 (Buy), or 4 (Strong Buy)
    """
    # Extract relevant features from state
    current_price = state[0]  # Assuming first state component is price
    sma_20 = state[1]        # Assuming second component is SMA_20
    sma_50 = state[2]        # Assuming third component is SMA_50
    volatility = state[3]    # Assuming fourth component is volatility
    
    # Simple trend-following strategy
    if current_price > sma_20 and sma_20 > sma_50:
        return 4 if volatility < 0.02 else 3  # Strong Buy or Buy based on volatility
    elif current_price < sma_20 and sma_20 < sma_50:
        return 0 if volatility < 0.02 else 1  # Strong Sell or Sell based on volatility
    else:
        return 2  # Hold

def run_simulation(symbol='AAPL', start_date='2023-01-01', end_date='2023-12-31', 
                  initial_balance=10000, max_steps=None):
    """Run a trading simulation using a simple strategy"""
    
    # Prepare data
    df = prepare_data(symbol, start_date, end_date)
    
    # Create environment
    env = StockTradingEnv(df, initial_balance=initial_balance, live_trading=False)
    
    # Run one episode
    state = env.reset()
    done = False
    steps = 0
    
    print("\nStarting simulation...")
    print(f"Initial portfolio value: ${env.portfolio_value:.2f}")
    
    while not done:
        # Get action from strategy
        action = simple_trading_strategy(state)
        
        # Take action
        state, reward, done, info = env.step(action)
        steps += 1
        
        # Print progress every 20 steps
        if steps % 20 == 0:
            print(f"Step {steps}: Portfolio value: ${env.portfolio_value:.2f}")
        
        # Optional step limit
        if max_steps and steps >= max_steps:
            break
    
    # Print final results
    print("\nSimulation finished!")
    print(f"Final portfolio value: ${env.portfolio_value:.2f}")
    print(f"Total return: {((env.portfolio_value - initial_balance) / initial_balance * 100):.2f}%")
    print(f"Number of steps: {steps}")
    
    return env

if __name__ == "__main__":
    # Run simulation
    env = run_simulation(
        symbol='AAPL',
        start_date='2023-01-01',
        end_date='2023-12-31',
        initial_balance=10000,
        max_steps=100  # Limit to 100 steps for testing
    )
    
    # Plot results if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(env.portfolio_values)
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Trading Steps')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.savefig('portfolio_performance.png')
        plt.close()
        print("\nPortfolio performance plot saved as 'portfolio_performance.png'")
    except ImportError:
        print("\nMatplotlib not installed. Skipping performance plot.")
