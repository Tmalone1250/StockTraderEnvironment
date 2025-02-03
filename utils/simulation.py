"""Simulation utilities"""

import numpy as np
import pandas as pd

def run_simulation(strategy_func, df, initial_capital=10000.0):
    """
    Run trading simulation with enhanced metrics and risk management
    """
    print(f"\nStarting simulation with {strategy_func.__name__}...")
    print(f"Initial portfolio value: ${initial_capital:.2f}")
    
    # Initialize portfolio metrics
    portfolio_values = np.full(len(df), np.nan)
    returns = np.full(len(df), np.nan)
    holdings = np.full(len(df), np.nan)
    position_sizes = np.full(len(df), np.nan)
    cash = np.full(len(df), np.nan)
    
    # Set initial values
    portfolio_values[0] = initial_capital
    cash[0] = initial_capital
    returns[0] = 0.0
    
    # Initialize strategy state
    from strategies.base import State
    state = State()
    
    # Run simulation
    for step in range(len(df)):
        try:
            current_price = df['Close'].iloc[step]
            
            # Get position size from strategy
            position_size, state = strategy_func(state, df, step)
            position_sizes[step] = position_size
            
            # Calculate holdings value
            if step == 0:
                holdings[step] = position_size * initial_capital
                cash[step] = initial_capital - holdings[step]
            else:
                # Update holdings based on price change
                price_change = current_price / df['Close'].iloc[step-1]
                holdings[step] = holdings[step-1] * price_change
                
                # If position size changed, adjust holdings and cash
                if position_sizes[step] != position_sizes[step-1]:
                    # Close out old position
                    cash[step] = cash[step-1] + holdings[step]
                    # Enter new position
                    holdings[step] = cash[step] * position_sizes[step]
                    cash[step] = cash[step] - holdings[step]
                else:
                    cash[step] = cash[step-1]
            
            # Calculate total portfolio value
            portfolio_values[step] = cash[step] + holdings[step]
            
            # Calculate returns
            if step > 0:
                returns[step] = (portfolio_values[step] / portfolio_values[step-1] - 1)
            
            # Print progress
            if step % 20 == 0:
                print(f"Step {step}: Portfolio value: ${portfolio_values[step]:.2f}")
            
        except Exception as e:
            print(f"Error in simulation step {step}: {str(e)}")
            if step > 0:
                portfolio_values[step] = portfolio_values[step-1]
                holdings[step] = holdings[step-1]
                cash[step] = cash[step-1]
                position_sizes[step] = position_sizes[step-1]
                returns[step] = 0.0
    
    # Convert arrays to pandas Series
    portfolio_values = pd.Series(portfolio_values, index=df.index)
    returns_series = pd.Series(returns, index=df.index)
    
    return portfolio_values, returns_series
