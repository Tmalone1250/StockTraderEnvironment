"""Simple trading strategies"""

from .base import State

def buy_and_hold_strategy(state, df, current_step):
    """
    Simple buy-and-hold strategy that maintains a constant full position
    """
    try:
        # Always maintain a full position
        position_size = 1.0
        
        # Update state
        state.position_size = position_size
        state.entry_price = df['Close'].iloc[current_step]
        
        return position_size, state
        
    except Exception as e:
        print(f"Error in strategy: {str(e)}")
        return 1.0, state  # Return to neutral position on error
