"""Mean Reversion Strategy"""

from ..base import State

def mean_reversion_strategy(state, df, current_step):
    """
    Mean reversion strategy with trend following elements
    Uses RSI for mean reversion signals and MACD for trend confirmation
    """
    try:
        if current_step < 26:  # Need enough data for indicators
            return 0.0, state
            
        # Get current indicators
        rsi = df['RSI'].iloc[current_step]
        macd = df['MACD'].iloc[current_step]
        signal = df['Signal'].iloc[current_step]
        close = df['Close'].iloc[current_step]
        volatility = df['Volatility'].iloc[current_step]
        
        # Base position size on volatility
        base_position = 1.0
        vol_adjust = 0.3 / volatility if volatility > 0 else 1.0
        position_size = base_position * min(vol_adjust, 1.0)
        
        # Trend direction from MACD
        trend_bullish = macd > signal
        trend_bearish = macd < signal
        
        # Mean reversion signals
        oversold = rsi < 30
        overbought = rsi > 70
        
        # Position sizing based on signals
        if oversold and trend_bullish:
            # Strong buy signal
            final_position = position_size
        elif overbought and trend_bearish:
            # Strong sell signal
            final_position = -position_size
        elif oversold:
            # Moderate buy signal
            final_position = position_size * 0.5
        elif overbought:
            # Moderate sell signal
            final_position = -position_size * 0.5
        else:
            # Hold current position
            final_position = state.position_size if state.position_size is not None else 0.0
        
        # Update state
        state.position_size = final_position
        state.entry_price = close
        
        return final_position, state
        
    except Exception as e:
        print(f"Error in strategy: {str(e)}")
        return 0.0, state  # Return to neutral position on error
