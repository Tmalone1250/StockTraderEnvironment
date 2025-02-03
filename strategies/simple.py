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

def moving_average_crossover_strategy(state, df, current_step):
    """
    Enhanced Moving Average Crossover Strategy with Strong Trend Confirmation
    - Buy when 10-day MA crosses above 30-day MA with strong momentum
    - Hold positions longer when trend is confirmed
    - Exit only on clear trend reversal signals
    - Aggressive position sizing in strong trends
    """
    try:
        if current_step < 30:  # Need enough data for 30-day MA
            return 0.0, state
            
        # Get current indicators
        current_signal = df['MA_Signal'].iloc[current_step]
        prev_signal = df['MA_Signal'].iloc[current_step - 1]
        close = df['Close'].iloc[current_step]
        volatility = df['Volatility'].iloc[current_step]
        rsi = df['RSI'].iloc[current_step]
        roc = df['ROC'].iloc[current_step]
        
        # Calculate trend strength
        ma10 = df['MA10'].iloc[current_step]
        ma30 = df['MA30'].iloc[current_step]
        trend_strength = abs((ma10 / ma30) - 1)
        
        # Strong trend confirmation
        strong_uptrend = (
            roc > 2 and  # Positive momentum
            rsi > 45 and  # Not oversold
            ma10 > ma30 and  # Above longer MA
            close > ma10  # Price above short MA
        )
        
        strong_downtrend = (
            roc < -2 and  # Negative momentum
            rsi < 55 and  # Not overbought
            ma10 < ma30 and  # Below longer MA
            close < ma10  # Price below short MA
        )
        
        # Base position size on volatility
        base_position = 1.0
        vol_adjust = 0.3 / volatility if volatility > 0 else 1.0
        position_size = base_position * min(vol_adjust, 1.0)
        
        # Aggressive position sizing in strong trends
        trend_multiplier = 1.0
        if trend_strength > 0.03:  # Strong trend
            trend_multiplier = 2.0
        if trend_strength > 0.06:  # Very strong trend
            trend_multiplier = 3.0
        
        position_size *= trend_multiplier
        position_size = min(position_size, 1.0)  # Cap at 100%
        
        # Position management with strong trend confirmation
        if current_signal > prev_signal and strong_uptrend:
            # New bullish trend with confirmation
            final_position = position_size
        elif current_signal < prev_signal and strong_downtrend:
            # New bearish trend with confirmation
            final_position = 0.0
        elif state.position_size is not None and state.position_size > 0:
            # Managing existing long position
            if strong_downtrend:
                final_position = 0.0  # Exit on strong downtrend
            else:
                final_position = state.position_size  # Hold position
        else:
            # No position and no new signal
            final_position = 0.0
        
        # Update state
        state.position_size = final_position
        state.entry_price = close
        
        return final_position, state
        
    except Exception as e:
        print(f"Error in strategy: {str(e)}")
        return 0.0, state  # Return to neutral position on error

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
