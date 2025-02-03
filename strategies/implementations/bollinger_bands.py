"""Bollinger Bands Strategy"""

from ..base import State

def bollinger_bands_strategy(state, df, current_step):
    """
    Aggressive Bollinger Bands Strategy with Enhanced Trend Following
    - Very aggressive entries and position sizing in strong uptrends
    - Uses multiple timeframe trend confirmation
    - Dynamic position sizing based on trend strength and momentum
    - Maintains strict risk management in uncertain conditions
    """
    try:
        if current_step < 30:  # Need enough data for indicators
            return 0.0, state
            
        # Get current indicators
        bb_pct = df['BB_PCT'].iloc[current_step]
        bb_width = df['BB_Width'].iloc[current_step]
        close = df['Close'].iloc[current_step]
        volatility = df['Volatility'].iloc[current_step]
        rsi = df['RSI'].iloc[current_step]
        roc = df['ROC'].iloc[current_step]
        
        # Update risk metrics
        portfolio_value = close * state.position_size if state.position_size else close
        state.update_risk_metrics(close, portfolio_value, current_step)
        
        # Check if we should exit based on risk management
        if state.should_exit_trade(close, volatility):
            return 0.0, state
        
        # Calculate trend strength
        ma10 = df['MA10'].iloc[current_step]
        ma30 = df['MA30'].iloc[current_step]
        trend_strength = abs((ma10 / ma30) - 1)
        
        # Even more aggressive trend confirmation
        uptrend_conditions = [
            ma10 > ma30,  # Primary trend
            close > ma10,  # Price momentum
            roc > -1,  # Any momentum (very lenient)
            rsi > 30,  # Not extremely oversold
            volatility < 0.6  # Allow higher volatility
        ]
        
        downtrend_conditions = [
            ma10 < ma30,  # Primary trend
            close < ma10,  # Price momentum
            roc < 1,  # Any momentum (very lenient)
            rsi < 70,  # Not extremely overbought
            volatility < 0.6  # Allow higher volatility
        ]
        
        # Count how many conditions are met
        uptrend_strength = sum(uptrend_conditions)
        downtrend_strength = sum(downtrend_conditions)
        
        # Base position size calculation - start more aggressive
        base_position = 1.0
        
        # Get risk-adjusted position size
        position_size = state.get_position_size(base_position, volatility)
        
        # More aggressive position scaling based on trend strength
        if trend_strength > 0.01:  # Weak trend
            position_size *= 1.2
        if trend_strength > 0.02:  # Moderate trend
            position_size *= 1.5
        if trend_strength > 0.04:  # Strong trend
            position_size *= 2.0
            
        # Additional position scaling based on RSI
        if 40 <= rsi <= 70:  # Wider sweet spot
            position_size *= 1.3
            
        # Momentum boost
        if abs(roc) > 2:
            position_size *= 1.2
            
        position_size = min(position_size, 1.0)  # Cap at 100%
        
        # Bollinger Band signals
        oversold = bb_pct < -0.8
        overbought = bb_pct > 0.8
        squeeze = bb_width < 0.1
        expansion = bb_width > 0.2
        
        # More aggressive entry conditions
        if oversold and uptrend_strength >= 2:  # Need only 2 conditions
            final_position = position_size
        elif bb_pct < -0.5 and uptrend_strength >= 3:  # Less oversold but stronger trend
            final_position = position_size * 0.8
        elif overbought and downtrend_strength >= 2:  # Need only 2 conditions
            final_position = 0.0
        elif bb_pct > 0.5 and downtrend_strength >= 4:  # More conservative on exits
            final_position = 0.0
        elif state.position_size is not None and state.position_size > 0:
            # Managing existing long position
            if downtrend_strength >= 4:  # Still conservative on exits
                final_position = 0.0
            else:
                # Scale position based on trend strength
                current_position = state.position_size
                if uptrend_strength >= 4:  # Strong trend continuation
                    final_position = position_size  # Allow position to grow
                else:
                    final_position = current_position * 0.8  # Gradual reduction
        else:
            # No position and no new signal
            final_position = 0.0
            
        # Adjust for volatility conditions
        if squeeze:
            final_position *= 0.5  # Reduce position in low volatility
        elif expansion:
            final_position *= 1.2  # Increase position in high volatility
            final_position = min(final_position, 1.0)  # Still cap at 100%
        
        # Update state
        state.position_size = final_position
        state.entry_price = close
        
        return final_position, state
        
    except Exception as e:
        print(f"Error in strategy: {str(e)}")
        return 0.0, state  # Return to neutral position on error
