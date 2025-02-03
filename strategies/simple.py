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
    Enhanced Moving Average Crossover Strategy with Advanced Risk Management
    - Very aggressive entries with strong risk management
    - Dynamic position sizing based on market conditions
    - Trailing stops that tighten in profit
    - Multiple timeframe trend confirmation
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
        
        # More aggressive entry conditions
        if current_signal > prev_signal and uptrend_strength >= 2:  # Need only 2 conditions
            final_position = position_size
        elif current_signal == 1 and uptrend_strength >= 3:  # Or 3 conditions in existing trend
            final_position = position_size
        elif current_signal < prev_signal and downtrend_strength >= 2:  # Need only 2 conditions
            final_position = 0.0
        elif current_signal == 0 and downtrend_strength >= 4:  # More conservative on exits
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
        close = df['Close'].iloc[current_step]
        bb_upper = df['BB_Upper'].iloc[current_step]
        bb_lower = df['BB_Lower'].iloc[current_step]
        bb_ma = df['BB_MA20'].iloc[current_step]
        bb_width = df['BB_Width'].iloc[current_step]
        bb_pct = df['BB_PCT'].iloc[current_step]
        rsi = df['RSI'].iloc[current_step]
        volatility = df['Volatility'].iloc[current_step]
        macd = df['MACD'].iloc[current_step]
        signal = df['Signal'].iloc[current_step]
        
        # Get moving averages for multiple timeframe trend analysis
        ma10 = df['MA10'].iloc[current_step]
        ma30 = df['MA30'].iloc[current_step]
        
        # Previous values for momentum
        prev_close = df['Close'].iloc[current_step - 1]
        prev_bb_pct = df['BB_PCT'].iloc[current_step - 1]
        prev_macd = df['MACD'].iloc[current_step - 1]
        prev_ma10 = df['MA10'].iloc[current_step - 1]
        prev_ma30 = df['MA30'].iloc[current_step - 1]
        
        # Calculate returns for momentum
        returns_20d = (close / df['Close'].iloc[max(0, current_step-20)]) - 1
        
        # Update risk metrics
        portfolio_value = close * state.position_size if state.position_size else close
        state.update_risk_metrics(close, portfolio_value, current_step)
        
        # Check if we should exit based on risk management
        if state.should_exit_trade(close, volatility):
            return 0.0, state
            
        # Calculate base position size - start aggressive
        base_position = 1.0
        
        # Enhanced trend identification
        strong_uptrend = (
            ma10 > ma30 and  # Short-term trend up
            close > ma10 and  # Price above short MA
            (ma10 > prev_ma10 or df['Strong_Trend'].iloc[current_step]) and  # Rising short MA or strong trend
            df['ROC_MA'].iloc[current_step] > 0  # Positive momentum
        )
        
        moderate_uptrend = (
            ma10 > ma30 and  # Short-term trend up
            close > ma10  # Price above short MA
        )
        
        strong_downtrend = (
            ma10 < ma30 and  # Short-term trend down
            close < ma10 and  # Price below short MA
            (ma10 < prev_ma10 or df['Strong_Trend'].iloc[current_step]) and  # Falling short MA or strong trend
            df['ROC_MA'].iloc[current_step] < 0  # Negative momentum
        )
        
        # Calculate trend strength
        trend_strength = df['Trend_Strength'].iloc[current_step]
        momentum_strength = abs(df['ROC_MA'].iloc[current_step])
        
        # Adjust position size based on market conditions
        bb_width_percentile = df['BB_Width'].iloc[max(0, current_step-100):current_step+1].rank(pct=True).iloc[-1]
        
        if strong_uptrend:
            if bb_width_percentile < 0.3:  # Tight bands in strong uptrend
                base_position *= 3.0  # Very aggressive
            else:
                base_position *= 2.5  # Aggressive
                
            # Additional boost based on momentum
            if momentum_strength > 0.5:  # Strong momentum
                base_position *= 1.5
            elif momentum_strength > 0.3:  # Moderate momentum
                base_position *= 1.3
                
        elif moderate_uptrend:
            if bb_width_percentile < 0.3:
                base_position *= 2.0  # Still aggressive
            else:
                base_position *= 1.7  # Moderately aggressive
                
        elif strong_downtrend:
            base_position *= 0.2  # Very defensive
        else:  # Sideways or unclear trend
            if bb_width_percentile < 0.3:
                base_position *= 1.2  # Slightly aggressive in tight ranges
            else:
                base_position *= 0.8  # Conservative in wide ranges
            
        # Get risk-adjusted position size
        position_size = state.get_position_size(base_position, volatility)
        
        # Enhanced entry signals with trend confirmation
        oversold_bounce = (
            (bb_pct < 0.3 or close < bb_lower) and  # Price near or below lower band
            (prev_bb_pct < bb_pct or close > prev_close) and  # Showing reversal signs
            (
                (strong_uptrend and rsi > 25) or  # Very lenient in strong uptrend
                (moderate_uptrend and rsi > 30) or  # Lenient in moderate uptrend
                (not moderate_uptrend and rsi > 35)  # Stricter in other conditions
            ) and
            (macd > prev_macd or macd > signal)  # MACD momentum
        )
        
        overbought_reversal = (
            (bb_pct > 0.7 or close > bb_upper) and  # Price near or above upper band
            (prev_bb_pct > bb_pct or close < prev_close) and  # Showing weakness
            (
                (strong_downtrend and rsi < 65) or  # More lenient in downtrend
                (not strong_uptrend and rsi < 75) or  # Moderate in normal conditions
                rsi < 80  # Upper limit for strong uptrends
            ) and
            (macd < prev_macd or macd < signal)  # MACD momentum
        )
        
        # Position Management
        if state.position_size is None or state.position_size == 0:
            if oversold_bounce:
                # Scale position size based on signal strength and trend
                signal_strength = sum([
                    bb_pct < 0.2,  # Strong oversold
                    rsi < 30,  # Strong oversold RSI
                    macd > signal,  # MACD bullish
                    close > prev_close,  # Price momentum
                    strong_uptrend,  # In strong uptrend
                    moderate_uptrend,  # In any uptrend
                    trend_strength > 0.02,  # Notable trend
                    momentum_strength > 0.03  # Notable momentum
                ])
                
                position_scale = 1.0 + (signal_strength * 0.3)  # Up to 3.4x size
                final_position = position_size * position_scale
            else:
                final_position = 0.0
        else:
            if overbought_reversal and not strong_uptrend:
                final_position = 0.0
            else:
                # Dynamic position management
                current_position = state.position_size
                
                # Scale up in strong uptrends
                if strong_uptrend:
                    if bb_pct < 0.6:  # More room to run
                        final_position = min(current_position * 1.5, position_size * 2.0)
                    else:
                        final_position = current_position
                # Scale up in moderate uptrends
                elif moderate_uptrend and trend_strength > 0.02:
                    if bb_pct < 0.5:  # Not too extended
                        final_position = min(current_position * 1.3, position_size * 1.5)
                    else:
                        final_position = current_position
                # Scale down in weakness or near resistance
                elif bb_pct > 0.8 or (macd < signal and rsi > 70) or strong_downtrend:
                    final_position = current_position * 0.5  # Faster reduction
                # Hold in normal conditions
                else:
                    final_position = current_position
        
        # Ensure position size doesn't exceed max
        final_position = min(final_position, 1.0)
        
        # Update state
        state.position_size = final_position
        state.entry_price = close
        
        return final_position, state
        
    except Exception as e:
        print(f"Error in strategy: {str(e)}")
        return 0.0, state  # Return to neutral position on error
