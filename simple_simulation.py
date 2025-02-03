import pandas as pd
import numpy as np
from trading_env import TradingEnv
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

def calculate_rsi(prices, n=14):
    """Calculate Relative Strength Index (RSI)"""
    gains = (prices - prices.shift(1)).clip(lower=0)
    losses = (prices.shift(1) - prices).clip(lower=0)
    avg_gain = gains.ewm(com=n-1, adjust=False).mean()
    avg_loss = losses.ewm(com=n-1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_adx(df, period=14):
    """Calculate Average Directional Index (ADX)"""
    df = df.copy()
    
    # Calculate +DM and -DM
    high_diff = df['High'].diff()
    low_diff = df['Low'].diff()
    
    pos_dm = high_diff.where((high_diff > 0) & (high_diff > -low_diff), 0.0)
    neg_dm = (-low_diff).where((low_diff < 0) & (-low_diff > high_diff), 0.0)
    
    # Calculate True Range
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': abs(df['High'] - df['Close'].shift(1)),
        'lc': abs(df['Low'] - df['Close'].shift(1))
    }).max(axis=1)
    
    # Smooth with Wilder's smoothing
    tr_smooth = tr.rolling(period).mean()
    pos_dm_smooth = pos_dm.rolling(period).mean()
    neg_dm_smooth = neg_dm.rolling(period).mean()
    
    # Calculate +DI and -DI
    pos_di = 100 * pos_dm_smooth / tr_smooth
    neg_di = 100 * neg_dm_smooth / tr_smooth
    
    # Calculate ADX
    dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
    adx = dx.rolling(period).mean()
    
    return adx, pos_di, neg_di

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift())
    tr3 = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def get_market_data(symbol='NVDA', period='1y'):
    """
    Fetch market data and calculate technical indicators
    """
    print(f"\nFetched market data for {symbol}:")
    print(f"Period: {period}")
    
    # Fetch data
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    
    # Also fetch SPY data for market benchmark
    spy_ticker = yf.Ticker('SPY')
    spy_data = spy_ticker.history(period=period)
    
    # Print data info
    print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Number of trading days: {len(df)}")
    print(f"Current price: ${df['Close'].iloc[-1]:.2f}")
    print(f"Period return: {((df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100):.2f}%\n")
    
    # Calculate returns
    df['Returns'] = df['Close'].pct_change()
    df['Market_Returns'] = spy_data['Close'].pct_change()  # Add market returns
    
    # Calculate volatility
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
    
    # Calculate momentum indicators
    df['ROC_5'] = df['Close'].pct_change(periods=5)
    df['ROC_20'] = df['Close'].pct_change(periods=20)
    
    # Calculate Z-Score
    df['Z_Score'] = (df['Close'] - df['Close'].rolling(window=20).mean()) / df['Close'].rolling(window=20).std()
    
    # Market regime classification
    df['Trend'] = 0  # Initialize trend column
    df.loc[df['Close'] > df['SMA_50'], 'Trend'] = 1  # Uptrend
    df.loc[df['Close'] < df['SMA_50'], 'Trend'] = -1  # Downtrend
    
    # Volatility regime
    vol_percentile = df['Volatility'].rolling(window=60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    df['Volatility_Regime'] = 'Normal'
    df.loc[vol_percentile > 0.8, 'Volatility_Regime'] = 'High'
    df.loc[vol_percentile < 0.2, 'Volatility_Regime'] = 'Low'
    
    # Market regime
    df['Market_Regime'] = 'Sideways'
    df.loc[(df['Trend'] == 1) & (df['Volatility_Regime'] != 'High'), 'Market_Regime'] = 'Bull'
    df.loc[(df['Trend'] == -1) & (df['Volatility_Regime'] != 'Low'), 'Market_Regime'] = 'Bear'
    
    return df

def calculate_transition_metrics(df, current_step, lookback=20):
    """
    Calculate transition metrics for regime changes:
    - Trend momentum and acceleration
    - Volatility state changes
    - Regime persistence probability
    - Transition velocity
    """
    if current_step < lookback:
        return None
    
    # Get historical data
    hist_market_regime = df['Market_Regime'].iloc[current_step-lookback:current_step+1]
    hist_vol_regime = df['Volatility_Regime'].iloc[current_step-lookback:current_step+1]
    hist_trend = df['Trend'].iloc[current_step-lookback:current_step+1]
    hist_mom = df['ROC_20'].iloc[current_step-lookback:current_step+1]
    
    # Calculate momentum metrics
    trend_velocity = hist_trend.diff().mean()
    trend_acceleration = hist_trend.diff().diff().mean()
    mom_velocity = hist_mom.diff().mean()
    mom_acceleration = hist_mom.diff().diff().mean()
    
    # Volatility dynamics
    vol_velocity = hist_vol_regime.diff().mean()
    vol_acceleration = hist_vol_regime.diff().diff().mean()
    
    # Regime persistence
    regime_changes = (hist_market_regime.diff() != 0).sum()
    persistence_prob = 1 - (regime_changes / lookback)
    
    # Transition velocity (how fast regimes are changing)
    transition_velocity = (
        abs(trend_velocity) * 0.4 +
        abs(mom_velocity) * 0.3 +
        abs(vol_velocity) * 0.3
    )
    
    # Directional bias
    directional_bias = (
        np.sign(trend_velocity) * 0.4 +
        np.sign(mom_velocity) * 0.3 +
        np.sign(vol_velocity) * 0.3
    )
    
    return {
        'trend_velocity': trend_velocity,
        'trend_acceleration': trend_acceleration,
        'mom_velocity': mom_velocity,
        'mom_acceleration': mom_acceleration,
        'vol_velocity': vol_velocity,
        'vol_acceleration': vol_acceleration,
        'persistence_prob': persistence_prob,
        'transition_velocity': transition_velocity,
        'directional_bias': directional_bias
    }

def calculate_drawdown_metrics(df, current_step, lookback=60):
    """
    Calculate enhanced drawdown metrics with early warning signals
    """
    if current_step < 20:
        return {
            'current_drawdown': 0,
            'drawdown_warning': 0,
            'recovery_strength': 0,
            'vol_adjusted_drawdown': 0
        }
    
    try:
        # Get recent price and volume data
        prices = df['Close'].iloc[max(0, current_step-lookback):current_step+1]
        returns = df['Returns'].iloc[max(0, current_step-lookback):current_step+1]
        volume = df['Volume'].iloc[max(0, current_step-lookback):current_step+1]
        
        # Calculate rolling peaks and current drawdown
        rolling_peak = prices.expanding().max()
        drawdown = (prices - rolling_peak) / rolling_peak * 100
        current_drawdown = drawdown.iloc[-1]
        
        # Enhanced early warning system
        warning_signals = []
        
        # 1. Volume-weighted price momentum
        vol_weighted_returns = returns * (volume / volume.mean())
        recent_momentum = vol_weighted_returns.tail(5).mean() * 100
        warning_signals.append(-1 if recent_momentum < -1 else 0)
        
        # 2. Volatility acceleration
        vol_window = min(20, len(returns))
        if vol_window > 5:
            recent_vol = returns.tail(5).std() * np.sqrt(252)
            historical_vol = returns.tail(vol_window).std() * np.sqrt(252)
            vol_ratio = recent_vol / historical_vol if historical_vol != 0 else 1
            warning_signals.append(-1 if vol_ratio > 1.5 else 0)
        
        # 3. Price structure breakdown
        if len(prices) >= 20:
            ma20 = prices.rolling(window=20).mean()
            ma_cross = (prices.iloc[-1] < ma20.iloc[-1]) and (prices.iloc[-2] > ma20.iloc[-2])
            warning_signals.append(-1 if ma_cross else 0)
        
        # 4. Volume spike detection
        if len(volume) >= 5:
            vol_ratio = volume.iloc[-1] / volume.tail(5).mean()
            price_down = returns.iloc[-1] < 0
            warning_signals.append(-1 if vol_ratio > 2 and price_down else 0)
        
        # Combine warning signals
        drawdown_warning = sum(warning_signals) / len(warning_signals) if warning_signals else 0
        
        # Recovery strength calculation
        recovery_signals = []
        
        # 1. Price momentum recovery
        if len(returns) >= 5:
            recovery_signals.append(1 if recent_momentum > 1 else 0)
        
        # 2. Volume-supported recovery
        if len(prices) >= 5:
            price_up = returns.iloc[-1] > 0
            vol_confirm = volume.iloc[-1] > volume.tail(5).mean()
            recovery_signals.append(1 if price_up and vol_confirm else 0)
        
        # 3. Moving average recovery
        if len(prices) >= 20:
            ma_recovery = (prices.iloc[-1] > ma20.iloc[-1]) and (prices.iloc[-2] < ma20.iloc[-2])
            recovery_signals.append(1 if ma_recovery else 0)
        
        # Combine recovery signals
        recovery_strength = sum(recovery_signals) / len(recovery_signals) if recovery_signals else 0
        
        # Volatility-adjusted drawdown
        vol_adjusted_drawdown = current_drawdown * (1 + max(0, vol_ratio - 1))
        
        return {
            'current_drawdown': current_drawdown,
            'drawdown_warning': drawdown_warning,
            'recovery_strength': recovery_strength,
            'vol_adjusted_drawdown': vol_adjusted_drawdown
        }
        
    except Exception as e:
        print(f"Error calculating drawdown metrics: {str(e)}")
        return {
            'current_drawdown': 0,
            'drawdown_warning': 0,
            'recovery_strength': 0,
            'vol_adjusted_drawdown': 0
        }

def calculate_recovery_metrics(df, current_step, lookback=60):
    """
    Calculate enhanced recovery metrics with institutional confirmation
    """
    if current_step < 20:
        return {
            'recovery_score': 0,
            'institutional_support': 0,
            'momentum_divergence': 0,
            'volume_profile': 0
        }
    
    try:
        # Get recent price and volume data
        prices = df['Close'].iloc[max(0, current_step-lookback):current_step+1]
        returns = df['Returns'].iloc[max(0, current_step-lookback):current_step+1]
        volume = df['Volume'].iloc[max(0, current_step-lookback):current_step+1]
        
        # Calculate recovery score
        recovery_score = 0
        
        # 1. Price momentum recovery
        if len(returns) >= 5:
            recent_momentum = returns.tail(5).mean() * 100
            recovery_score += 0.4 if recent_momentum > 1 else 0
        
        # 2. Volume-supported recovery
        if len(prices) >= 5:
            price_up = returns.iloc[-1] > 0
            vol_confirm = volume.iloc[-1] > volume.tail(5).mean()
            recovery_score += 0.3 if price_up and vol_confirm else 0
        
        # 3. Moving average recovery
        if len(prices) >= 20:
            ma20 = prices.rolling(window=20).mean()
            ma_recovery = (prices.iloc[-1] > ma20.iloc[-1]) and (prices.iloc[-2] < ma20.iloc[-2])
            recovery_score += 0.3 if ma_recovery else 0
        
        # Institutional support calculation
        institutional_support = 0
        
        # 1. Volume profile
        if len(volume) >= 5:
            vol_profile = volume.iloc[-1] / volume.tail(5).mean()
            institutional_support += 0.4 if vol_profile > 1.5 else 0
        
        # 2. Price action confirmation
        if len(prices) >= 5:
            price_up = returns.iloc[-1] > 0
            institutional_support += 0.3 if price_up else 0
        
        # 3. Momentum divergence
        if len(returns) >= 10:
            short_mom = returns.tail(5).mean() * 100
            long_mom = returns.tail(10).mean() * 100
            momentum_divergence = short_mom - long_mom
            institutional_support += 0.3 if momentum_divergence > 1 else 0
        
        # Volume profile calculation
        volume_profile = 0
        
        # 1. Volume spike detection
        if len(volume) >= 5:
            vol_ratio = volume.iloc[-1] / volume.tail(5).mean()
            volume_profile += 0.5 if vol_ratio > 2 else 0
        
        # 2. Volume trend confirmation
        if len(volume) >= 10:
            vol_trend = volume.rolling(window=10).mean()
            volume_profile += 0.5 if volume.iloc[-1] > vol_trend.iloc[-1] else 0
        
        return {
            'recovery_score': recovery_score,
            'institutional_support': institutional_support,
            'momentum_divergence': momentum_divergence,
            'volume_profile': volume_profile
        }
        
    except Exception as e:
        print(f"Error calculating recovery metrics: {str(e)}")
        return {
            'recovery_score': 0,
            'institutional_support': 0,
            'momentum_divergence': 0,
            'volume_profile': 0
        }

def classify_market_regime(df, current_step):
    """
    Enhanced market regime classification with sophisticated transitions
    """
    if current_step < 50:
        return "Mixed", 0, {}
    
    try:
        # Get recent data
        returns = df['Returns'].iloc[max(0, current_step-60):current_step+1]
        prices = df['Close'].iloc[max(0, current_step-60):current_step+1]
        volume = df['Volume'].iloc[max(0, current_step-60):current_step+1]
        
        # Enhanced trend analysis
        ma_20 = prices.rolling(window=20).mean()
        ma_50 = prices.rolling(window=50).mean()
        
        # Volume-weighted trend
        volume_ratio = volume / volume.rolling(window=20).mean()
        weighted_returns = returns * volume_ratio
        
        # Trend metrics
        trend_score = 0
        
        # Price trend (40% weight)
        if len(ma_20) > 1 and len(ma_50) > 1:
            short_trend = (ma_20.iloc[-1] / ma_20.iloc[-2] - 1) * 100
            long_trend = (ma_50.iloc[-1] / ma_50.iloc[-2] - 1) * 100
            trend_score += 0.4 * (short_trend * 0.7 + long_trend * 0.3)
        
        # Momentum (30% weight)
        if len(returns) >= 20:
            momentum = weighted_returns.tail(20).mean() * 100
            trend_score += 0.3 * momentum
        
        # Volatility regime (30% weight)
        if len(returns) >= 50:
            current_vol = returns.tail(20).std() * np.sqrt(252)
            historical_vol = returns.std() * np.sqrt(252)
            vol_ratio = current_vol / historical_vol if historical_vol != 0 else 1
            vol_score = -1 if vol_ratio > 1.5 else 1 if vol_ratio < 0.75 else 0
            trend_score += 0.3 * vol_score
        
        # Enhanced regime classification
        regime_type = []
        regime_strength = abs(trend_score)
        
        # Trend direction
        if trend_score > 1.5:
            regime_type.append("Strong_Bull")
        elif trend_score > 0.5:
            regime_type.append("Bull")
        elif trend_score < -1.5:
            regime_type.append("Strong_Bear")
        elif trend_score < -0.5:
            regime_type.append("Bear")
        else:
            regime_type.append("Mixed")
        
        # Volatility state
        if vol_ratio > 1.5:
            regime_type.append("High_Vol")
        elif vol_ratio < 0.75:
            regime_type.append("Low_Vol")
        
        # Momentum state
        if len(returns) >= 20:
            mom_score = weighted_returns.tail(20).mean() * 100
            if abs(mom_score) > 1:
                regime_type.append("Strong_Momentum" if mom_score > 0 else "Strong_Reversal")
        
        # Transition detection
        if len(ma_20) > 20 and len(ma_50) > 50:
            ma_cross = (ma_20.iloc[-1] > ma_50.iloc[-1]) != (ma_20.iloc[-2] > ma_50.iloc[-2])
            if ma_cross:
                regime_type.append("Transitional")
        
        # Market breadth
        if len(returns) >= 20:
            up_days = (returns.tail(20) > 0).sum()
            if up_days >= 15:
                regime_type.append("High_Breadth")
            elif up_days <= 5:
                regime_type.append("Low_Breadth")
        
        # Combine regime types
        regime = "_".join(regime_type)
        
        # Calculate additional metrics
        metrics = {
            'trend_score': trend_score,
            'volatility_ratio': vol_ratio,
            'momentum_score': momentum if 'momentum' in locals() else 0,
            'regime_strength': regime_strength
        }
        
        return regime, regime_strength, metrics
        
    except Exception as e:
        print(f"Error in regime classification: {str(e)}")
        return "Mixed", 0, {}

def calculate_position_size(regime_type, regime_metrics, regime_strength, current_price):
    """
    Calculate position size with enhanced recovery detection
    """
    try:
        # Base position size
        base_size = 0.6
        
        # Get regime components
        regime_components = regime_type.split('_')
        
        # Calculate trend multiplier
        trend_multiplier = 1.0
        if 'Strong_Bull' in regime_type:
            trend_multiplier = 1.3
        elif 'Bull' in regime_type:
            trend_multiplier = 1.1
        elif 'Strong_Bear' in regime_type:
            trend_multiplier = 0.5
        elif 'Bear' in regime_type:
            trend_multiplier = 0.7
        
        # Volatility adjustment
        vol_multiplier = 1.0
        if 'High_Vol' in regime_type:
            vol_multiplier = 0.7
        elif 'Low_Vol' in regime_type:
            vol_multiplier = 1.2
        
        # Enhanced recovery detection
        recovery_metrics = regime_metrics.get('recovery_metrics', {})
        recovery_score = recovery_metrics.get('recovery_score', 0)
        momentum_divergence = recovery_metrics.get('momentum_divergence', 0)
        volume_profile = recovery_metrics.get('volume_profile', 0)
        institutional_support = recovery_metrics.get('institutional_support', 0)
        
        # Recovery multiplier with institutional confirmation
        recovery_multiplier = 1.0
        if recovery_score > 0.8 and institutional_support > 0.7:
            recovery_multiplier = 1.4  # Strong recovery with institutional support
        elif recovery_score > 0.6:
            recovery_multiplier = 1.2  # Moderate recovery
        elif recovery_score > 0.4:
            recovery_multiplier = 1.1  # Early recovery signs
        
        # Momentum divergence boost
        if momentum_divergence > 0.7 and volume_profile > 0.6:
            recovery_multiplier *= 1.2  # Additional boost for strong divergence with volume confirmation
        
        # Progressive position sizing based on recovery strength
        position_size = (
            base_size *
            trend_multiplier *
            vol_multiplier *
            recovery_multiplier *
            regime_strength
        )
        
        # Dynamic position bounds based on conviction
        min_size = 0.1
        max_size = 1.0
        if institutional_support > 0.8 and recovery_score > 0.8:
            max_size = 1.2  # Allow slightly higher allocation for high-conviction setups
        
        position_size = max(min_size, min(max_size, position_size))
        
        return position_size
        
    except Exception as e:
        print(f"Error calculating position size: {str(e)}")
        return 0.5  # Default to moderate position size on error

def calculate_trend_metrics(df, current_step):
    """
    Calculate comprehensive trend metrics with enhanced alpha generation
    """
    metrics = {}
    
    try:
        # Price data for the last 60 days
        prices = df['Close'].iloc[max(0, current_step-60):current_step+1]
        returns = df['Returns'].iloc[max(0, current_step-60):current_step+1]
        volume = df['Volume'].iloc[max(0, current_step-60):current_step+1]
        
        # Enhanced momentum calculation with volume weighting
        volume_weighted_returns = returns * (volume / volume.mean())
        metrics['momentum_score'] = volume_weighted_returns.mean() * 100
        
        # Multi-timeframe trend strength
        trend_strengths = []
        for period in [5, 10, 20, 50]:
            if len(prices) >= period:
                ma = prices.rolling(window=period).mean()
                ma_slope = (ma.iloc[-1] - ma.iloc[-2]) / ma.iloc[-2] * 100
                trend_strengths.append(ma_slope)
        
        # Weight shorter timeframes more heavily for faster response
        weights = [0.4, 0.3, 0.2, 0.1]  # Increased weight on shorter timeframes
        metrics['trend_strength'] = sum(s * w for s, w in zip(trend_strengths, weights[:len(trend_strengths)]))
        
        # Enhanced short-term momentum with acceleration
        if len(returns) >= 5:
            recent_returns = returns[-5:]
            metrics['short_momentum'] = (recent_returns.mean() * 100 * 
                                       (1 + abs(recent_returns.mean() / recent_returns.std()) if recent_returns.std() != 0 else 1))
        
        # Volatility regime calculation with adaptive thresholds
        vol_window = min(50, len(returns))
        if vol_window > 0:
            current_vol = returns.iloc[-vol_window:].std() * np.sqrt(252)
            historical_vol = returns.std() * np.sqrt(252)
            metrics['volatility_regime'] = current_vol / historical_vol if historical_vol != 0 else 1
            
            # Volatility trend for regime shifts
            vol_ma = pd.Series(returns).rolling(window=20).std() * np.sqrt(252)
            metrics['vol_trend'] = (vol_ma.iloc[-1] / vol_ma.iloc[-20] if len(vol_ma) >= 20 else 1)
        
        # Enhanced mean reversion signals
        if len(prices) >= 20:
            ma20 = prices.rolling(window=20).mean()
            std20 = prices.rolling(window=20).std()
            zscore = (prices - ma20) / std20
            metrics['mean_reversion_signal'] = -zscore.iloc[-1]  # Negative zscore for mean reversion
            
            # Oversold/Overbought with volume confirmation
            vol_ratio = volume.iloc[-1] / volume.rolling(window=20).mean().iloc[-1]
            if zscore.iloc[-1] < -2 and vol_ratio > 1.5:
                metrics['oversold_signal'] = True
            elif zscore.iloc[-1] > 2 and vol_ratio > 1.5:
                metrics['overbought_signal'] = True
        
        # Enhanced trend quality metrics
        if len(prices) >= 50:
            # Trend consistency
            ma50 = prices.rolling(window=50).mean()
            above_ma = (prices > ma50).astype(int)
            metrics['trend_consistency'] = above_ma.rolling(window=20).mean().iloc[-1]
            
            # Trend momentum
            mom20 = prices.pct_change(periods=20)
            mom50 = prices.pct_change(periods=50)
            metrics['trend_momentum'] = (mom20.iloc[-1] * 0.7 + mom50.iloc[-1] * 0.3) * 100
            
            # Volume trend confirmation
            vol_ma = volume.rolling(window=20).mean()
            metrics['volume_trend_confirm'] = (volume.iloc[-1] / vol_ma.iloc[-1]) * (1 if mom20.iloc[-1] > 0 else -1)
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating trend metrics: {str(e)}")
        return {
            'trend_strength': 0,
            'momentum_score': 0,
            'short_momentum': 0,
            'volatility_regime': 1,
            'mean_reversion_signal': 0
        }

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

def mean_reversion_strategy(state, df, current_step):
    """
    Aggressive Mean Reversion Strategy:
    - Uses tighter Z-score thresholds for more frequent trading
    - Increases position sizes in favorable conditions
    - More aggressive momentum filters
    - Maintains risk management with volatility scaling
    """
    if current_step < 20:  # Need enough history for indicators
        return 2  # Hold
    
    # State components
    current_price = state.price
    current_returns = state.returns
    current_volatility = state.volatility
    
    try:
        # Get current indicators
        z_score = df['Z_Score'].iloc[current_step]
        rsi = df['RSI'].iloc[current_step]
        roc_5 = df['ROC_5'].iloc[current_step]
        roc_20 = df['ROC_20'].iloc[current_step]
        
        # Initialize score (0 = strong sell, 4 = strong buy)
        score = 2.0  # Start neutral
        
        # 1. Mean Reversion Component (50% weight)
        if z_score > 1.5:
            score -= 1.5
        elif z_score > 1.0:
            score -= 0.8
        elif z_score < -1.5:
            score += 1.5
        elif z_score < -1.0:
            score += 0.8
            
        # 2. Momentum Filter (30% weight)
        momentum_score = 0
        
        # RSI signals
        if rsi > 65:
            momentum_score -= 0.4
        elif rsi < 35:
            momentum_score += 0.4
            
        # Rate of Change signals
        if roc_5 > 0:
            if roc_20 > 0:
                momentum_score += 0.4
            else:
                momentum_score += 0.2
        elif roc_5 < 0:
            if roc_20 < 0:
                momentum_score -= 0.4
            else:
                momentum_score -= 0.2
            
        score += momentum_score
        
        # 3. Volatility Adjustment (20% weight)
        vol_percentile = pd.Series(current_volatility).rank(pct=True)[0]
        if vol_percentile > 0.8:
            score = 2 + (score - 2) * 0.7
        
        # Convert score to action
        if score >= 2.8:
            return 4  # Strong Buy
        elif score >= 2.3:
            return 3  # Buy
        elif score <= 1.2:
            return 0  # Strong Sell
        elif score <= 1.7:
            return 1  # Sell
        else:
            return 2  # Hold
            
    except Exception as e:
        print(f"Warning: Error in strategy calculation: {e}")
        return 2  # Hold as fallback

def adaptive_mean_reversion_strategy(state, df, current_step):
    """
    Adaptive mean reversion strategy with:
    1. Multiple timeframe mean reversion signals
    2. MACD and DMI trend filters
    3. Volatility-based position sizing
    4. Dynamic stop losses
    """
    try:
        # Default position size
        position_size = 1.0
        
        # Skip if not enough data
        if current_step < 50:
            return position_size, state
            
        # Get current price and calculate basic metrics
        current_price = df['Close'].iloc[current_step]
        
        # Calculate multiple timeframe moving averages
        sma_10 = df['Close'].iloc[current_step-10:current_step+1].mean()
        sma_20 = df['Close'].iloc[current_step-20:current_step+1].mean()
        sma_50 = df['Close'].iloc[current_step-50:current_step+1].mean()
        
        # Calculate MACD
        macd, signal = calculate_macd(df['Close'].iloc[:current_step+1])
        macd_current = macd.iloc[-1]
        signal_current = signal.iloc[-1]
        macd_hist = macd_current - signal_current
        
        # Calculate DMI
        pdi, ndi = calculate_dmi(df['High'].iloc[:current_step+1], 
                               df['Low'].iloc[:current_step+1],
                               df['Close'].iloc[:current_step+1])
        pdi_current = pdi.iloc[-1]
        ndi_current = ndi.iloc[-1]
        adx = abs(pdi_current - ndi_current)
        
        # Calculate Bollinger Bands (20-day)
        bb_std = df['Close'].iloc[current_step-20:current_step+1].std()
        bb_upper = sma_20 + 2 * bb_std
        bb_lower = sma_20 - 2 * bb_std
        
        # Calculate RSI
        delta = df['Close'].diff()
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        avg_gain = gains.iloc[current_step-14:current_step].mean()
        avg_loss = -losses.iloc[current_step-14:current_step].mean()
        rs = avg_gain / avg_loss if avg_loss != 0 else 1
        rsi = min(100, max(0, 100 - (100 / (1 + rs))))
        
        # Calculate momentum
        returns_5d = df['Returns'].iloc[current_step-5:current_step].mean()
        returns_20d = df['Returns'].iloc[current_step-20:current_step].mean()
        volatility_20d = df['Returns'].iloc[current_step-20:current_step].std()
        
        # Multi-timeframe trend analysis
        trend_short = current_price > sma_10
        trend_medium = current_price > sma_20
        trend_long = current_price > sma_50
        
        # Trend strength indicators
        trend_alignment = sum([trend_short, trend_medium, trend_long])  # 0 to 3
        price_to_sma50 = current_price / sma_50 - 1 if sma_50 != 0 else 0
        trend_strength = abs(price_to_sma50)
        
        # MACD trend signals
        macd_trend = macd_current > signal_current
        macd_strength = abs(macd_hist) / current_price  # Normalized MACD histogram
        
        # DMI trend signals
        dmi_trend = pdi_current > ndi_current
        trend_intensity = min(adx / 30, 1.0)  # Normalized ADX, capped at 1.0
        
        # Composite trend score (-1 to 1)
        trend_score = (
            0.3 * (trend_alignment / 3 - 0.5) +  # Multi-timeframe alignment
            0.3 * (1 if macd_trend else -1) * macd_strength +  # MACD
            0.4 * (1 if dmi_trend else -1) * trend_intensity  # DMI
        )
        
        # Volatility regime
        vol_lookback = df['Returns'].iloc[current_step-100:current_step].std()
        is_high_vol = volatility_20d > vol_lookback if vol_lookback != 0 else False
        
        # Base position size
        position_size = 1.0
        
        # Position sizing based on mean reversion and trend signals
        if current_price > bb_upper and rsi > 70:  # Overbought
            position_size = 0.5  # Reduce position
            
            # Adjust based on trend score
            if trend_score > 0.3:  # Strong uptrend
                position_size = 0.8  # Less reduction in strong uptrend
            elif trend_score < -0.3:  # Strong downtrend
                position_size = 0.3  # More reduction in strong downtrend
                
        elif current_price < bb_lower and rsi < 30:  # Oversold
            position_size = 1.5  # Increase position
            
            # Adjust based on trend score
            if trend_score < -0.3:  # Strong downtrend
                position_size = 1.2  # Less increase in strong downtrend
            elif trend_score > 0.3:  # Strong uptrend
                position_size = 1.8  # More increase in strong uptrend
        
        # Additional trend-based adjustments
        if abs(position_size - 1.0) > 0.1:  # If we have a significant position
            # Trend confirmation adjustment
            trend_adjustment = 0.2 * trend_score  # -0.2 to +0.2
            position_size = 1.0 + (position_size - 1.0) * (1.0 + trend_adjustment)
            
            # Volatility adjustment
            if is_high_vol:
                vol_dampener = 0.6
                position_size = 1.0 + (position_size - 1.0) * vol_dampener
        
        # Dynamic stop loss based on trend and volatility
        if state.position_size is not None and state.entry_price is not None:
            returns_since_entry = (current_price / state.entry_price - 1) if state.entry_price != 0 else 0
            
            # Base stop loss on volatility and trend intensity
            base_stop = -2 * volatility_20d if volatility_20d is not None else -0.1
            trend_adjusted_stop = base_stop * (1 + trend_intensity)  # Wider stops in strong trends
            
            if returns_since_entry < trend_adjusted_stop:
                position_size = 1.0  # Return to neutral
        
        # Position change smoothing
        if state.position_size is not None:
            max_position_change = 0.2 * (1 + trend_intensity)  # More aggressive in strong trends
            position_delta = position_size - state.position_size
            if abs(position_delta) > max_position_change:
                position_size = state.position_size + (max_position_change if position_delta > 0 else -max_position_change)
        
        # Ensure position size is within bounds
        position_size = max(0.0, min(2.0, position_size))
        
        # Update state
        state.position_size = position_size
        state.entry_price = current_price
        
        return position_size, state
        
    except Exception as e:
        print(f"Error in strategy: {str(e)}")
        return 1.0, state  # Return to neutral position on error

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD and signal line"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_dmi(high, low, close, period=14):
    """Calculate Directional Movement Index"""
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    pdi = 100 * pd.Series(pos_dm).rolling(window=period).mean() / atr
    ndi = 100 * pd.Series(neg_dm).rolling(window=period).mean() / atr
    
    return pdi, ndi

class State:
    """
    Class to hold strategy state
    """
    def __init__(self):
        self.position_size = 0.0
        self.entry_price = None
        self.metrics = {}

def calculate_portfolio_metrics(portfolio_values, benchmark_values, risk_free_rate=0.04):
    """
    Calculate comprehensive portfolio analytics
    """
    # Convert to numpy arrays for calculations
    portfolio_values = np.array(portfolio_values)
    benchmark_values = np.array(benchmark_values)
    
    # Calculate returns
    portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]
    
    # Daily metrics
    daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
    excess_returns = portfolio_returns - daily_rf_rate
    
    # Basic performance metrics
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    benchmark_return = (benchmark_values[-1] / benchmark_values[0] - 1) * 100
    
    # Risk metrics
    volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
    benchmark_vol = np.std(benchmark_returns) * np.sqrt(252)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
    
    # Drawdown analysis
    rolling_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - rolling_max) / rolling_max
    max_drawdown = np.min(drawdowns) * 100
    
    # Risk-adjusted returns
    sharpe_ratio = (np.mean(excess_returns) * 252) / (np.std(portfolio_returns) * np.sqrt(252))
    sortino_ratio = (np.mean(excess_returns) * 252) / (downside_vol if downside_vol > 0 else np.inf)
    
    # Beta and Alpha
    covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
    variance = np.var(benchmark_returns)
    beta = covariance / variance if variance != 0 else 1
    
    # Alpha (annualized)
    alpha = (np.mean(portfolio_returns) * 252) - (daily_rf_rate * 252) - (beta * (np.mean(benchmark_returns) * 252 - daily_rf_rate * 252))
    
    # Information ratio
    tracking_error = np.std(portfolio_returns - benchmark_returns) * np.sqrt(252)
    information_ratio = np.mean(portfolio_returns - benchmark_returns) * 252 / tracking_error if tracking_error != 0 else 0
    
    # Win rate and average trade metrics
    positive_days = np.sum(portfolio_returns > 0)
    total_days = len(portfolio_returns)
    win_rate = (positive_days / total_days) * 100
    
    avg_daily_return = np.mean(portfolio_returns) * 100
    avg_up_day = np.mean(portfolio_returns[portfolio_returns > 0]) * 100 if len(portfolio_returns[portfolio_returns > 0]) > 0 else 0
    avg_down_day = np.mean(portfolio_returns[portfolio_returns < 0]) * 100 if len(portfolio_returns[portfolio_returns < 0]) > 0 else 0
    
    # Regime analysis
    bull_returns = portfolio_returns[benchmark_returns > 0]
    bear_returns = portfolio_returns[benchmark_returns <= 0]
    bull_performance = np.mean(bull_returns) * 252 * 100 if len(bull_returns) > 0 else 0
    bear_performance = np.mean(bear_returns) * 252 * 100 if len(bear_returns) > 0 else 0
    
    # Calculate time-weighted returns
    portfolio_twrr = (1 + portfolio_returns).prod() ** (252/len(portfolio_returns)) - 1
    benchmark_twrr = (1 + benchmark_returns).prod() ** (252/len(benchmark_returns)) - 1
    
    return {
        'total_return': total_return,
        'benchmark_return': benchmark_return,
        'excess_return': total_return - benchmark_return,
        'volatility': volatility * 100,
        'benchmark_volatility': benchmark_vol * 100,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'beta': beta,
        'alpha': alpha * 100,
        'information_ratio': information_ratio,
        'tracking_error': tracking_error * 100,
        'win_rate': win_rate,
        'avg_daily_return': avg_daily_return,
        'avg_up_day': avg_up_day,
        'avg_down_day': avg_down_day,
        'bull_performance': bull_performance,
        'bear_performance': bear_performance,
        'annualized_return': portfolio_twrr * 100,
        'benchmark_annualized': benchmark_twrr * 100,
        'downside_volatility': downside_vol * 100
    }

def run_simulation(strategy_func, df, initial_capital=10000.0):
    """
    Run trading simulation with enhanced metrics and risk management
    """
    print(f"\nStarting simulation with {strategy_func.__name__} Strategy on SPY...")
    print(f"Initial portfolio value: ${initial_capital:.2f}")
    
    # Initialize portfolio metrics with NaN arrays
    portfolio_values = np.full(len(df), np.nan)
    returns = np.full(len(df), np.nan)
    holdings = np.full(len(df), np.nan)
    position_sizes = np.full(len(df), np.nan)
    
    # Set initial portfolio value
    portfolio_values[0] = initial_capital
    returns[0] = 0.0  # First day return is 0
    
    # Initialize state
    state = State()
    
    # Run simulation
    for step in range(len(df)):
        try:
            # Get current market data
            current_price = df['Close'].iloc[step]
            
            # Get position size from strategy
            position_size, state = strategy_func(state, df, step)
            position_sizes[step] = position_size
            
            # Calculate holdings and portfolio value
            current_holdings = position_size * initial_capital
            if step > 0:
                # Update holdings based on price change
                price_change = current_price / df['Close'].iloc[step-1]
                current_holdings = holdings[step-1] * price_change
            
            holdings[step] = current_holdings
            portfolio_values[step] = current_holdings
            
            # Calculate returns (skip first day)
            if step > 0:
                returns[step] = (portfolio_values[step] / portfolio_values[step-1] - 1)
            
            # Print progress
            if step % 20 == 0:
                print(f"Step {step}: Portfolio value: ${portfolio_values[step]:.2f}")
            
        except Exception as e:
            print(f"Error in simulation step {step}: {str(e)}")
            # On error, use previous values
            if step > 0:
                portfolio_values[step] = portfolio_values[step-1]
                holdings[step] = holdings[step-1]
                position_sizes[step] = position_sizes[step-1]
                returns[step] = 0.0  # No return on error days
    
    # Convert arrays to pandas Series
    portfolio_values = pd.Series(portfolio_values, index=df.index)
    returns_series = pd.Series(returns, index=df.index)
    
    # Calculate performance metrics
    final_portfolio_value = portfolio_values.iloc[-1]
    strategy_return = (final_portfolio_value / initial_capital - 1) * 100
    
    benchmark_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    excess_return = strategy_return - benchmark_return
    
    # Risk metrics
    volatility = returns_series.std() * np.sqrt(252) * 100
    
    # Calculate drawdown
    rolling_max = portfolio_values.expanding().max()
    drawdowns = (portfolio_values - rolling_max) / rolling_max * 100
    max_drawdown = drawdowns.min()
    
    # Risk-adjusted metrics
    risk_free_rate = 0.02  # Assumed 2% risk-free rate
    excess_returns = returns_series - risk_free_rate/252
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns_series.std() if returns_series.std() != 0 else 0
    
    # Sortino ratio (downside deviation)
    downside_returns = returns_series[returns_series < 0]
    sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() != 0 else 0
    
    # Calculate alpha and beta using market returns
    market_returns = df['Market_Returns'].fillna(0)  # Fill NaN values with 0
    strategy_returns = returns_series.fillna(0)  # Fill NaN values with 0
    
    # Beta calculation
    if len(market_returns) > 1 and len(strategy_returns) > 1:
        try:
            # Calculate covariance matrix
            cov_matrix = np.cov(strategy_returns.values, market_returns.values)
            if cov_matrix.shape == (2, 2):  # Ensure valid covariance matrix
                covariance = cov_matrix[0,1]
                market_variance = np.var(market_returns.values)
                beta = covariance / market_variance if market_variance != 0 else 1
                
                # Calculate annualized alpha
                market_return_annual = market_returns.mean() * 252 * 100
                strategy_return_annual = strategy_returns.mean() * 252 * 100
                alpha = strategy_return_annual - (risk_free_rate + beta * (market_return_annual - risk_free_rate))
            else:
                beta = 1
                alpha = 0
        except Exception as e:
            print(f"Error calculating alpha/beta: {str(e)}")
            beta = 1
            alpha = 0
    else:
        beta = 1
        alpha = 0
    
    # Information ratio
    tracking_error = (strategy_returns - market_returns).std() * np.sqrt(252)
    information_ratio = excess_return / (tracking_error * 100) if tracking_error != 0 else 0
    
    # Trading statistics
    position_changes = np.diff(position_sizes) != 0
    num_trades = np.sum(position_changes)
    winning_days = np.sum(returns_series > 0)
    total_days = len(returns_series)
    win_rate = (winning_days / total_days * 100) if total_days > 0 else 0
    
    # Average returns
    avg_daily_return = returns_series.mean() * 100
    avg_up_day = returns_series[returns_series > 0].mean() * 100 if len(returns_series[returns_series > 0]) > 0 else 0
    avg_down_day = returns_series[returns_series < 0].mean() * 100 if len(returns_series[returns_series < 0]) > 0 else 0
    
    # Market condition performance
    bull_returns = returns_series[market_returns > 0].sum() * 100
    bear_returns = returns_series[market_returns < 0].sum() * 100
    
    # Print performance report
    print("\n=== Performance Report ===")
    print(f"Strategy Return: {strategy_return:.2f}%")
    print(f"Benchmark Return: {benchmark_return:.2f}%")
    print(f"Excess Return: {excess_return:.2f}%\n")
    
    print("Risk Metrics:")
    print(f"Volatility: {volatility:.2f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {sortino_ratio:.2f}\n")
    
    print("Risk-Adjusted Metrics:")
    print(f"Alpha: {alpha:.2f}%")
    print(f"Beta: {beta:.2f}")
    print(f"Information Ratio: {information_ratio:.2f}\n")
    
    print("Trading Statistics:")
    print(f"Number of Trades: {num_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Avg Daily Return: {avg_daily_return:.2f}%")
    print(f"Avg Up Day: {avg_up_day:.2f}%")
    print(f"Avg Down Day: {avg_down_day:.2f}%\n")
    
    print("Market Condition Performance:")
    print(f"Bull Market Performance: {bull_returns:.2f}%")
    print(f"Bear Market Performance: {bear_returns:.2f}%\n")
    
    print("=========================")
    
    return portfolio_values, returns_series

if __name__ == "__main__":
    # Run simulation with buy-and-hold strategy
    df = get_market_data(symbol='NVDA', period='1y')
    env = run_simulation(buy_and_hold_strategy, df)
