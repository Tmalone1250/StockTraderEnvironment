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

def get_market_data(symbol='SPY', period='1y'):
    """Fetch and prepare market data with additional indicators"""
    try:
        # Fetch data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            print("No data received from Yahoo Finance")
            return None
            
        # Calculate returns and volatility
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['ATR'] = calculate_atr(df)
        
        # Enhanced volatility analysis
        df['Vol_MA'] = df['Volatility'].rolling(window=20).mean()
        df['Vol_Regime'] = df['Volatility'] / df['Vol_MA']
        df['Vol_3D'] = df['Returns'].rolling(window=3).std()  # Short-term vol
        df['Vol_10D'] = df['Returns'].rolling(window=10).std()  # Medium-term vol
        df['Vol_Trend'] = df['Vol_3D'] / df['Vol_10D']  # Volatility trend
        
        # Volatility percentile rank (0-1)
        window = 60
        df['Vol_Rank'] = df['Volatility'].rolling(window=window).apply(
            lambda x: pd.Series(x).rank().iloc[-1] / window
        )
        
        # Basic indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['STD_20'] = df['Close'].rolling(window=20).std()
        df['RSI'] = calculate_rsi(df['Close'])
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_Upper'] = df['SMA_20'] + (df['STD_20'] * 2)
        df['BB_Lower'] = df['SMA_20'] - (df['STD_20'] * 2)
        
        # Rolling max for drawdown calculation
        df['Rolling_Max'] = df['Close'].rolling(window=20).max()
        df['Drawdown'] = (df['Close'] - df['Rolling_Max']) / df['Rolling_Max']
        
        # Volume analysis
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Forward fill any remaining NaN values
        df = df.ffill()
        
        # Print market summary
        current_price = df['Close'].iloc[-1]
        start_price = df['Close'].iloc[0]
        period_return = (current_price - start_price) / start_price * 100
        
        print(f"\nFetched market data for {symbol}:")
        print(f"Period: {period}")
        print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"Number of trading days: {len(df)}")
        print(f"Current price: ${current_price:.2f}")
        print(f"Period return: {period_return:.2f}%\n")
        
        return df
        
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return None

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
    current_price = state[0]
    current_returns = state[1]
    current_volatility = state[2]
    
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

def buy_and_hold_strategy(state, df, current_step):
    """
    Advanced Technical Strategy with Enhanced Risk Management:
    - Adaptive volatility-based position sizing
    - Multi-timeframe volatility analysis
    - Regime-based risk adjustment
    - Dynamic leverage scaling
    - Volatility trend following
    """
    if current_step < 26:  # Need enough history for indicators
        return 2  # Hold
    
    try:
        # State components
        current_price = state[0]
        current_returns = state[1]
        current_volatility = state[2]
        current_holdings = state[5]
        portfolio_value = state[6]
        
        # Get technical indicators
        rsi = df['RSI'].iloc[current_step]
        macd = df['MACD'].iloc[current_step]
        signal = df['Signal_Line'].iloc[current_step]
        bb_upper = df['BB_Upper'].iloc[current_step]
        bb_lower = df['BB_Lower'].iloc[current_step]
        sma_20 = df['SMA_20'].iloc[current_step]
        
        # Enhanced volatility metrics
        vol_regime = df['Vol_Regime'].iloc[current_step]
        vol_trend = df['Vol_Trend'].iloc[current_step]
        vol_rank = df['Vol_Rank'].iloc[current_step]
        vol_3d = df['Vol_3D'].iloc[current_step]
        vol_10d = df['Vol_10D'].iloc[current_step]
        
        # Other risk metrics
        atr = df['ATR'].iloc[current_step]
        drawdown = df['Drawdown'].iloc[current_step]
        volume_ratio = df['Volume_Ratio'].iloc[current_step]
        
        # Volatility regime classification
        is_low_vol = vol_rank < 0.3 and vol_regime < 0.9
        is_high_vol = vol_rank > 0.7 or vol_regime > 1.3
        is_vol_expanding = vol_trend > 1.1
        is_vol_contracting = vol_trend < 0.9
        
        # Calculate adjusted drawdown early
        base_drawdown = -0.08
        adjusted_drawdown = (
            base_drawdown * 0.8 if is_high_vol else
            base_drawdown * 1.2 if is_low_vol else
            base_drawdown
        )
        
        # Dynamic base position size based on volatility regime
        base_position_size = 2.0  # Maximum leverage
        if is_high_vol:
            base_position_size *= 0.7  # Reduce size in high vol
        elif is_low_vol:
            base_position_size *= 1.2  # Increase size in low vol
            
        # Volatility trend adjustment
        vol_trend_adj = 1.0
        if is_vol_expanding:
            vol_trend_adj *= 0.8  # Reduce exposure when vol is rising
        elif is_vol_contracting:
            vol_trend_adj *= 1.2  # Increase exposure when vol is falling
            
        # Final position size limits
        max_position_size = min(
            base_position_size * vol_trend_adj,
            2.5  # Hard cap on leverage
        )
        
        # Dynamic stop loss based on volatility regime
        stop_base = 2.5
        if is_high_vol:
            stop_base *= 0.8  # Tighter stops in high vol
        elif is_low_vol:
            stop_base *= 1.2  # Wider stops in low vol
            
        momentum_5d = current_price / df['Close'].iloc[current_step - 5] - 1
        momentum_scale = 1.0 + abs(momentum_5d)
        stop_loss_pct = (atr / current_price) * stop_base * momentum_scale
        
        # Calculate trend strength with volatility adjustment
        trend_strength = (
            0.4 * (current_price > sma_20) +
            0.3 * (macd > signal) +
            0.3 * (momentum_5d > 0)
        ) * (1.2 if is_low_vol else 0.8 if is_high_vol else 1.0)
        
        # Current position info
        position_size = current_holdings * current_price / portfolio_value if portfolio_value > 0 else 0
        unrealized_return = (current_price / df['Close'].iloc[current_step - 1] - 1) if position_size > 0 else 0
        
        # Define core signals with volatility-adjusted thresholds
        trend_signal = (
            current_price > sma_20 and
            macd > signal * (1.1 if is_high_vol else 1.0) and
            volume_ratio > (1.2 if is_high_vol else 1.0)
        )
        
        breakout_signal = (
            current_price > bb_upper and
            volume_ratio > (1.4 if is_high_vol else 1.2) and
            momentum_5d > 0
        )
        
        # Dynamic leverage based on signal strength and vol regime
        signal_strength = (
            (trend_signal * 0.6) +
            (breakout_signal * 0.4)
        ) * trend_strength * vol_trend_adj
        
        target_leverage = min(
            max_position_size,
            1.0 + (signal_strength * (max_position_size - 1.0))
        )
        
        # Risk Management Checks
        if position_size > 0:
            # Calculate max drawdown with position scaling
            max_drawdown = adjusted_drawdown * (1 + position_size * 0.5)
            
            # Emergency exit conditions
            if (
                drawdown < max_drawdown or
                unrealized_return < -stop_loss_pct or
                (is_high_vol and position_size > 1.3) or
                (rsi > (85 if is_low_vol else 80) and volume_ratio > 2.0)
            ):
                return 0  # Emergency exit
            
            # Profit taking with volatility-adjusted thresholds
            elif (
                rsi > (78 if is_low_vol else 73) and
                position_size > (1.5 if is_low_vol else 1.3) and
                (is_high_vol or volume_ratio > 1.6)
            ):
                return 1  # Reduce position
        
        # Entry and Position Sizing
        if position_size < target_leverage:
            # Strong entry conditions
            if (
                trend_signal and breakout_signal and
                rsi < (75 if is_low_vol else 70) and
                not is_high_vol and
                drawdown > (adjusted_drawdown * 0.7)
            ):
                return min(5, int(3 + 2.5 * trend_strength))
            
            # Regular entry conditions
            elif (
                trend_signal and
                rsi < (70 if is_low_vol else 65) and
                not is_high_vol and
                drawdown > (adjusted_drawdown * 0.5)
            ):
                return min(4, int(3 + 1.5 * trend_strength))
            
            # Light position entry
            elif (
                breakout_signal or
                (current_price > sma_20 and volume_ratio > 1.2)
            ) and not is_high_vol:
                return 3
        
        # Hold by default
        return 2
                
    except Exception as e:
        print(f"Warning: Error in strategy calculation: {e}")
        return 2  # Hold as fallback

def run_simulation(symbol='SPY', period='1y', initial_balance=10000, max_steps=None, strategy='mean_reversion'):
    """Run a trading simulation using real market data"""
    
    # Fetch market data
    df = get_market_data(symbol, period)
    if df is None:
        print("Failed to fetch market data. Simulation cancelled.")
        return None
    
    print(f"\nStarting simulation with {strategy} Strategy on {symbol}...")
    
    # Initialize environment
    env = TradingEnv(df)
    state = env.reset()
    env.balance = initial_balance
    
    done = False
    steps = 0
    trades = []
    
    print(f"Initial portfolio value: ${env.portfolio_value:.2f}")
    
    # Run simulation
    while not done and (max_steps is None or steps < max_steps):
        # Get action from strategy
        if strategy == 'mean_reversion':
            action = mean_reversion_strategy(state, df, steps)
        elif strategy == 'buy_and_hold':
            action = buy_and_hold_strategy(state, df, steps)
        else:
            print(f"Unknown strategy: {strategy}")
            break
            
        # Record trade if not holding
        if action != 2:
            trades.append({
                'step': steps,
                'action': ['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy', 'Max Leverage'][action],
                'price': df['Close'].iloc[steps],
                'portfolio_value': env.portfolio_value
            })
        
        # Take action
        next_state, reward, done, info = env.step(action)
        state = next_state
        steps += 1
        
        # Print progress every 20 steps
        if steps % 20 == 0:
            print(f"Step {steps}: Portfolio value: ${env.portfolio_value:.2f}")
    
    # Print final results
    print("\nSimulation finished!")
    print(f"Final portfolio value: ${env.portfolio_value:.2f}")
    print(f"Total return: {((env.portfolio_value / initial_balance - 1) * 100):.2f}%")
    print(f"Number of steps: {steps}")
    print(f"Number of trades: {len(trades)}\n")
    
    # Plot portfolio performance
    plt.figure(figsize=(12, 6))
    plt.plot(env.portfolio_values)
    plt.title(f'Portfolio Value Over Time - {strategy} Strategy')
    plt.xlabel('Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.savefig(f'portfolio_performance_{symbol}.png')
    plt.close()
    
    return env

if __name__ == "__main__":
    # Run simulation with buy_and_hold strategy
    env = run_simulation(
        symbol='SPY',
        period='1y',
        initial_balance=10000,
        strategy='buy_and_hold'
    )
