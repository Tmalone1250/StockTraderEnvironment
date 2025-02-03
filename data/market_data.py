"""Market data handling functions"""

import yfinance as yf
import pandas as pd
import numpy as np

def calculate_technical_indicators(df):
    """Calculate technical indicators for trading strategies"""
    # Moving Averages for Crossover Strategy
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['MA_Signal'] = (df['MA10'] > df['MA30']).astype(float)  # 1 for bullish, 0 for bearish
    
    # Price momentum
    df['ROC'] = df['Close'].pct_change(10) * 100  # 10-day Rate of Change
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BBUpper'] = df['SMA20'] + (std20 * 2)
    df['BBLower'] = df['SMA20'] - (std20 * 2)
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Volatility
    df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
    
    # ADX for trend strength
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    return df

def get_market_data(symbol='GOOGL', period='1y'):
    """
    Fetch market data for a given symbol and calculate technical indicators
    """
    print(f"\nFetched market data for {symbol}:")
    print(f"Period: {period}")
    
    # Fetch data
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    
    # Calculate basic metrics
    df['Returns'] = df['Close'].pct_change()
    
    # Fetch SPY data for market returns
    spy = yf.Ticker('SPY')
    spy_data = spy.history(period=period)
    df['Market_Returns'] = spy_data['Close'].pct_change()
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Number of trading days: {len(df)}")
    print(f"Current price: ${df['Close'].iloc[-1]:.2f}")
    print(f"Period return: {(df['Close'].iloc[-1]/df['Close'].iloc[0]-1)*100:.2f}%\n")
    
    return df
