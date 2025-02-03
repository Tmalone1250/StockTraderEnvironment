"""Market data handling functions"""

import yfinance as yf
import pandas as pd
import numpy as np

def get_market_data(symbol='NVDA', period='1y'):
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
    
    print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Number of trading days: {len(df)}")
    print(f"Current price: ${df['Close'].iloc[-1]:.2f}")
    print(f"Period return: {(df['Close'].iloc[-1]/df['Close'].iloc[0]-1)*100:.2f}%\n")
    
    return df
