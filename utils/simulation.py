"""Simulation utilities"""

import numpy as np
import pandas as pd
from typing import Callable, Tuple, List

def run_simulation(df: pd.DataFrame, strategy: Callable, state, initial_capital: float = 10000.0) -> Tuple[List[float], pd.Series]:
    """
    Run a trading simulation using the provided strategy
    
    Args:
        df: DataFrame with market data
        strategy: Trading strategy function
        state: Trading state object
        initial_capital: Initial capital for the simulation
        
    Returns:
        Tuple of (portfolio_values, returns_series)
    """
    portfolio_values = [initial_capital]
    returns = []
    current_position = 0.0
    
    # Run simulation step by step
    print(f"Initial portfolio value: ${initial_capital:.2f}")
    
    for step in range(len(df)):
        if step % 20 == 0:
            print(f"Step {step}: Portfolio value: ${portfolio_values[-1]:.2f}")
            
        # Get strategy position
        position, state = strategy(state, df, step)
        
        # Calculate returns
        if step > 0:
            price_return = df['Returns'].iloc[step]
            portfolio_return = price_return * current_position
            portfolio_value = portfolio_values[-1] * (1 + portfolio_return)
            portfolio_values.append(portfolio_value)
            returns.append(portfolio_return)
        
        current_position = position
    
    returns_series = pd.Series(returns, index=df.index[1:])
    
    # Calculate performance metrics
    total_return = (portfolio_values[-1] / initial_capital - 1) * 100
    benchmark_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    excess_return = total_return - benchmark_return
    
    # Risk metrics
    volatility = np.std(returns) * np.sqrt(252) * 100
    max_drawdown = calculate_max_drawdown(portfolio_values) * 100
    sharpe_ratio = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
    sortino_ratio = calculate_sortino_ratio(returns)
    
    # Calculate alpha and beta
    market_returns = df['Market_Returns'].iloc[1:].values
    beta = calculate_beta(returns, market_returns)
    alpha = calculate_alpha(returns, market_returns, beta) * 100
    
    # Trading statistics
    win_rate = len([r for r in returns if r > 0]) / len(returns) * 100
    avg_daily_return = np.mean(returns) * 100
    avg_up_day = np.mean([r for r in returns if r > 0]) * 100 if any(r > 0 for r in returns) else 0
    avg_down_day = np.mean([r for r in returns if r < 0]) * 100 if any(r < 0 for r in returns) else 0
    
    # Market condition performance
    bull_returns = [r for r, m in zip(returns, market_returns) if m > 0]
    bear_returns = [r for r, m in zip(returns, market_returns) if m <= 0]
    bull_performance = np.mean(bull_returns) * 252 * 100 if bull_returns else 0
    bear_performance = np.mean(bear_returns) * 252 * 100 if bear_returns else 0
    
    print("\n=== Performance Report ===")
    print(f"Strategy Return: {total_return:.2f}%")
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
    print(f"Information Ratio: {excess_return/volatility:.2f}\n")
    
    print("Trading Statistics:")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Avg Daily Return: {avg_daily_return:.2f}%")
    print(f"Avg Up Day: {avg_up_day:.2f}%")
    print(f"Avg Down Day: {avg_down_day:.2f}%\n")
    
    print("Market Condition Performance:")
    print(f"Bull Market Performance: {bull_performance:.2f}%")
    print(f"Bear Market Performance: {bear_performance:.2f}%\n")
    
    print("=========================")
    
    return portfolio_values, returns_series

def calculate_max_drawdown(portfolio_values: List[float]) -> float:
    """Calculate maximum drawdown from portfolio values"""
    peak = portfolio_values[0]
    max_drawdown = 0
    
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    return max_drawdown

def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio"""
    returns = np.array(returns)
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        return 0
    
    excess_return = np.mean(returns) - risk_free_rate
    downside_std = np.std(negative_returns)
    
    return (excess_return * np.sqrt(252)) / downside_std if downside_std > 0 else 0

def calculate_beta(returns: np.ndarray, market_returns: np.ndarray) -> float:
    """Calculate portfolio beta"""
    if len(returns) != len(market_returns):
        return 0
    
    covariance = np.cov(returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    
    return covariance / market_variance if market_variance > 0 else 0

def calculate_alpha(returns: np.ndarray, market_returns: np.ndarray, beta: float) -> float:
    """Calculate portfolio alpha"""
    if len(returns) != len(market_returns):
        return 0
    
    return np.mean(returns) - beta * np.mean(market_returns)
