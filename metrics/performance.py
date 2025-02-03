"""Performance metrics calculation"""

import numpy as np
import pandas as pd

def calculate_performance_metrics(portfolio_values, returns_series, df, initial_capital):
    """Calculate comprehensive performance metrics"""
    
    # Basic performance
    final_portfolio_value = portfolio_values.iloc[-1]
    strategy_return = (final_portfolio_value / initial_capital - 1) * 100
    benchmark_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    excess_return = strategy_return - benchmark_return
    
    # Risk metrics
    volatility = returns_series.std() * np.sqrt(252) * 100
    rolling_max = portfolio_values.expanding().max()
    drawdowns = (portfolio_values - rolling_max) / rolling_max * 100
    max_drawdown = drawdowns.min()
    
    # Risk-adjusted metrics
    risk_free_rate = 0.02  # Assumed 2% risk-free rate
    excess_returns = returns_series - risk_free_rate/252
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns_series.std() if returns_series.std() != 0 else 0
    
    # Sortino ratio
    downside_returns = returns_series[returns_series < 0]
    sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() != 0 else 0
    
    # Market metrics
    market_returns = df['Market_Returns'].fillna(0)
    strategy_returns = returns_series.fillna(0)
    
    # Calculate beta and alpha
    if len(market_returns) > 1 and len(strategy_returns) > 1:
        try:
            cov_matrix = np.cov(strategy_returns.values, market_returns.values)
            if cov_matrix.shape == (2, 2):
                covariance = cov_matrix[0,1]
                market_variance = np.var(market_returns.values)
                beta = covariance / market_variance if market_variance != 0 else 1
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
    winning_days = np.sum(returns_series > 0)
    total_days = len(returns_series)
    win_rate = (winning_days / total_days * 100) if total_days > 0 else 0
    avg_daily_return = returns_series.mean() * 100
    avg_up_day = returns_series[returns_series > 0].mean() * 100 if len(returns_series[returns_series > 0]) > 0 else 0
    avg_down_day = returns_series[returns_series < 0].mean() * 100 if len(returns_series[returns_series < 0]) > 0 else 0
    
    # Market condition performance
    bull_returns = returns_series[market_returns > 0].sum() * 100
    bear_returns = returns_series[market_returns < 0].sum() * 100
    
    return {
        'strategy_return': strategy_return,
        'benchmark_return': benchmark_return,
        'excess_return': excess_return,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'alpha': alpha,
        'beta': beta,
        'information_ratio': information_ratio,
        'win_rate': win_rate,
        'avg_daily_return': avg_daily_return,
        'avg_up_day': avg_up_day,
        'avg_down_day': avg_down_day,
        'bull_market_performance': bull_returns,
        'bear_market_performance': bear_returns
    }
