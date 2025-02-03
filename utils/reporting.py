"""Reporting utilities"""

def print_performance_report(metrics):
    """Print a formatted performance report"""
    print("\n=== Performance Report ===")
    print(f"Strategy Return: {metrics['strategy_return']:.2f}%")
    print(f"Benchmark Return: {metrics['benchmark_return']:.2f}%")
    print(f"Excess Return: {metrics['excess_return']:.2f}%\n")
    
    print("Risk Metrics:")
    print(f"Volatility: {metrics['volatility']:.2f}%")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}\n")
    
    print("Risk-Adjusted Metrics:")
    print(f"Alpha: {metrics['alpha']:.2f}%")
    print(f"Beta: {metrics['beta']:.2f}")
    print(f"Information Ratio: {metrics['information_ratio']:.2f}\n")
    
    print("Trading Statistics:")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Avg Daily Return: {metrics['avg_daily_return']:.2f}%")
    print(f"Avg Up Day: {metrics['avg_up_day']:.2f}%")
    print(f"Avg Down Day: {metrics['avg_down_day']:.2f}%\n")
    
    print("Market Condition Performance:")
    print(f"Bull Market Performance: {metrics['bull_market_performance']:.2f}%")
    print(f"Bear Market Performance: {metrics['bear_market_performance']:.2f}%\n")
    
    print("=========================")
