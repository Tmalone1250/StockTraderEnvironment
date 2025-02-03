"""Main simulation runner"""

from data.market_data import get_market_data
from strategies.simple import buy_and_hold_strategy
from utils.simulation import run_simulation
from metrics.performance import calculate_performance_metrics
from utils.reporting import print_performance_report

def main():
    # Get market data
    df = get_market_data(symbol='NVDA', period='1y')
    
    # Run simulation
    portfolio_values, returns_series = run_simulation(buy_and_hold_strategy, df)
    
    # Calculate and print performance metrics
    metrics = calculate_performance_metrics(portfolio_values, returns_series, df, initial_capital=10000.0)
    print_performance_report(metrics)

if __name__ == "__main__":
    main()
