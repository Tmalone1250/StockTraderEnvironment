"""Main simulation script"""

from data.market_data import get_market_data
from strategies.rl_strategy import rl_enhanced_strategy, RLState
from utils.simulation import run_simulation

def main():
    # Get market data
    df = get_market_data(symbol='GOOGL', period='1y')
    
    # Run simulation with RL strategy
    initial_state = RLState()
    final_portfolio_value, performance_metrics = run_simulation(df, rl_enhanced_strategy, initial_state)

if __name__ == "__main__":
    main()
