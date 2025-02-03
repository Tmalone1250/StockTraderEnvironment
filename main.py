"""Main simulation script"""

from data.market_data import get_market_data
from utils.simulation import run_simulation
from strategies.ml_strategy import ml_enhanced_strategy, MLState

def main():
    # Get market data
    df = get_market_data('GOOGL', '1y')
    
    # Run simulation with ML enhanced strategy
    initial_state = MLState()
    run_simulation(df, ml_enhanced_strategy, initial_state)

if __name__ == "__main__":
    main()
