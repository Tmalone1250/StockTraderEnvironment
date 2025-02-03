import pandas as pd
import numpy as np
from StockTradingEnv import StockTradingEnv
import matplotlib.pyplot as plt

def generate_test_data(days=252):
    """Generate synthetic stock data with mean-reverting characteristics"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Generate mean-reverting price series
    price = 100
    prices = []
    mean_price = 100
    mean_reversion_speed = 0.1
    volatility = 0.015
    
    for i in range(days):
        # Add mean reversion component
        price += mean_reversion_speed * (mean_price - price)
        
        # Add random walk component
        price *= (1 + np.random.normal(0, volatility))
        
        # Slowly varying mean to create trends
        if i % 63 == 0:  # Change mean every quarter
            mean_price *= (1 + np.random.normal(0, 0.05))
            
        prices.append(price)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Close': prices,
    }, index=dates)
    
    # Calculate technical indicators
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # Calculate Z-score (deviation from moving mean in standard deviations)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['STD_20'] = df['Close'].rolling(window=20).std()
    df['Z_Score'] = (df['Close'] - df['SMA_20']) / df['STD_20']
    
    # Momentum indicators
    df['ROC_5'] = df['Close'].pct_change(periods=5)  # 5-day Rate of Change
    df['ROC_20'] = df['Close'].pct_change(periods=20)  # 20-day Rate of Change
    
    # RSI for momentum confirmation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df

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
        # More aggressive Z-score thresholds
        if z_score > 1.5:  # Reduced from 2.0
            score -= 1.5  # Increased from 1.0
        elif z_score > 1.0:  # Reduced from 1.5
            score -= 0.8  # Increased from 0.5
        elif z_score < -1.5:  # Increased from -2.0
            score += 1.5  # Increased from 1.0
        elif z_score < -1.0:  # Increased from -1.5
            score += 0.8  # Increased from 0.5
            
        # 2. Momentum Filter (30% weight)
        momentum_score = 0
        
        # More aggressive RSI signals
        if rsi > 65:  # Reduced from 70
            momentum_score -= 0.4  # Increased from 0.3
        elif rsi < 35:  # Increased from 30
            momentum_score += 0.4  # Increased from 0.3
            
        # More aggressive Rate of Change signals
        if roc_5 > 0:  # Any positive momentum
            if roc_20 > 0:  # Confirmed by longer trend
                momentum_score += 0.4  # Increased from 0.3
            else:
                momentum_score += 0.2  # New condition
        elif roc_5 < 0:  # Any negative momentum
            if roc_20 < 0:  # Confirmed by longer trend
                momentum_score -= 0.4  # Increased from 0.3
            else:
                momentum_score -= 0.2  # New condition
            
        score += momentum_score
        
        # 3. Volatility Adjustment (20% weight)
        vol_percentile = pd.Series(current_volatility).rank(pct=True)[0]
        if vol_percentile > 0.8:  # High volatility
            # Less conservative in volatility adjustment
            score = 2 + (score - 2) * 0.7  # Changed from 0.5 to 0.7
        
        # More aggressive thresholds for actions
        if score >= 2.8:  # Reduced from 3.0
            return 4  # Strong Buy
        elif score >= 2.3:  # Reduced from 2.5
            return 3  # Buy
        elif score <= 1.2:  # Increased from 1.0
            return 0  # Strong Sell
        elif score <= 1.7:  # Increased from 1.5
            return 1  # Sell
        else:
            return 2  # Hold
            
    except Exception as e:
        print(f"Warning: Error in strategy calculation: {e}")
        return 2  # Hold as fallback

def run_simulation(initial_balance=10000, max_steps=100):
    """Run a trading simulation using synthetic data"""
    
    # Generate test data
    df = generate_test_data()
    
    # Create environment
    env = StockTradingEnv(df, initial_balance=initial_balance, live_trading=False)
    
    # Run one episode
    state = env.reset()
    done = False
    steps = 0
    trades = []
    
    print("\nStarting simulation with Mean Reversion Strategy...")
    print(f"Initial portfolio value: ${env.portfolio_value:.2f}")
    
    while not done:
        # Get action from strategy
        action = mean_reversion_strategy(state, df, steps)
        
        # Record trade
        if action != 2:  # If not holding
            trades.append({
                'step': steps,
                'action': ['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy'][action],
                'price': state[0],
                'portfolio_value': env.portfolio_value,
                'z_score': df['Z_Score'].iloc[steps]
            })
        
        # Take action
        state, reward, done, info = env.step(action)
        steps += 1
        
        # Print progress every 20 steps
        if steps % 20 == 0:
            print(f"Step {steps}: Portfolio value: ${env.portfolio_value:.2f}")
        
        # Optional step limit
        if max_steps and steps >= max_steps:
            break
    
    # Print final results
    print("\nSimulation finished!")
    print(f"Final portfolio value: ${env.portfolio_value:.2f}")
    print(f"Total return: {((env.portfolio_value - initial_balance) / initial_balance * 100):.2f}%")
    print(f"Number of steps: {steps}")
    print(f"Number of trades: {len(trades)}")
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot portfolio value
    ax1.plot(env.portfolio_values)
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_xlabel('Trading Steps')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True)
    
    # Plot price and Z-score
    price_data = df['Close'].iloc[:steps]
    z_scores = df['Z_Score'].iloc[:steps]
    
    ax2.plot(price_data, label='Price', alpha=0.7)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(z_scores, '--', color='orange', label='Z-Score', alpha=0.5)
    
    # Add trade markers
    for trade in trades:
        if 'Buy' in trade['action']:
            ax2.scatter(trade['step'], trade['price'], color='green', marker='^', s=100)
        elif 'Sell' in trade['action']:
            ax2.scatter(trade['step'], trade['price'], color='red', marker='v', s=100)
    
    ax2.set_title('Price Action with Z-Score and Trade Signals')
    ax2.set_xlabel('Trading Steps')
    ax2.set_ylabel('Price ($)')
    ax2.grid(True)
    
    # Adjust legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot RSI
    ax3.plot(df['RSI'].iloc[:steps], label='RSI')
    ax3.axhline(y=70, color='r', linestyle='--', label='Overbought')
    ax3.axhline(y=30, color='g', linestyle='--', label='Oversold')
    ax3.set_title('RSI Indicator')
    ax3.set_xlabel('Trading Steps')
    ax3.set_ylabel('RSI')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('portfolio_performance.png')
    plt.close()
    
    print("\nPortfolio performance plot saved as 'portfolio_performance.png'")
    
    return env

if __name__ == "__main__":
    # Run simulation
    env = run_simulation(
        initial_balance=10000,
        max_steps=100
    )
