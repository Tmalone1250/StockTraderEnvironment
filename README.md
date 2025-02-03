# High-Risk Stock Trading AI with Robinhood Integration

This project implements a high-risk stock trading AI environment with Robinhood integration for both paper trading and live trading capabilities.

## Features

- High-risk trading strategies with up to 2x leverage
- Short selling capabilities with margin requirements
- Advanced risk management system:
  - Position size limits
  - Stop-loss mechanisms
  - Trailing stops
  - Volatility-based position sizing
- Real-time trading through Robinhood API
- Support for both simulation and live trading modes
- Historical data analysis using yfinance
- Reinforcement learning environment compatible with OpenAI Gym

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Copy `.env.example` to `.env` and fill in your Robinhood credentials:
```bash
cp .env.example .env
```

## Configuration

Edit `.env` file with your Robinhood credentials:
```
ROBINHOOD_USERNAME=your_email@example.com
ROBINHOOD_PASSWORD=your_password
ROBINHOOD_MFA_CODE=your_mfa_code  # Optional: MFA secret key for 2FA
```

## Usage

### Simulation Mode
```python
import pandas as pd
from StockTradingEnv import StockTradingEnv

# Load historical data
data = pd.read_csv('stock_data.csv')

# Create environment in simulation mode
env = StockTradingEnv(data, initial_balance=10000, live_trading=False)

# Run simulation
state = env.reset()
done = False
while not done:
    action = your_trading_strategy(state)  # Your trading logic here
    next_state, reward, done, info = env.step(action)
    state = next_state
```

### Live Trading Mode
```python
import yfinance as yf
from StockTradingEnv import StockTradingEnv

# Get real-time data
ticker = 'AAPL'
data = yf.download(ticker, start='2024-01-01')

# Create environment in live trading mode
env = StockTradingEnv(data, live_trading=True)

# Run live trading
state = env.reset()
done = False
while not done:
    action = your_trading_strategy(state)  # Your trading logic here
    next_state, reward, done, info = env.step(action)
    state = next_state

# Cleanup
env.close()
```

## Risk Management

The environment includes several risk management features:

1. Position Limits: Maximum 25% of portfolio in single position
2. Leverage Limits: Maximum 2x leverage
3. Stop Losses:
   - Fixed stop loss at -15%
   - Trailing stop at -10%
   - Portfolio-wide drawdown limit at -25%
4. Volatility Scaling: Position sizes are adjusted based on market volatility

## Warning

High-risk trading strategies can result in significant losses. This software is for educational purposes only. Always paper trade first and understand the risks before live trading.

## License

MIT License
