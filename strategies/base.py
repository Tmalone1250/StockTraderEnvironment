"""Base classes for trading strategies"""

class State:
    """
    Base class for storing strategy state
    Includes position tracking and risk management
    """
    def __init__(self):
        self.position_size = 0.0
        self.entry_price = None
        self.peak_value = None
        self.entry_time = None
        self.trailing_stop = None
        self.max_drawdown = 0.0
        self.volatility_regime = 'normal'  # low, normal, high
        self.consecutive_losses = 0
        self.trade_duration = 0
        
    def update_risk_metrics(self, current_price, portfolio_value, current_step):
        """Update risk management metrics"""
        if self.entry_price is None:
            self.entry_price = current_price
            self.entry_time = current_step
            self.peak_value = portfolio_value
            return
            
        # Update peak value and drawdown
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Update trade duration
        if self.position_size != 0:
            self.trade_duration = current_step - self.entry_time
        else:
            self.trade_duration = 0
            self.trailing_stop = None  # Reset trailing stop when not in position
            
        # Update trailing stop - more dynamic now
        if self.position_size > 0:
            if self.trailing_stop is None:
                self.trailing_stop = current_price * 0.93  # Initial 7% stop
            else:
                # Tighten stop as profit increases
                profit_pct = (current_price - self.entry_price) / self.entry_price
                if profit_pct > 0.1:  # In good profit
                    self.trailing_stop = max(self.trailing_stop, current_price * 0.97)  # Tighter 3% trail
                elif profit_pct > 0.05:  # Small profit
                    self.trailing_stop = max(self.trailing_stop, current_price * 0.95)  # 5% trail
                else:
                    self.trailing_stop = max(self.trailing_stop, current_price * 0.93)  # Normal 7% trail
                
    def should_exit_trade(self, current_price, volatility):
        """
        Determine if we should exit the trade based on risk metrics
        Returns: (bool) True if should exit, False otherwise
        """
        if self.position_size == 0 or self.entry_price is None:
            return False
            
        # Stop loss hit
        if self.trailing_stop and current_price < self.trailing_stop:
            return True
            
        # Max drawdown exceeded - more tolerant now
        if self.max_drawdown > 0.20:  # 20% max drawdown
            return True
            
        # Position decay based on time - longer holding period
        if self.trade_duration > 60:  # Start reducing position after 60 days
            return True
            
        # Volatility regime shift - more tolerant
        if volatility > 0.5:  # Very high volatility environment
            return True
            
        return False
        
    def get_position_size(self, base_size, volatility):
        """
        Calculate position size based on risk factors
        Returns: (float) Adjusted position size
        """
        # Base volatility adjustment - more aggressive
        vol_scalar = 1.0
        if volatility < 0.2:
            vol_scalar = 1.5  # Increase size in low vol
            self.volatility_regime = 'low'
        elif volatility > 0.4:
            vol_scalar = 0.6  # Less reduction in high vol
            self.volatility_regime = 'high'
        else:
            self.volatility_regime = 'normal'
            
        # Drawdown adjustment - more tolerant
        dd_scalar = 1.0 - (self.max_drawdown * 1.5)  # Less reduction for drawdown
        
        # Time decay adjustment - slower decay
        time_scalar = 1.0
        if self.trade_duration > 60:
            time_scalar = max(0.7, 1.0 - (self.trade_duration - 60) / 60)
            
        # Combine all adjustments
        adjusted_size = base_size * vol_scalar * dd_scalar * time_scalar
        return min(max(adjusted_size, 0.0), 1.0)  # Ensure between 0 and 1
