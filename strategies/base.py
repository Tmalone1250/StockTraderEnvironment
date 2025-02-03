"""Base strategy classes and utilities"""

class State:
    """
    Class to hold strategy state
    """
    def __init__(self):
        self.position_size = None
        self.entry_price = None
        self.last_signal = None
        self.trend_state = None
        self.volatility_regime = None
