"""Trading strategies package"""

from .base import State
from .implementations import (
    buy_and_hold_strategy,
    moving_average_crossover_strategy,
    mean_reversion_strategy,
    bollinger_bands_strategy
)

__all__ = [
    'State',
    'buy_and_hold_strategy',
    'moving_average_crossover_strategy',
    'mean_reversion_strategy',
    'bollinger_bands_strategy'
]
