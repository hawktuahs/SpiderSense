"""
SmartSense Forecasting Library

Modular time series forecasting with BaseForecaster interface
and adapters for various models (ARIMA, ETS, NHITS, TSFM).
"""

from .base import BaseForecaster
from .adapters.baseline import NaiveSeasonalForecaster, ETSForecaster
from .adapters.arima import ARIMAForecaster

try:
    from .adapters.nhits_tiny import NHITSTinyForecaster
    NHITS_AVAILABLE = True
except ImportError:
    NHITS_AVAILABLE = False

__all__ = [
    "BaseForecaster",
    "NaiveSeasonalForecaster", 
    "ETSForecaster",
    "ARIMAForecaster"
]

if NHITS_AVAILABLE:
    __all__.append("NHITSTinyForecaster")
