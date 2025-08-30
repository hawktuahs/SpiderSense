"""
BaseForecaster interface for SmartSense forecasting models.

Provides a unified API for all forecasting adapters with support for:
- Prediction intervals (quantiles)
- External variables (weather, holidays)
- Multi-step ahead forecasting
- Residual computation for anomaly detection
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class ForecastResult:
    """Container for forecast results with prediction intervals."""
    timestamps: pd.DatetimeIndex
    yhat: np.ndarray  # Point forecasts
    yhat_lower: Optional[np.ndarray] = None  # Lower PI (e.g., 10%)
    yhat_upper: Optional[np.ndarray] = None  # Upper PI (e.g., 90%)
    residuals: Optional[np.ndarray] = None  # In-sample residuals
    model_info: Optional[Dict] = None  # Model-specific metadata


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models in SmartSense.
    
    Design principles:
    - CPU-friendly: All models must run efficiently on CPU-only environments
    - Deterministic: Fixed random seeds for reproducible results
    - Memory efficient: <1GB memory usage target
    - Fast inference: <10s cold start on free tier
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.is_fitted = False
        self.freq = None
        self.seasonal_periods = None
        self._model = None
        
    @abstractmethod
    def fit(
        self, 
        df: pd.DataFrame, 
        freq: str,
        target_col: str = "value",
        exog_cols: Optional[List[str]] = None,
        seasonal_periods: Optional[int] = None
    ) -> "BaseForecaster":
        """
        Fit the forecasting model.
        
        Args:
            df: Time series data with DatetimeIndex
            freq: Frequency string ('H', 'D', etc.)
            target_col: Name of target variable column
            exog_cols: List of exogenous variable columns
            seasonal_periods: Number of periods in seasonal cycle
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(
        self, 
        h: int,
        exog_future: Optional[pd.DataFrame] = None,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ) -> ForecastResult:
        """
        Generate forecasts with prediction intervals.
        
        Args:
            h: Forecast horizon (number of steps)
            exog_future: Future exogenous variables
            quantiles: Quantile levels for prediction intervals
            
        Returns:
            ForecastResult with timestamps, forecasts, and intervals
        """
        pass
    
    def get_residuals(self) -> np.ndarray:
        """
        Get in-sample residuals for anomaly detection.
        
        Returns:
            Array of residuals (y_true - y_pred)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing residuals")
        return self._residuals
    
    def get_model_info(self) -> Dict:
        """Get model-specific information and parameters."""
        return {
            "model_type": self.__class__.__name__,
            "is_fitted": self.is_fitted,
            "freq": self.freq,
            "seasonal_periods": self.seasonal_periods,
            "random_seed": self.random_seed
        }
    
    def _validate_input(self, df: pd.DataFrame, target_col: str) -> None:
        """Validate input data format and requirements."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        if df[target_col].isna().sum() > len(df) * 0.1:
            raise ValueError("Too many missing values in target column (>10%)")
    
    def _infer_seasonal_periods(self, freq: str) -> int:
        """Infer seasonal periods based on frequency."""
        seasonal_map = {
            'H': 24,      # Hourly: daily seasonality
            'D': 7,       # Daily: weekly seasonality  
            'W': 52,      # Weekly: yearly seasonality
            'M': 12,      # Monthly: yearly seasonality
        }
        return seasonal_map.get(freq, 1)
    
    def _generate_future_index(self, last_timestamp: pd.Timestamp, h: int) -> pd.DatetimeIndex:
        """Generate future timestamp index for forecasts."""
        return pd.date_range(
            start=last_timestamp + pd.Timedelta(self.freq),
            periods=h,
            freq=self.freq
        )
