"""
Baseline forecasting models for SmartSense.

Implements simple but effective baseline forecasters:
- NaiveSeasonalForecaster: Seasonal naive with trend
- ETSForecaster: Exponential smoothing (statsmodels)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import seasonal_decompose

from ..base import BaseForecaster, ForecastResult


class NaiveSeasonalForecaster(BaseForecaster):
    """
    Seasonal naive forecaster with optional trend adjustment.
    
    Uses last seasonal cycle + linear trend for forecasting.
    Fast, interpretable baseline that often performs well.
    """
    
    def __init__(self, random_seed: int = 42, trend_adjust: bool = True):
        super().__init__(random_seed)
        self.trend_adjust = trend_adjust
        self._seasonal_values = None
        self._trend_slope = None
        
    def fit(
        self, 
        df: pd.DataFrame, 
        freq: str,
        target_col: str = "value",
        exog_cols: Optional[List[str]] = None,
        seasonal_periods: Optional[int] = None
    ) -> "NaiveSeasonalForecaster":
        """Fit seasonal naive model."""
        self._validate_input(df, target_col)
        
        self.freq = freq
        self.seasonal_periods = seasonal_periods or self._infer_seasonal_periods(freq)
        
        y = df[target_col].values
        self._fit_data = df.copy()
        self._target_col = target_col
        
        # Extract seasonal pattern from last complete cycle
        if len(y) >= self.seasonal_periods:
            self._seasonal_values = y[-self.seasonal_periods:]
        else:
            # If insufficient data, use simple repetition
            self._seasonal_values = np.tile(y, self.seasonal_periods)[:self.seasonal_periods]
        
        # Compute trend if enabled
        if self.trend_adjust and len(y) >= 2 * self.seasonal_periods:
            # Simple linear trend over last two cycles
            recent_data = y[-2 * self.seasonal_periods:]
            x = np.arange(len(recent_data))
            self._trend_slope = np.polyfit(x, recent_data, 1)[0]
        else:
            self._trend_slope = 0.0
        
        # Compute residuals for anomaly detection
        fitted_values = self._get_fitted_values(y)
        self._residuals = y[-len(fitted_values):] - fitted_values
        
        self.is_fitted = True
        return self
    
    def predict(
        self, 
        h: int,
        exog_future: Optional[pd.DataFrame] = None,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ) -> ForecastResult:
        """Generate seasonal naive forecasts."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Generate base seasonal forecast
        n_full_cycles = h // self.seasonal_periods
        remainder = h % self.seasonal_periods
        
        base_forecast = np.tile(self._seasonal_values, n_full_cycles)
        if remainder > 0:
            base_forecast = np.concatenate([base_forecast, self._seasonal_values[:remainder]])
        
        # Add trend adjustment
        if self.trend_adjust:
            trend_adjustment = self._trend_slope * np.arange(1, h + 1)
            base_forecast += trend_adjustment
        
        # Generate future timestamps
        last_timestamp = self._fit_data.index[-1]
        future_index = self._generate_future_index(last_timestamp, h)
        
        # Compute prediction intervals using residual bootstrap
        residual_std = np.std(self._residuals)
        
        # Simple normal approximation for PI
        z_scores = {0.1: -1.645, 0.5: 0.0, 0.9: 1.645}
        yhat_lower = base_forecast + z_scores[0.1] * residual_std
        yhat_upper = base_forecast + z_scores[0.9] * residual_std
        
        return ForecastResult(
            timestamps=future_index,
            yhat=base_forecast,
            yhat_lower=yhat_lower,
            yhat_upper=yhat_upper,
            residuals=self._residuals,
            model_info=self.get_model_info()
        )
    
    def _get_fitted_values(self, y: np.ndarray) -> np.ndarray:
        """Get fitted values for residual computation."""
        if len(y) < self.seasonal_periods:
            return np.full(len(y), np.mean(y))
        
        fitted = []
        for i in range(len(y)):
            if i < self.seasonal_periods:
                # Use mean for initial values
                fitted.append(np.mean(y[:self.seasonal_periods]))
            else:
                # Use seasonal lag
                seasonal_idx = i % self.seasonal_periods
                fitted.append(self._seasonal_values[seasonal_idx])
        
        return np.array(fitted)


class ETSForecaster(BaseForecaster):
    """
    Exponential Smoothing (ETS) forecaster using statsmodels.
    
    Automatically selects best ETS configuration (Error, Trend, Seasonal).
    More sophisticated than naive but still fast and interpretable.
    """
    
    def __init__(self, random_seed: int = 42, auto_select: bool = True):
        super().__init__(random_seed)
        self.auto_select = auto_select
        self._ets_model = None
        self._fitted_model = None
        
    def fit(
        self, 
        df: pd.DataFrame, 
        freq: str,
        target_col: str = "value",
        exog_cols: Optional[List[str]] = None,
        seasonal_periods: Optional[int] = None
    ) -> "ETSForecaster":
        """Fit ETS model with automatic configuration selection."""
        self._validate_input(df, target_col)
        
        self.freq = freq
        self.seasonal_periods = seasonal_periods or self._infer_seasonal_periods(freq)
        
        y = df[target_col].values
        self._fit_data = df.copy()
        self._target_col = target_col
        
        # Configure ETS model
        if self.auto_select:
            # Try different configurations and select best AIC
            best_aic = float('inf')
            best_config = None
            
            configs = [
                ('add', 'add', 'add'),
                ('add', 'add', None),
                ('add', None, 'add'),
                ('add', None, None),
            ]
            
            for error, trend, seasonal in configs:
                try:
                    if seasonal and len(y) < 2 * self.seasonal_periods:
                        continue  # Skip seasonal if insufficient data
                    
                    model = ETSModel(
                        y, 
                        error=error, 
                        trend=trend, 
                        seasonal=seasonal,
                        seasonal_periods=self.seasonal_periods if seasonal else None
                    )
                    fitted = model.fit(disp=False)
                    
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_config = (error, trend, seasonal)
                        
                except Exception:
                    continue
            
            if best_config is None:
                # Fallback to simple exponential smoothing
                best_config = ('add', None, None)
        else:
            # Default configuration
            best_config = ('add', 'add', 'add' if len(y) >= 2 * self.seasonal_periods else None)
        
        # Fit final model
        error, trend, seasonal = best_config
        self._ets_model = ETSModel(
            y,
            error=error,
            trend=trend, 
            seasonal=seasonal,
            seasonal_periods=self.seasonal_periods if seasonal else None
        )
        
        self._fitted_model = self._ets_model.fit(disp=False)
        
        # Compute residuals
        self._residuals = self._fitted_model.resid
        
        self.is_fitted = True
        return self
    
    def predict(
        self, 
        h: int,
        exog_future: Optional[pd.DataFrame] = None,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ) -> ForecastResult:
        """Generate ETS forecasts with prediction intervals."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Generate forecasts
        forecast_result = self._fitted_model.forecast(steps=h)
        
        # Get prediction intervals if available
        try:
            pred_int = self._fitted_model.get_prediction(
                start=len(self._fitted_model.fittedvalues),
                end=len(self._fitted_model.fittedvalues) + h - 1
            )
            conf_int = pred_int.conf_int(alpha=0.2)  # 80% CI (10% to 90%)
            yhat_lower = conf_int.iloc[:, 0].values
            yhat_upper = conf_int.iloc[:, 1].values
        except Exception:
            # Fallback to simple residual-based intervals
            residual_std = np.std(self._residuals)
            yhat_lower = forecast_result - 1.645 * residual_std
            yhat_upper = forecast_result + 1.645 * residual_std
        
        # Generate future timestamps
        last_timestamp = self._fit_data.index[-1]
        future_index = self._generate_future_index(last_timestamp, h)
        
        return ForecastResult(
            timestamps=future_index,
            yhat=forecast_result,
            yhat_lower=yhat_lower,
            yhat_upper=yhat_upper,
            residuals=self._residuals,
            model_info=self.get_model_info()
        )
    
    def get_model_info(self) -> Dict:
        """Get ETS model information."""
        base_info = super().get_model_info()
        if self._fitted_model:
            base_info.update({
                "ets_config": {
                    "error": self._ets_model.error,
                    "trend": self._ets_model.trend,
                    "seasonal": self._ets_model.seasonal
                },
                "aic": self._fitted_model.aic,
                "params": self._fitted_model.params.to_dict() if hasattr(self._fitted_model.params, 'to_dict') else str(self._fitted_model.params)
            })
        return base_info
