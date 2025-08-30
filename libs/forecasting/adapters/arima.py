"""
ARIMA forecasting adapter for SmartSense.

Uses pmdarima for automatic ARIMA model selection with:
- Automatic order selection (p,d,q) and seasonal (P,D,Q,s)
- CPU-optimized configuration for fast fitting
- Prediction intervals via model variance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings
from pmdarima import auto_arima
from pmdarima.arima import ARIMA

from ..base import BaseForecaster, ForecastResult


class ARIMAForecaster(BaseForecaster):
    """
    Auto-ARIMA forecaster using pmdarima.
    
    Automatically selects optimal ARIMA(p,d,q)(P,D,Q,s) configuration
    with CPU-friendly constraints for fast fitting on free tier.
    """
    
    def __init__(
        self, 
        random_seed: int = 42,
        max_p: int = 3,
        max_q: int = 3,
        max_P: int = 2,
        max_Q: int = 2,
        max_d: int = 2,
        max_D: int = 1
    ):
        super().__init__(random_seed)
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self.max_d = max_d
        self.max_D = max_D
        self._arima_model = None
        
    def fit(
        self, 
        df: pd.DataFrame, 
        freq: str,
        target_col: str = "value",
        exog_cols: Optional[List[str]] = None,
        seasonal_periods: Optional[int] = None
    ) -> "ARIMAForecaster":
        """Fit auto-ARIMA model with CPU-optimized constraints."""
        self._validate_input(df, target_col)
        
        self.freq = freq
        self.seasonal_periods = seasonal_periods or self._infer_seasonal_periods(freq)
        
        y = df[target_col].values
        self._fit_data = df.copy()
        self._target_col = target_col
        
        # Prepare exogenous variables
        exog = None
        if exog_cols:
            exog = df[exog_cols].values
        
        # Configure auto-ARIMA with CPU-friendly constraints
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Determine if seasonal modeling is feasible
            use_seasonal = (
                len(y) >= 2 * self.seasonal_periods and 
                self.seasonal_periods > 1 and
                self.seasonal_periods <= 52  # Reasonable seasonal period limit
            )
            
            try:
                self._arima_model = auto_arima(
                    y,
                    exogenous=exog,
                    seasonal=use_seasonal,
                    m=self.seasonal_periods if use_seasonal else 1,
                    max_p=self.max_p,
                    max_q=self.max_q,
                    max_P=self.max_P if use_seasonal else 0,
                    max_Q=self.max_Q if use_seasonal else 0,
                    max_d=self.max_d,
                    max_D=self.max_D if use_seasonal else 0,
                    stepwise=True,  # Faster stepwise search
                    approximation=True,  # Faster CSS approximation
                    error_action='ignore',
                    suppress_warnings=True,
                    random_state=self.random_seed,
                    n_jobs=1,  # Single thread for consistency
                    maxiter=50  # Limit iterations for speed
                )
            except Exception as e:
                # Fallback to simple ARIMA(1,1,1) if auto-selection fails
                print(f"Auto-ARIMA failed, using fallback: {e}")
                self._arima_model = ARIMA(
                    order=(1, 1, 1),
                    seasonal_order=(0, 0, 0, 0),
                    suppress_warnings=True
                ).fit(y, exogenous=exog)
        
        # Compute residuals
        self._residuals = self._arima_model.resid()
        
        self.is_fitted = True
        return self
    
    def predict(
        self, 
        h: int,
        exog_future: Optional[pd.DataFrame] = None,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ) -> ForecastResult:
        """Generate ARIMA forecasts with prediction intervals."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare future exogenous variables
        exog_future_array = None
        if exog_future is not None:
            exog_future_array = exog_future.values
        
        # Generate forecasts with confidence intervals
        try:
            forecast_result, conf_int = self._arima_model.predict(
                n_periods=h,
                exogenous=exog_future_array,
                return_conf_int=True,
                alpha=0.2  # 80% confidence interval (10% to 90%)
            )
            
            yhat = forecast_result
            yhat_lower = conf_int[:, 0]
            yhat_upper = conf_int[:, 1]
            
        except Exception as e:
            # Fallback to point forecasts only
            print(f"Confidence intervals failed, using point forecasts: {e}")
            yhat = self._arima_model.predict(
                n_periods=h,
                exogenous=exog_future_array
            )
            
            # Simple residual-based intervals
            residual_std = np.std(self._residuals)
            yhat_lower = yhat - 1.645 * residual_std
            yhat_upper = yhat + 1.645 * residual_std
        
        # Generate future timestamps
        last_timestamp = self._fit_data.index[-1]
        future_index = self._generate_future_index(last_timestamp, h)
        
        return ForecastResult(
            timestamps=future_index,
            yhat=yhat,
            yhat_lower=yhat_lower,
            yhat_upper=yhat_upper,
            residuals=self._residuals,
            model_info=self.get_model_info()
        )
    
    def get_model_info(self) -> Dict:
        """Get ARIMA model information."""
        base_info = super().get_model_info()
        if self._arima_model:
            try:
                base_info.update({
                    "arima_order": self._arima_model.order,
                    "seasonal_order": self._arima_model.seasonal_order,
                    "aic": self._arima_model.aic(),
                    "bic": self._arima_model.bic(),
                    "params": self._arima_model.params().tolist() if hasattr(self._arima_model.params(), 'tolist') else str(self._arima_model.params())
                })
            except Exception:
                base_info.update({"arima_info": "Model info unavailable"})
        return base_info
    
    def update(self, new_data: np.ndarray, exog: Optional[np.ndarray] = None) -> "ARIMAForecaster":
        """
        Update ARIMA model with new observations (online learning).
        
        Args:
            new_data: New target observations
            exog: New exogenous variables (if used)
            
        Returns:
            Self for method chaining
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before updating")
        
        try:
            # Update model with new observations
            self._arima_model = self._arima_model.append(new_data, exogenous=exog)
            
            # Update residuals
            self._residuals = self._arima_model.resid()
            
        except Exception as e:
            print(f"Model update failed: {e}")
            # Continue with existing model if update fails
        
        return self
