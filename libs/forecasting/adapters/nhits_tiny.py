"""
NHITS-tiny adapter for SmartSense.

CPU-optimized neural forecasting using Darts library with minimal
memory footprint and fast training for free-tier deployment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings
import logging

try:
    from darts import TimeSeries
    from darts.models import NHiTSModel
    import torch
    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False

from ..base import BaseForecaster, ForecastResult

logger = logging.getLogger(__name__)


class NHITSTinyForecaster(BaseForecaster):
    """
    CPU-optimized NHITS model for energy load forecasting.
    
    Uses minimal architecture with constraints for fast training
    and inference on CPU-only environments.
    """
    
    def __init__(
        self,
        random_seed: int = 42,
        input_chunk_length: int = 24,
        output_chunk_length: int = 12,
        num_stacks: int = 2,
        num_blocks: int = 1,
        num_layers: int = 2,
        layer_widths: int = 32,
        pooling_kernel_sizes: Optional[List[int]] = None,
        n_epochs: int = 50,
        batch_size: int = 16
    ):
        """
        Initialize NHITS-tiny model with CPU-friendly constraints.
        
        Args:
            random_seed: Random seed for reproducibility
            input_chunk_length: Length of input sequences
            output_chunk_length: Length of output sequences  
            num_stacks: Number of NHITS stacks (reduced for CPU)
            num_blocks: Number of blocks per stack (minimal)
            num_layers: Number of layers per block (minimal)
            layer_widths: Width of hidden layers (small)
            pooling_kernel_sizes: Pooling sizes for each stack
            n_epochs: Training epochs (reduced for speed)
            batch_size: Batch size (small for memory)
        """
        super().__init__(random_seed)
        
        if not DARTS_AVAILABLE:
            raise ImportError("Darts library required for NHITS. Install with: pip install darts")
        
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.layer_widths = layer_widths
        self.pooling_kernel_sizes = pooling_kernel_sizes or [2, 2]
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        self._model = None
        self._series = None
        self._scaler_mean = None
        self._scaler_std = None
        
        # Force CPU usage
        self.device = "cpu"
        
    def fit(
        self,
        df: pd.DataFrame,
        freq: str,
        target_col: str = "value",
        exog_cols: Optional[List[str]] = None,
        seasonal_periods: Optional[int] = None
    ) -> "NHITSTinyForecaster":
        """Fit NHITS model with CPU optimization."""
        self._validate_input(df, target_col)
        
        self.freq = freq
        self.seasonal_periods = seasonal_periods or self._infer_seasonal_periods(freq)
        
        # Store data info
        self._fit_data = df.copy()
        self._target_col = target_col
        
        # Check minimum data requirements
        min_required = self.input_chunk_length + self.output_chunk_length
        if len(df) < min_required:
            logger.warning(f"Insufficient data for NHITS. Need at least {min_required} points, got {len(df)}")
            # Fallback to simple model
            return self._fit_fallback(df, freq, target_col)
        
        try:
            # Prepare data for Darts
            target_series = self._prepare_darts_series(df, target_col)
            
            # Scale data for better training
            self._scaler_mean = target_series.values().mean()
            self._scaler_std = target_series.values().std()
            if self._scaler_std == 0:
                self._scaler_std = 1.0
            
            scaled_values = (target_series.values() - self._scaler_mean) / self._scaler_std
            scaled_series = TimeSeries.from_times_and_values(
                target_series.time_index,
                scaled_values,
                freq=freq
            )
            
            # Initialize model with CPU-friendly settings
            self._model = NHiTSModel(
                input_chunk_length=self.input_chunk_length,
                output_chunk_length=self.output_chunk_length,
                num_stacks=self.num_stacks,
                num_blocks=self.num_blocks,
                num_layers=self.num_layers,
                layer_widths=self.layer_widths,
                pooling_kernel_sizes=self.pooling_kernel_sizes,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                random_state=self.random_seed,
                force_reset=True,
                save_checkpoints=False,  # Disable checkpointing for speed
                pl_trainer_kwargs={
                    "accelerator": "cpu",
                    "devices": 1,
                    "enable_progress_bar": False,
                    "enable_model_summary": False,
                    "logger": False
                }
            )
            
            # Suppress warnings during training
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Fit model
                logger.info("Training NHITS-tiny model...")
                self._model.fit(scaled_series, verbose=False)
            
            # Compute residuals for anomaly detection
            self._compute_residuals(scaled_series)
            
            self.is_fitted = True
            logger.info("NHITS-tiny model trained successfully")
            
        except Exception as e:
            logger.warning(f"NHITS training failed: {e}. Using fallback.")
            return self._fit_fallback(df, freq, target_col)
        
        return self
    
    def predict(
        self,
        h: int,
        exog_future: Optional[pd.DataFrame] = None,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ) -> ForecastResult:
        """Generate NHITS forecasts."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self._model is None:
            # Use fallback prediction
            return self._predict_fallback(h, quantiles)
        
        try:
            # Generate forecasts
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Predict with scaled model
                forecast_scaled = self._model.predict(n=h, verbose=False)
                
                # Unscale predictions
                forecast_values = (forecast_scaled.values() * self._scaler_std) + self._scaler_mean
                forecast_values = forecast_values.flatten()
            
            # Generate future timestamps
            last_timestamp = self._fit_data.index[-1]
            future_index = self._generate_future_index(last_timestamp, h)
            
            # Simple prediction intervals using residual statistics
            if hasattr(self, '_residuals') and len(self._residuals) > 0:
                residual_std = np.std(self._residuals)
                
                # Normal approximation for quantiles
                z_scores = {0.1: -1.645, 0.5: 0.0, 0.9: 1.645}
                yhat_lower = forecast_values + z_scores[0.1] * residual_std
                yhat_upper = forecast_values + z_scores[0.9] * residual_std
            else:
                # Fallback to simple percentage bands
                uncertainty = np.std(self._fit_data[self._target_col]) * 0.2
                yhat_lower = forecast_values - uncertainty
                yhat_upper = forecast_values + uncertainty
            
            return ForecastResult(
                timestamps=future_index,
                yhat=forecast_values,
                yhat_lower=yhat_lower,
                yhat_upper=yhat_upper,
                residuals=getattr(self, '_residuals', np.array([])),
                model_info=self.get_model_info()
            )
            
        except Exception as e:
            logger.warning(f"NHITS prediction failed: {e}. Using fallback.")
            return self._predict_fallback(h, quantiles)
    
    def _prepare_darts_series(self, df: pd.DataFrame, target_col: str) -> TimeSeries:
        """Convert DataFrame to Darts TimeSeries."""
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex for Darts")
        
        # Create TimeSeries
        series = TimeSeries.from_dataframe(
            df,
            time_col=None,  # Use index
            value_cols=target_col,
            freq=self.freq
        )
        
        return series
    
    def _compute_residuals(self, series: TimeSeries) -> None:
        """Compute in-sample residuals for anomaly detection."""
        try:
            # Generate in-sample predictions
            n_predictions = min(len(series) - self.input_chunk_length, 50)  # Limit for speed
            if n_predictions <= 0:
                self._residuals = np.array([])
                return
            
            # Historical forecasts for residual computation
            historical_forecasts = self._model.historical_forecasts(
                series,
                start=self.input_chunk_length,
                forecast_horizon=1,
                stride=max(1, n_predictions // 20),  # Subsample for speed
                retrain=False,
                verbose=False
            )
            
            if historical_forecasts is not None and len(historical_forecasts) > 0:
                # Unscale predictions and actuals
                pred_values = (historical_forecasts.values() * self._scaler_std) + self._scaler_mean
                
                # Get corresponding actual values
                pred_times = historical_forecasts.time_index
                actual_values = []
                for t in pred_times:
                    if t in series.time_index:
                        actual_val = series[t].values()[0]
                        actual_val = (actual_val * self._scaler_std) + self._scaler_mean
                        actual_values.append(actual_val)
                
                if len(actual_values) == len(pred_values):
                    self._residuals = np.array(actual_values) - pred_values.flatten()
                else:
                    self._residuals = np.array([])
            else:
                self._residuals = np.array([])
                
        except Exception as e:
            logger.warning(f"Residual computation failed: {e}")
            self._residuals = np.array([])
    
    def _fit_fallback(self, df: pd.DataFrame, freq: str, target_col: str) -> "NHITSTinyForecaster":
        """Fallback to simple seasonal naive if NHITS fails."""
        logger.info("Using seasonal naive fallback")
        
        # Store fallback data
        self._fallback_data = df[target_col].values
        self._fallback_seasonal_periods = self._infer_seasonal_periods(freq)
        
        # Simple residuals
        if len(self._fallback_data) >= self._fallback_seasonal_periods:
            seasonal_forecast = np.tile(
                self._fallback_data[-self._fallback_seasonal_periods:],
                len(self._fallback_data) // self._fallback_seasonal_periods + 1
            )[:len(self._fallback_data)]
            self._residuals = self._fallback_data - seasonal_forecast
        else:
            mean_val = np.mean(self._fallback_data)
            self._residuals = self._fallback_data - mean_val
        
        self.is_fitted = True
        return self
    
    def _predict_fallback(self, h: int, quantiles: List[float]) -> ForecastResult:
        """Fallback prediction using seasonal naive."""
        if not hasattr(self, '_fallback_data'):
            raise ValueError("No fallback data available")
        
        # Seasonal naive forecast
        if len(self._fallback_data) >= self._fallback_seasonal_periods:
            pattern = self._fallback_data[-self._fallback_seasonal_periods:]
            n_full_cycles = h // self._fallback_seasonal_periods
            remainder = h % self._fallback_seasonal_periods
            
            forecast = np.tile(pattern, n_full_cycles)
            if remainder > 0:
                forecast = np.concatenate([forecast, pattern[:remainder]])
        else:
            # Simple mean forecast
            forecast = np.full(h, np.mean(self._fallback_data))
        
        # Simple prediction intervals
        residual_std = np.std(self._residuals) if len(self._residuals) > 0 else np.std(self._fallback_data) * 0.1
        yhat_lower = forecast - 1.645 * residual_std
        yhat_upper = forecast + 1.645 * residual_std
        
        # Generate timestamps
        last_timestamp = self._fit_data.index[-1]
        future_index = self._generate_future_index(last_timestamp, h)
        
        return ForecastResult(
            timestamps=future_index,
            yhat=forecast,
            yhat_lower=yhat_lower,
            yhat_upper=yhat_upper,
            residuals=self._residuals,
            model_info={"model_type": "NHITSTiny_Fallback", "note": "Using seasonal naive fallback"}
        )
    
    def get_model_info(self) -> Dict:
        """Get NHITS model information."""
        base_info = super().get_model_info()
        
        if self._model is not None:
            base_info.update({
                "architecture": "NHITS-tiny",
                "input_chunk_length": self.input_chunk_length,
                "output_chunk_length": self.output_chunk_length,
                "num_stacks": self.num_stacks,
                "num_blocks": self.num_blocks,
                "num_layers": self.num_layers,
                "layer_widths": self.layer_widths,
                "n_epochs": self.n_epochs,
                "device": self.device,
                "parameters": self._count_parameters()
            })
        else:
            base_info.update({
                "architecture": "fallback",
                "note": "NHITS training failed, using seasonal naive"
            })
        
        return base_info
    
    def _count_parameters(self) -> int:
        """Count model parameters."""
        try:
            if self._model and hasattr(self._model, 'model'):
                return sum(p.numel() for p in self._model.model.parameters())
            return 0
        except:
            return 0


# Alias for backward compatibility
NHiTSTinyForecaster = NHITSTinyForecaster
