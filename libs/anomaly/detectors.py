"""
Anomaly detection algorithms for SmartSense.

Implements robust statistical methods for detecting anomalies in energy load data:
- RobustAnomalyDetector: MAD-based z-score with seasonal awareness
- EWMAAnomalyDetector: Exponentially weighted moving average for real-time detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class AnomalyResult:
    """Container for anomaly detection results."""
    indices: np.ndarray  # Indices of detected anomalies
    scores: np.ndarray   # Anomaly scores
    labels: List[str]    # Anomaly types ("spike", "drop", "shift")
    messages: List[str]  # Human-readable descriptions
    threshold: float     # Detection threshold used
    metadata: Optional[Dict] = None


class BaseAnomalyDetector(ABC):
    """Base class for anomaly detectors."""
    
    @abstractmethod
    def detect(self, residuals: np.ndarray, **kwargs) -> AnomalyResult:
        """Detect anomalies in residual series."""
        pass


class RobustAnomalyDetector(BaseAnomalyDetector):
    """
    Robust anomaly detector using Median Absolute Deviation (MAD).
    
    More robust to outliers than standard deviation-based methods.
    Includes seasonal awareness and configurable thresholds.
    """
    
    def __init__(
        self,
        threshold: float = 3.0,
        seasonal_periods: Optional[int] = None,
        min_periods: int = 10,
        cooldown_periods: int = 3
    ):
        self.threshold = threshold
        self.seasonal_periods = seasonal_periods
        self.min_periods = min_periods
        self.cooldown_periods = cooldown_periods
    
    def detect(
        self, 
        residuals: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
        **kwargs
    ) -> AnomalyResult:
        """
        Detect anomalies using robust MAD-based z-score.
        
        Args:
            residuals: Residual series (y_true - y_pred)
            timestamps: Optional timestamps for context
            
        Returns:
            AnomalyResult with detected anomalies
        """
        if len(residuals) < self.min_periods:
            return AnomalyResult(
                indices=np.array([]),
                scores=np.array([]),
                labels=[],
                messages=[],
                threshold=self.threshold
            )
        
        # Compute robust z-scores using MAD
        robust_scores = self._compute_robust_zscore(residuals)
        
        # Apply seasonal adjustment if specified
        if self.seasonal_periods and len(residuals) >= 2 * self.seasonal_periods:
            robust_scores = self._apply_seasonal_adjustment(robust_scores, residuals)
        
        # Detect anomalies above threshold
        anomaly_mask = np.abs(robust_scores) > self.threshold
        
        # Apply cooldown to avoid duplicate detections
        anomaly_mask = self._apply_cooldown(anomaly_mask)
        
        # Extract anomaly information
        anomaly_indices = np.where(anomaly_mask)[0]
        anomaly_scores = robust_scores[anomaly_mask]
        
        # Classify anomaly types and generate messages
        labels, messages = self._classify_anomalies(
            anomaly_indices, anomaly_scores, residuals, timestamps
        )
        
        return AnomalyResult(
            indices=anomaly_indices,
            scores=anomaly_scores,
            labels=labels,
            messages=messages,
            threshold=self.threshold,
            metadata={
                "method": "robust_mad",
                "seasonal_periods": self.seasonal_periods,
                "total_points": len(residuals),
                "anomaly_rate": len(anomaly_indices) / len(residuals)
            }
        )
    
    def _compute_robust_zscore(self, residuals: np.ndarray) -> np.ndarray:
        """Compute robust z-score using Median Absolute Deviation."""
        median = np.median(residuals)
        mad = np.median(np.abs(residuals - median))
        
        # Avoid division by zero
        if mad == 0:
            mad = np.std(residuals) * 0.6745  # Fallback to std * normal MAD factor
        
        # Robust z-score: (x - median) / (1.4826 * MAD)
        # 1.4826 is the consistency factor for normal distribution
        robust_zscore = (residuals - median) / (1.4826 * mad)
        return robust_zscore
    
    def _apply_seasonal_adjustment(self, scores: np.ndarray, residuals: np.ndarray) -> np.ndarray:
        """Apply seasonal adjustment to anomaly scores."""
        adjusted_scores = scores.copy()
        
        # Compute seasonal statistics
        n_seasons = len(residuals) // self.seasonal_periods
        if n_seasons >= 2:
            for i in range(self.seasonal_periods):
                # Get values for this seasonal position
                seasonal_indices = np.arange(i, len(residuals), self.seasonal_periods)
                seasonal_values = residuals[seasonal_indices]
                
                if len(seasonal_values) >= 3:
                    # Compute seasonal MAD
                    seasonal_median = np.median(seasonal_values)
                    seasonal_mad = np.median(np.abs(seasonal_values - seasonal_median))
                    
                    if seasonal_mad > 0:
                        # Adjust scores for this seasonal position
                        for idx in seasonal_indices:
                            if idx < len(adjusted_scores):
                                adjusted_scores[idx] = (residuals[idx] - seasonal_median) / (1.4826 * seasonal_mad)
        
        return adjusted_scores
    
    def _apply_cooldown(self, anomaly_mask: np.ndarray) -> np.ndarray:
        """Apply cooldown period to avoid duplicate detections."""
        if self.cooldown_periods <= 0:
            return anomaly_mask
        
        adjusted_mask = anomaly_mask.copy()
        last_anomaly = -self.cooldown_periods - 1
        
        for i in range(len(anomaly_mask)):
            if anomaly_mask[i]:
                if i - last_anomaly <= self.cooldown_periods:
                    adjusted_mask[i] = False
                else:
                    last_anomaly = i
        
        return adjusted_mask
    
    def _classify_anomalies(
        self, 
        indices: np.ndarray, 
        scores: np.ndarray, 
        residuals: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex]
    ) -> Tuple[List[str], List[str]]:
        """Classify anomalies and generate human-readable messages."""
        labels = []
        messages = []
        
        for i, (idx, score) in enumerate(zip(indices, scores)):
            # Classify based on score sign and magnitude
            if score > 0:
                label = "spike" if score > 2 * self.threshold else "high"
            else:
                label = "drop" if score < -2 * self.threshold else "low"
            
            # Generate message
            if timestamps is not None and idx < len(timestamps):
                time_str = timestamps[idx].strftime("%Y-%m-%d %H:%M")
                message = f"{label.capitalize()}: {score:+.1f}σ at {time_str}"
            else:
                message = f"{label.capitalize()}: {score:+.1f}σ at index {idx}"
            
            labels.append(label)
            messages.append(message)
        
        return labels, messages


class EWMAAnomalyDetector(BaseAnomalyDetector):
    """
    Exponentially Weighted Moving Average (EWMA) anomaly detector.
    
    Suitable for real-time anomaly detection with adaptive thresholds.
    Responds quickly to recent changes while maintaining stability.
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        threshold: float = 3.0,
        min_periods: int = 10,
        adapt_threshold: bool = True
    ):
        self.alpha = alpha
        self.threshold = threshold
        self.min_periods = min_periods
        self.adapt_threshold = adapt_threshold
        self._ewma_mean = None
        self._ewma_var = None
    
    def detect(
        self, 
        residuals: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
        **kwargs
    ) -> AnomalyResult:
        """
        Detect anomalies using EWMA control chart approach.
        
        Args:
            residuals: Residual series (y_true - y_pred)
            timestamps: Optional timestamps for context
            
        Returns:
            AnomalyResult with detected anomalies
        """
        if len(residuals) < self.min_periods:
            return AnomalyResult(
                indices=np.array([]),
                scores=np.array([]),
                labels=[],
                messages=[],
                threshold=self.threshold
            )
        
        # Initialize EWMA statistics
        ewma_mean = np.zeros(len(residuals))
        ewma_var = np.zeros(len(residuals))
        
        # Initialize with first value
        ewma_mean[0] = residuals[0]
        ewma_var[0] = 0
        
        # Compute EWMA statistics
        for i in range(1, len(residuals)):
            ewma_mean[i] = self.alpha * residuals[i] + (1 - self.alpha) * ewma_mean[i-1]
            
            # EWMA variance estimation
            squared_residual = (residuals[i] - ewma_mean[i-1]) ** 2
            ewma_var[i] = self.alpha * squared_residual + (1 - self.alpha) * ewma_var[i-1]
        
        # Compute EWMA z-scores
        ewma_std = np.sqrt(ewma_var)
        ewma_std[ewma_std == 0] = np.std(residuals[:self.min_periods])  # Avoid division by zero
        
        ewma_scores = (residuals - ewma_mean) / ewma_std
        
        # Adaptive threshold based on recent volatility
        if self.adapt_threshold:
            recent_window = min(50, len(residuals) // 4)
            recent_volatility = np.std(ewma_scores[-recent_window:]) if recent_window > 0 else 1.0
            adaptive_threshold = self.threshold * max(0.5, min(2.0, recent_volatility))
        else:
            adaptive_threshold = self.threshold
        
        # Detect anomalies
        anomaly_mask = np.abs(ewma_scores) > adaptive_threshold
        
        # Only consider anomalies after minimum periods
        anomaly_mask[:self.min_periods] = False
        
        # Extract anomaly information
        anomaly_indices = np.where(anomaly_mask)[0]
        anomaly_scores = ewma_scores[anomaly_mask]
        
        # Classify anomalies and generate messages
        labels, messages = self._classify_anomalies(
            anomaly_indices, anomaly_scores, residuals, timestamps, adaptive_threshold
        )
        
        # Store state for potential online updates
        self._ewma_mean = ewma_mean[-1] if len(ewma_mean) > 0 else 0
        self._ewma_var = ewma_var[-1] if len(ewma_var) > 0 else 0
        
        return AnomalyResult(
            indices=anomaly_indices,
            scores=anomaly_scores,
            labels=labels,
            messages=messages,
            threshold=adaptive_threshold,
            metadata={
                "method": "ewma",
                "alpha": self.alpha,
                "adaptive_threshold": adaptive_threshold,
                "base_threshold": self.threshold,
                "total_points": len(residuals),
                "anomaly_rate": len(anomaly_indices) / len(residuals)
            }
        )
    
    def _classify_anomalies(
        self, 
        indices: np.ndarray, 
        scores: np.ndarray, 
        residuals: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex],
        threshold: float
    ) -> Tuple[List[str], List[str]]:
        """Classify EWMA anomalies and generate messages."""
        labels = []
        messages = []
        
        for i, (idx, score) in enumerate(zip(indices, scores)):
            # Classify based on score and magnitude
            if score > 0:
                if score > 2 * threshold:
                    label = "severe_spike"
                elif score > 1.5 * threshold:
                    label = "spike"
                else:
                    label = "high"
            else:
                if score < -2 * threshold:
                    label = "severe_drop"
                elif score < -1.5 * threshold:
                    label = "drop"
                else:
                    label = "low"
            
            # Generate message with EWMA context
            if timestamps is not None and idx < len(timestamps):
                time_str = timestamps[idx].strftime("%Y-%m-%d %H:%M")
                message = f"EWMA {label}: {score:+.1f}σ at {time_str}"
            else:
                message = f"EWMA {label}: {score:+.1f}σ at index {idx}"
            
            labels.append(label)
            messages.append(message)
        
        return labels, messages
    
    def update_online(self, new_residual: float) -> Tuple[float, bool]:
        """
        Update EWMA statistics with new residual (for real-time detection).
        
        Args:
            new_residual: New residual value
            
        Returns:
            Tuple of (anomaly_score, is_anomaly)
        """
        if self._ewma_mean is None or self._ewma_var is None:
            # Initialize if not fitted
            self._ewma_mean = new_residual
            self._ewma_var = 0
            return 0.0, False
        
        # Update EWMA mean
        prev_mean = self._ewma_mean
        self._ewma_mean = self.alpha * new_residual + (1 - self.alpha) * self._ewma_mean
        
        # Update EWMA variance
        squared_residual = (new_residual - prev_mean) ** 2
        self._ewma_var = self.alpha * squared_residual + (1 - self.alpha) * self._ewma_var
        
        # Compute anomaly score
        ewma_std = np.sqrt(self._ewma_var) if self._ewma_var > 0 else 1.0
        anomaly_score = (new_residual - prev_mean) / ewma_std
        
        # Check if anomaly
        is_anomaly = abs(anomaly_score) > self.threshold
        
        return anomaly_score, is_anomaly
