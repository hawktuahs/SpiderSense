"""
Anomaly detection pipeline for SmartSense.

Combines multiple detection methods and provides a unified interface
for anomaly detection in energy load forecasting residuals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from .detectors import RobustAnomalyDetector, EWMAAnomalyDetector, AnomalyResult
from .changepoint import ChangePointDetector, ChangePointResult, SimpleChangePointDetector


@dataclass
class PipelineResult:
    """Combined results from anomaly detection pipeline."""
    anomalies: AnomalyResult
    changepoints: Optional[ChangePointResult] = None
    summary: Optional[Dict] = None


class AnomalyPipeline:
    """
    Unified anomaly detection pipeline for SmartSense.
    
    Combines residual-based anomaly detection with changepoint analysis
    to provide comprehensive anomaly insights for energy load data.
    """
    
    def __init__(
        self,
        primary_detector: str = "robust",
        enable_changepoints: bool = True,
        seasonal_periods: Optional[int] = None,
        **detector_kwargs
    ):
        """
        Initialize anomaly detection pipeline.
        
        Args:
            primary_detector: Primary detector ("robust", "ewma")
            enable_changepoints: Whether to run changepoint detection
            seasonal_periods: Seasonal periods for seasonal adjustment
            **detector_kwargs: Additional arguments for detectors
        """
        self.primary_detector = primary_detector
        self.enable_changepoints = enable_changepoints
        self.seasonal_periods = seasonal_periods
        
        # Initialize primary detector
        if primary_detector == "robust":
            self.detector = RobustAnomalyDetector(
                seasonal_periods=seasonal_periods,
                **detector_kwargs
            )
        elif primary_detector == "ewma":
            self.detector = EWMAAnomalyDetector(**detector_kwargs)
        else:
            raise ValueError(f"Unknown detector: {primary_detector}")
        
        # Initialize changepoint detector if enabled
        self.changepoint_detector = None
        if enable_changepoints:
            try:
                self.changepoint_detector = ChangePointDetector()
            except ImportError:
                # Fallback to simple detector if ruptures not available
                self.changepoint_detector = SimpleChangePointDetector()
    
    def detect(
        self,
        residuals: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
        original_data: Optional[np.ndarray] = None,
        **kwargs
    ) -> PipelineResult:
        """
        Run complete anomaly detection pipeline.
        
        Args:
            residuals: Forecast residuals (y_true - y_pred)
            timestamps: Optional timestamps for context
            original_data: Original time series (for changepoint detection)
            **kwargs: Additional arguments for detectors
            
        Returns:
            PipelineResult with anomalies and changepoints
        """
        # Primary anomaly detection
        anomaly_result = self.detector.detect(
            residuals=residuals,
            timestamps=timestamps,
            **kwargs
        )
        
        # Changepoint detection
        changepoint_result = None
        if self.enable_changepoints and self.changepoint_detector:
            # Use original data if available, otherwise use residuals
            cp_data = original_data if original_data is not None else residuals
            changepoint_result = self.changepoint_detector.detect(
                data=cp_data,
                timestamps=timestamps
            )
        
        # Generate summary
        summary = self._generate_summary(anomaly_result, changepoint_result, residuals)
        
        return PipelineResult(
            anomalies=anomaly_result,
            changepoints=changepoint_result,
            summary=summary
        )
    
    def detect_realtime(
        self,
        new_residual: float,
        timestamp: Optional[pd.Timestamp] = None
    ) -> Dict:
        """
        Real-time anomaly detection for streaming data.
        
        Args:
            new_residual: New residual value
            timestamp: Optional timestamp
            
        Returns:
            Dictionary with anomaly information
        """
        if not isinstance(self.detector, EWMAAnomalyDetector):
            raise ValueError("Real-time detection only supported with EWMA detector")
        
        score, is_anomaly = self.detector.update_online(new_residual)
        
        result = {
            "is_anomaly": is_anomaly,
            "score": score,
            "residual": new_residual,
            "timestamp": timestamp.isoformat() if timestamp else None
        }
        
        if is_anomaly:
            # Classify anomaly type
            if score > 0:
                label = "spike" if score > 2 * self.detector.threshold else "high"
            else:
                label = "drop" if score < -2 * self.detector.threshold else "low"
            
            result.update({
                "label": label,
                "message": f"Real-time {label}: {score:+.1f}σ"
            })
        
        return result
    
    def _generate_summary(
        self,
        anomaly_result: AnomalyResult,
        changepoint_result: Optional[ChangePointResult],
        residuals: np.ndarray
    ) -> Dict:
        """Generate summary statistics for the pipeline results."""
        summary = {
            "total_points": len(residuals),
            "anomaly_count": len(anomaly_result.indices),
            "anomaly_rate": len(anomaly_result.indices) / len(residuals) if len(residuals) > 0 else 0,
            "primary_detector": self.primary_detector,
            "residual_stats": {
                "mean": np.mean(residuals),
                "std": np.std(residuals),
                "min": np.min(residuals),
                "max": np.max(residuals)
            }
        }
        
        # Anomaly type distribution
        if anomaly_result.labels:
            label_counts = {}
            for label in anomaly_result.labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            summary["anomaly_types"] = label_counts
        
        # Changepoint information
        if changepoint_result:
            summary["changepoints"] = {
                "count": len(changepoint_result.changepoints),
                "method": changepoint_result.method,
                "segments": len(changepoint_result.segments)
            }
            
            if changepoint_result.segments:
                segment_lengths = [end - start for start, end in changepoint_result.segments]
                summary["changepoints"]["avg_segment_length"] = np.mean(segment_lengths)
                summary["changepoints"]["min_segment_length"] = np.min(segment_lengths)
                summary["changepoints"]["max_segment_length"] = np.max(segment_lengths)
        
        # Severity assessment
        if anomaly_result.scores.size > 0:
            max_score = np.max(np.abs(anomaly_result.scores))
            if max_score > 5:
                severity = "critical"
            elif max_score > 3:
                severity = "high"
            elif max_score > 2:
                severity = "medium"
            else:
                severity = "low"
            summary["severity"] = severity
            summary["max_anomaly_score"] = max_score
        else:
            summary["severity"] = "none"
            summary["max_anomaly_score"] = 0
        
        return summary
    
    def get_anomaly_timeline(
        self,
        result: PipelineResult,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> pd.DataFrame:
        """
        Generate anomaly timeline for visualization.
        
        Args:
            result: PipelineResult from detection
            timestamps: Optional timestamps
            
        Returns:
            DataFrame with anomaly timeline
        """
        if timestamps is None:
            timestamps = pd.date_range(start='2024-01-01', periods=result.summary['total_points'], freq='H')
        
        # Create base timeline
        timeline = pd.DataFrame({
            'timestamp': timestamps,
            'is_anomaly': False,
            'anomaly_score': 0.0,
            'anomaly_type': '',
            'is_changepoint': False
        })
        
        # Mark anomalies
        if len(result.anomalies.indices) > 0:
            for i, (idx, score, label) in enumerate(zip(
                result.anomalies.indices,
                result.anomalies.scores,
                result.anomalies.labels
            )):
                if idx < len(timeline):
                    timeline.loc[idx, 'is_anomaly'] = True
                    timeline.loc[idx, 'anomaly_score'] = score
                    timeline.loc[idx, 'anomaly_type'] = label
        
        # Mark changepoints
        if result.changepoints and result.changepoints.changepoints:
            for cp in result.changepoints.changepoints:
                if cp < len(timeline):
                    timeline.loc[cp, 'is_changepoint'] = True
        
        return timeline
    
    def export_results(
        self,
        result: PipelineResult,
        filepath: str,
        format: str = "csv"
    ) -> None:
        """
        Export anomaly detection results to file.
        
        Args:
            result: PipelineResult to export
            filepath: Output file path
            format: Export format ("csv", "json")
        """
        if format == "csv":
            # Create detailed results DataFrame
            export_data = []
            
            for i, (idx, score, label, message) in enumerate(zip(
                result.anomalies.indices,
                result.anomalies.scores,
                result.anomalies.labels,
                result.anomalies.messages
            )):
                export_data.append({
                    'index': idx,
                    'score': score,
                    'type': label,
                    'message': message,
                    'detector': self.primary_detector
                })
            
            df = pd.DataFrame(export_data)
            df.to_csv(filepath, index=False)
            
        elif format == "json":
            import json
            
            export_dict = {
                'anomalies': {
                    'indices': result.anomalies.indices.tolist(),
                    'scores': result.anomalies.scores.tolist(),
                    'labels': result.anomalies.labels,
                    'messages': result.anomalies.messages,
                    'metadata': result.anomalies.metadata
                },
                'summary': result.summary
            }
            
            if result.changepoints:
                export_dict['changepoints'] = {
                    'indices': result.changepoints.changepoints,
                    'scores': result.changepoints.scores,
                    'segments': result.changepoints.segments,
                    'metadata': result.changepoints.metadata
                }
            
            with open(filepath, 'w') as f:
                json.dump(export_dict, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported format: {format}")


def create_anomaly_pipeline(
    detector_type: str = "robust",
    seasonal_periods: Optional[int] = None,
    **kwargs
) -> AnomalyPipeline:
    """
    Factory function to create anomaly detection pipeline.
    
    Args:
        detector_type: Type of detector ("robust", "ewma")
        seasonal_periods: Seasonal periods for adjustment
        **kwargs: Additional detector arguments
        
    Returns:
        Configured AnomalyPipeline
    """
    return AnomalyPipeline(
        primary_detector=detector_type,
        seasonal_periods=seasonal_periods,
        **kwargs
    )
