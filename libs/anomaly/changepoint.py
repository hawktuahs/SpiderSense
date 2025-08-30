"""
Changepoint detection for SmartSense anomaly pipeline.

Detects regime shifts and structural breaks in energy load patterns
using the ruptures library with CPU-optimized configurations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False


@dataclass
class ChangePointResult:
    """Container for changepoint detection results."""
    changepoints: List[int]  # Indices of detected changepoints
    scores: List[float]      # Change scores at each point
    segments: List[Tuple[int, int]]  # Segment boundaries
    method: str              # Detection method used
    metadata: Optional[Dict] = None


class ChangePointDetector:
    """
    Changepoint detector for identifying regime shifts in energy data.
    
    Uses ruptures library with PELT (Pruned Exact Linear Time) algorithm
    for efficient detection of multiple changepoints.
    """
    
    def __init__(
        self,
        method: str = "pelt",
        model: str = "rbf",
        min_size: int = 10,
        jump: int = 1,
        penalty: Optional[float] = None
    ):
        """
        Initialize changepoint detector.
        
        Args:
            method: Detection method ("pelt", "binseg", "window")
            model: Cost function ("l2", "l1", "rbf", "normal")
            min_size: Minimum segment size
            jump: Subsample factor for speed
            penalty: Penalty for changepoint (auto if None)
        """
        if not RUPTURES_AVAILABLE:
            raise ImportError("ruptures library required for changepoint detection. Install with: pip install ruptures")
        
        self.method = method
        self.model = model
        self.min_size = min_size
        self.jump = jump
        self.penalty = penalty
        
    def detect(
        self, 
        data: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
        max_changepoints: int = 10
    ) -> ChangePointResult:
        """
        Detect changepoints in time series data.
        
        Args:
            data: Time series data (residuals or original values)
            timestamps: Optional timestamps for context
            max_changepoints: Maximum number of changepoints to detect
            
        Returns:
            ChangePointResult with detected changepoints
        """
        if len(data) < 2 * self.min_size:
            return ChangePointResult(
                changepoints=[],
                scores=[],
                segments=[(0, len(data))],
                method=self.method,
                metadata={"reason": "insufficient_data", "min_required": 2 * self.min_size}
            )
        
        try:
            # Initialize detector based on method
            if self.method == "pelt":
                detector = rpt.Pelt(model=self.model, min_size=self.min_size, jump=self.jump)
            elif self.method == "binseg":
                detector = rpt.Binseg(model=self.model, min_size=self.min_size, jump=self.jump)
            elif self.method == "window":
                window_size = min(50, len(data) // 4)
                detector = rpt.Window(width=window_size, model=self.model, min_size=self.min_size, jump=self.jump)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            # Fit detector
            detector.fit(data.reshape(-1, 1))
            
            # Determine penalty if not provided
            if self.penalty is None:
                # Auto-select penalty based on data characteristics
                if self.method == "pelt":
                    # For PELT, use log(n) * variance as default penalty
                    penalty = np.log(len(data)) * np.var(data)
                else:
                    # For other methods, use number of changepoints
                    penalty = max_changepoints
            else:
                penalty = self.penalty
            
            # Detect changepoints
            if self.method in ["pelt", "window"]:
                changepoints = detector.predict(pen=penalty)
            else:  # binseg
                changepoints = detector.predict(n_bkps=min(max_changepoints, penalty))
            
            # Remove the last point (end of series) if present
            if changepoints and changepoints[-1] == len(data):
                changepoints = changepoints[:-1]
            
            # Compute change scores
            scores = self._compute_change_scores(data, changepoints, detector)
            
            # Generate segments
            segments = self._generate_segments(changepoints, len(data))
            
            # Generate metadata
            metadata = {
                "method": self.method,
                "model": self.model,
                "penalty": penalty,
                "total_points": len(data),
                "n_changepoints": len(changepoints),
                "avg_segment_length": np.mean([end - start for start, end in segments]) if segments else 0
            }
            
            # Add timestamp information if available
            if timestamps is not None and changepoints:
                metadata["changepoint_times"] = [
                    timestamps[cp].isoformat() for cp in changepoints if cp < len(timestamps)
                ]
            
            return ChangePointResult(
                changepoints=changepoints,
                scores=scores,
                segments=segments,
                method=self.method,
                metadata=metadata
            )
            
        except Exception as e:
            # Fallback: return no changepoints if detection fails
            return ChangePointResult(
                changepoints=[],
                scores=[],
                segments=[(0, len(data))],
                method=self.method,
                metadata={"error": str(e), "fallback": True}
            )
    
    def _compute_change_scores(
        self, 
        data: np.ndarray, 
        changepoints: List[int], 
        detector
    ) -> List[float]:
        """Compute change magnitude scores at detected changepoints."""
        scores = []
        
        for cp in changepoints:
            try:
                # Compute score based on difference in segment statistics
                if cp > self.min_size and cp < len(data) - self.min_size:
                    # Before and after segments
                    before = data[max(0, cp - self.min_size):cp]
                    after = data[cp:min(len(data), cp + self.min_size)]
                    
                    # Score based on difference in means and variances
                    mean_diff = abs(np.mean(after) - np.mean(before))
                    var_diff = abs(np.var(after) - np.var(before))
                    
                    # Normalize by overall data statistics
                    data_std = np.std(data)
                    score = (mean_diff + var_diff) / (data_std + 1e-8)
                    scores.append(score)
                else:
                    scores.append(1.0)  # Default score for edge cases
                    
            except Exception:
                scores.append(1.0)  # Default score if computation fails
        
        return scores
    
    def _generate_segments(self, changepoints: List[int], data_length: int) -> List[Tuple[int, int]]:
        """Generate segment boundaries from changepoints."""
        if not changepoints:
            return [(0, data_length)]
        
        segments = []
        start = 0
        
        for cp in changepoints:
            segments.append((start, cp))
            start = cp
        
        # Add final segment
        segments.append((start, data_length))
        
        return segments
    
    def analyze_segments(
        self, 
        data: np.ndarray, 
        result: ChangePointResult
    ) -> Dict:
        """
        Analyze characteristics of detected segments.
        
        Args:
            data: Original time series data
            result: ChangePointResult from detection
            
        Returns:
            Dictionary with segment analysis
        """
        analysis = {
            "segments": [],
            "summary": {
                "n_segments": len(result.segments),
                "avg_length": 0,
                "stability_score": 0
            }
        }
        
        segment_stats = []
        
        for i, (start, end) in enumerate(result.segments):
            segment_data = data[start:end]
            
            if len(segment_data) > 0:
                stats = {
                    "segment_id": i,
                    "start": start,
                    "end": end,
                    "length": end - start,
                    "mean": np.mean(segment_data),
                    "std": np.std(segment_data),
                    "min": np.min(segment_data),
                    "max": np.max(segment_data),
                    "trend": np.polyfit(range(len(segment_data)), segment_data, 1)[0] if len(segment_data) > 1 else 0
                }
                
                analysis["segments"].append(stats)
                segment_stats.append(stats)
        
        # Compute summary statistics
        if segment_stats:
            analysis["summary"]["avg_length"] = np.mean([s["length"] for s in segment_stats])
            
            # Stability score: inverse of coefficient of variation across segment means
            segment_means = [s["mean"] for s in segment_stats]
            if len(segment_means) > 1:
                cv = np.std(segment_means) / (np.mean(segment_means) + 1e-8)
                analysis["summary"]["stability_score"] = 1 / (1 + cv)
            else:
                analysis["summary"]["stability_score"] = 1.0
        
        return analysis


class SimpleChangePointDetector:
    """
    Simple changepoint detector for cases where ruptures is not available.
    
    Uses basic statistical methods to detect significant changes in mean/variance.
    """
    
    def __init__(self, window_size: int = 20, threshold: float = 2.0):
        self.window_size = window_size
        self.threshold = threshold
    
    def detect(
        self, 
        data: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
        max_changepoints: int = 10
    ) -> ChangePointResult:
        """Simple changepoint detection using moving statistics."""
        if len(data) < 2 * self.window_size:
            return ChangePointResult(
                changepoints=[],
                scores=[],
                segments=[(0, len(data))],
                method="simple",
                metadata={"reason": "insufficient_data"}
            )
        
        changepoints = []
        scores = []
        
        # Sliding window approach
        for i in range(self.window_size, len(data) - self.window_size):
            # Before and after windows
            before = data[i - self.window_size:i]
            after = data[i:i + self.window_size]
            
            # Test for change in mean
            mean_before = np.mean(before)
            mean_after = np.mean(after)
            std_pooled = np.sqrt((np.var(before) + np.var(after)) / 2)
            
            if std_pooled > 0:
                t_stat = abs(mean_after - mean_before) / (std_pooled * np.sqrt(2 / self.window_size))
                
                if t_stat > self.threshold:
                    changepoints.append(i)
                    scores.append(t_stat)
        
        # Remove nearby changepoints (keep strongest)
        if changepoints:
            filtered_cps = []
            filtered_scores = []
            
            i = 0
            while i < len(changepoints):
                best_idx = i
                best_score = scores[i]
                
                # Look ahead for nearby changepoints
                j = i + 1
                while j < len(changepoints) and changepoints[j] - changepoints[i] < self.window_size:
                    if scores[j] > best_score:
                        best_idx = j
                        best_score = scores[j]
                    j += 1
                
                filtered_cps.append(changepoints[best_idx])
                filtered_scores.append(scores[best_idx])
                i = j
            
            changepoints = filtered_cps[:max_changepoints]
            scores = filtered_scores[:max_changepoints]
        
        segments = self._generate_segments(changepoints, len(data))
        
        return ChangePointResult(
            changepoints=changepoints,
            scores=scores,
            segments=segments,
            method="simple",
            metadata={
                "window_size": self.window_size,
                "threshold": self.threshold,
                "total_points": len(data)
            }
        )
    
    def _generate_segments(self, changepoints: List[int], data_length: int) -> List[Tuple[int, int]]:
        """Generate segment boundaries from changepoints."""
        if not changepoints:
            return [(0, data_length)]
        
        segments = []
        start = 0
        
        for cp in changepoints:
            segments.append((start, cp))
            start = cp
        
        segments.append((start, data_length))
        return segments
