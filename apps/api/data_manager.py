"""
Data management for SmartSense API.

Handles CSV ingestion, data validation, schema mapping,
and series storage with support for multi-meter data.
"""

import pandas as pd
import numpy as np
import uuid
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataManager:
    """Manages data ingestion, validation, and storage for SmartSense."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # In-memory storage for active series
        self.series_store: Dict[str, Dict] = {}
        
        # Schema mapping patterns
        self.column_patterns = {
            "timestamp": ["timestamp", "time", "date", "datetime", "ts"],
            "value": ["value", "load", "consumption", "demand", "energy", "power", "kwh"],
            "meter_id": ["meter_id", "meter", "device_id", "sensor_id", "id"],
            "temperature": ["temp", "temperature", "t"],
            "humidity": ["humidity", "humid", "rh"],
            "holiday": ["holiday", "holiday_flag", "is_holiday"]
        }
    
    async def ingest_dataframe(
        self,
        df: pd.DataFrame,
        filename: str,
        timezone: str = "Asia/Kolkata",
        freq_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ingest and process DataFrame for forecasting.
        
        Args:
            df: Input DataFrame
            filename: Original filename
            timezone: Timezone for timestamp parsing
            freq_hint: Frequency hint from user
            
        Returns:
            Dictionary with series information
        """
        # Generate unique series ID
        series_id = str(uuid.uuid4())[:8]
        
        # Map columns to standard schema
        schema_map = self._map_columns(df.columns.tolist())
        
        # Validate required columns
        if "timestamp" not in schema_map or "value" not in schema_map:
            raise ValueError("Required columns 'timestamp' and 'value' not found or mappable")
        
        # Rename columns according to schema
        df_processed = df.rename(columns={v: k for k, v in schema_map.items()})
        
        # Parse timestamps
        df_processed["timestamp"] = pd.to_datetime(df_processed["timestamp"])
        df_processed = df_processed.set_index("timestamp")
        
        # Localize timezone
        if df_processed.index.tz is None:
            df_processed.index = df_processed.index.tz_localize(timezone)
        else:
            df_processed.index = df_processed.index.tz_convert(timezone)
        
        # Sort by timestamp
        df_processed = df_processed.sort_index()
        
        # Infer frequency
        frequency = self._infer_frequency(df_processed.index, freq_hint)
        
        # Handle missing values
        df_processed = self._handle_missing_values(df_processed)
        
        # Detect multi-meter data
        has_multiple_meters = "meter_id" in df_processed.columns
        meter_ids = []
        
        if has_multiple_meters:
            meter_ids = df_processed["meter_id"].unique().tolist()
            logger.info(f"Detected {len(meter_ids)} meters: {meter_ids}")
        
        # Store series information
        series_info = {
            "series_id": series_id,
            "filename": filename,
            "dataframe": df_processed,
            "frequency": frequency,
            "schema_map": schema_map,
            "has_multiple_meters": has_multiple_meters,
            "meter_ids": meter_ids,
            "ingestion_time": datetime.utcnow().isoformat(),
            "n_records": len(df_processed),
            "date_range": [
                df_processed.index.min().isoformat(),
                df_processed.index.max().isoformat()
            ],
            "columns": df_processed.columns.tolist()
        }
        
        self.series_store[series_id] = series_info
        
        # Save to disk for persistence
        await self._save_series(series_id, series_info)
        
        return series_info
    
    async def get_series(
        self,
        series_id: str,
        meter_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve series data by ID.
        
        Args:
            series_id: Series identifier
            meter_id: Optional meter ID for multi-meter data
            
        Returns:
            Series information dictionary or None
        """
        if series_id not in self.series_store:
            # Try loading from disk
            await self._load_series(series_id)
        
        if series_id not in self.series_store:
            return None
        
        series_info = self.series_store[series_id].copy()
        df = series_info["dataframe"]
        
        # Filter by meter_id if specified
        if meter_id and "meter_id" in df.columns:
            df_filtered = df[df["meter_id"] == meter_id].copy()
            if len(df_filtered) == 0:
                return None
            series_info["dataframe"] = df_filtered
            series_info["meter_id"] = meter_id
        
        return series_info
    
    async def load_demo_data(self) -> List[Dict[str, str]]:
        """Load demo datasets for testing."""
        demo_series = []
        
        # Generate synthetic single-meter data
        single_meter_data = self._generate_demo_single_meter()
        series_info = await self.ingest_dataframe(
            df=single_meter_data,
            filename="demo_single_meter.csv",
            timezone="Asia/Kolkata"
        )
        demo_series.append({
            "series_id": series_info["series_id"],
            "name": "Demo Single Meter",
            "description": "Synthetic hourly energy load data for a commercial building"
        })
        
        # Generate synthetic multi-meter data
        multi_meter_data = self._generate_demo_multi_meter()
        series_info = await self.ingest_dataframe(
            df=multi_meter_data,
            filename="demo_multi_meter.csv",
            timezone="Asia/Kolkata"
        )
        demo_series.append({
            "series_id": series_info["series_id"],
            "name": "Demo Multi Meter",
            "description": "Synthetic hourly data for multiple meters in a campus"
        })
        
        return demo_series
    
    async def export_series(
        self,
        series_id: str,
        format: str = "csv",
        include_forecasts: bool = False
    ) -> str:
        """Export series data to file."""
        series_info = await self.get_series(series_id)
        if not series_info:
            raise ValueError(f"Series {series_id} not found")
        
        export_dir = self.data_dir / "exports"
        export_dir.mkdir(exist_ok=True)
        
        if format == "csv":
            filepath = export_dir / f"{series_id}.csv"
            series_info["dataframe"].to_csv(filepath)
        elif format == "json":
            filepath = export_dir / f"{series_id}.json"
            export_data = {
                "series_id": series_id,
                "metadata": {k: v for k, v in series_info.items() if k != "dataframe"},
                "data": series_info["dataframe"].reset_index().to_dict(orient="records")
            }
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return str(filepath)
    
    def _map_columns(self, columns: List[str]) -> Dict[str, str]:
        """Map DataFrame columns to standard schema."""
        schema_map = {}
        columns_lower = [col.lower() for col in columns]
        
        for schema_col, patterns in self.column_patterns.items():
            for pattern in patterns:
                for i, col_lower in enumerate(columns_lower):
                    if pattern in col_lower and schema_col not in schema_map:
                        schema_map[schema_col] = columns[i]
                        break
                if schema_col in schema_map:
                    break
        
        return schema_map
    
    def _infer_frequency(
        self,
        index: pd.DatetimeIndex,
        freq_hint: Optional[str] = None
    ) -> str:
        """Infer time series frequency."""
        if freq_hint:
            return freq_hint
        
        # Calculate most common time difference
        if len(index) < 2:
            return "H"  # Default to hourly
        
        diffs = index[1:] - index[:-1]
        mode_diff = diffs.mode()[0] if len(diffs.mode()) > 0 else diffs[0]
        
        # Map to pandas frequency strings
        if mode_diff <= timedelta(minutes=1):
            return "T"  # Minute
        elif mode_diff <= timedelta(hours=1):
            return "H"  # Hourly
        elif mode_diff <= timedelta(days=1):
            return "D"  # Daily
        elif mode_diff <= timedelta(weeks=1):
            return "W"  # Weekly
        else:
            return "M"  # Monthly
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df_clean = df.copy()
        
        # Forward fill short gaps (up to 3 consecutive missing values)
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            # Identify gaps
            missing_mask = df_clean[col].isna()
            
            # Forward fill short gaps
            df_clean[col] = df_clean[col].fillna(method='ffill', limit=3)
            
            # Log remaining missing values
            remaining_missing = df_clean[col].isna().sum()
            if remaining_missing > 0:
                logger.warning(f"Column {col}: {remaining_missing} missing values remain")
        
        # Remove rows with missing target values
        if "value" in df_clean.columns:
            initial_len = len(df_clean)
            df_clean = df_clean.dropna(subset=["value"])
            removed = initial_len - len(df_clean)
            if removed > 0:
                logger.info(f"Removed {removed} rows with missing target values")
        
        return df_clean
    
    def _generate_demo_single_meter(self) -> pd.DataFrame:
        """Generate synthetic single-meter demo data."""
        # 30 days of hourly data
        dates = pd.date_range(
            start="2024-01-01",
            end="2024-01-30",
            freq="H"
        )
        
        n_points = len(dates)
        np.random.seed(42)
        
        # Base load with daily and weekly patterns
        base_load = 100
        daily_pattern = 20 * np.sin(2 * np.pi * np.arange(n_points) / 24)
        weekly_pattern = 10 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 7))
        
        # Add noise and trend
        noise = np.random.normal(0, 5, n_points)
        trend = 0.1 * np.arange(n_points) / 24  # Slight upward trend
        
        # Combine components
        load = base_load + daily_pattern + weekly_pattern + trend + noise
        
        # Add some anomalies
        anomaly_indices = np.random.choice(n_points, size=5, replace=False)
        load[anomaly_indices] += np.random.normal(0, 30, 5)
        
        # Weather data
        temp = 25 + 5 * np.sin(2 * np.pi * np.arange(n_points) / 24) + np.random.normal(0, 2, n_points)
        humidity = 60 + 20 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 3)) + np.random.normal(0, 5, n_points)
        
        # Holiday flags (random)
        holiday_flag = np.random.choice([0, 1], size=n_points, p=[0.95, 0.05])
        
        return pd.DataFrame({
            "timestamp": dates,
            "value": np.maximum(load, 0),  # Ensure non-negative
            "temperature": temp,
            "humidity": np.clip(humidity, 0, 100),
            "holiday_flag": holiday_flag
        })
    
    def _generate_demo_multi_meter(self) -> pd.DataFrame:
        """Generate synthetic multi-meter demo data."""
        # 14 days of hourly data for 3 meters
        dates = pd.date_range(
            start="2024-02-01",
            end="2024-02-14",
            freq="H"
        )
        
        meter_ids = ["METER_001", "METER_002", "METER_003"]
        base_loads = [80, 120, 150]  # Different base loads per meter
        
        all_data = []
        
        for i, meter_id in enumerate(meter_ids):
            n_points = len(dates)
            np.random.seed(42 + i)  # Different seed per meter
            
            # Meter-specific patterns
            base_load = base_loads[i]
            daily_pattern = 15 * np.sin(2 * np.pi * np.arange(n_points) / 24 + i)
            weekly_pattern = 8 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 7))
            
            # Noise and meter-specific characteristics
            noise = np.random.normal(0, 3, n_points)
            meter_factor = 1 + 0.1 * i  # Different scaling per meter
            
            load = (base_load + daily_pattern + weekly_pattern) * meter_factor + noise
            
            # Add meter-specific anomalies
            anomaly_indices = np.random.choice(n_points, size=3, replace=False)
            load[anomaly_indices] += np.random.normal(0, 20, 3)
            
            # Common weather data (same for all meters)
            temp = 28 + 4 * np.sin(2 * np.pi * np.arange(n_points) / 24) + np.random.normal(0, 1.5, n_points)
            humidity = 65 + 15 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 2)) + np.random.normal(0, 3, n_points)
            
            meter_data = pd.DataFrame({
                "timestamp": dates,
                "meter_id": meter_id,
                "value": np.maximum(load, 0),
                "temperature": temp,
                "humidity": np.clip(humidity, 0, 100)
            })
            
            all_data.append(meter_data)
        
        return pd.concat(all_data, ignore_index=True)
    
    async def _save_series(self, series_id: str, series_info: Dict[str, Any]) -> None:
        """Save series to disk for persistence."""
        try:
            series_dir = self.data_dir / "series" / series_id
            series_dir.mkdir(parents=True, exist_ok=True)
            
            # Save DataFrame
            df_path = series_dir / "data.parquet"
            series_info["dataframe"].to_parquet(df_path)
            
            # Save metadata
            metadata = {k: v for k, v in series_info.items() if k != "dataframe"}
            metadata_path = series_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Failed to save series {series_id}: {e}")
    
    async def _load_series(self, series_id: str) -> None:
        """Load series from disk."""
        try:
            series_dir = self.data_dir / "series" / series_id
            
            if not series_dir.exists():
                return
            
            # Load DataFrame
            df_path = series_dir / "data.parquet"
            if df_path.exists():
                df = pd.read_parquet(df_path)
            else:
                return
            
            # Load metadata
            metadata_path = series_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # Reconstruct series info
            series_info = metadata.copy()
            series_info["dataframe"] = df
            
            self.series_store[series_id] = series_info
            
        except Exception as e:
            logger.warning(f"Failed to load series {series_id}: {e}")
