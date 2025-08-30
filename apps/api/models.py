"""
Pydantic models for SmartSense API request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime


class IngestResponse(BaseModel):
    """Response model for data ingestion endpoint."""
    series_id: str = Field(..., description="Unique identifier for the ingested series")
    filename: str = Field(..., description="Original filename")
    n_records: int = Field(..., description="Number of records ingested")
    date_range: List[str] = Field(..., description="Start and end dates")
    frequency: str = Field(..., description="Detected frequency (H, D, etc.)")
    columns: List[str] = Field(..., description="Available columns")
    schema_map: Dict[str, str] = Field(..., description="Column mapping")
    has_multiple_meters: bool = Field(..., description="Whether data contains multiple meters")
    meter_ids: List[str] = Field(default=[], description="List of meter IDs if applicable")


class ForecastResponse(BaseModel):
    """Response model for forecast endpoint."""
    series_id: str = Field(..., description="Series identifier")
    model: str = Field(..., description="Model used for forecasting")
    horizon: int = Field(..., description="Forecast horizon")
    timestamps: List[str] = Field(..., description="Forecast timestamps (ISO format)")
    forecasts: List[float] = Field(..., description="Point forecasts")
    lower_bound: Optional[List[float]] = Field(None, description="Lower prediction interval")
    upper_bound: Optional[List[float]] = Field(None, description="Upper prediction interval")
    model_info: Dict[str, Any] = Field(..., description="Model metadata")
    weather_included: bool = Field(False, description="Whether weather data was used")


class AnomalyResponse(BaseModel):
    """Response model for anomaly detection endpoint."""
    series_id: str = Field(..., description="Series identifier")
    detector: str = Field(..., description="Detector used")
    window_size: int = Field(..., description="Analysis window size")
    anomaly_indices: List[int] = Field(..., description="Indices of detected anomalies")
    anomaly_scores: List[float] = Field(..., description="Anomaly scores")
    anomaly_labels: List[str] = Field(..., description="Anomaly type labels")
    anomaly_messages: List[str] = Field(..., description="Human-readable messages")
    threshold: float = Field(..., description="Detection threshold used")
    changepoints: List[int] = Field(default=[], description="Detected changepoints")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")


class BacktestMetrics(BaseModel):
    """Metrics for a single model in backtesting."""
    model: str = Field(..., description="Model name")
    mape: float = Field(..., description="Mean Absolute Percentage Error")
    smape: float = Field(..., description="Symmetric MAPE")
    mae: float = Field(..., description="Mean Absolute Error")
    rmse: float = Field(..., description="Root Mean Square Error")
    mase: Optional[float] = Field(None, description="Mean Absolute Scaled Error")
    n_forecasts: int = Field(..., description="Number of forecasts evaluated")


class BacktestResponse(BaseModel):
    """Response model for backtesting endpoint."""
    series_id: str = Field(..., description="Series identifier")
    models_tested: List[str] = Field(..., description="Models included in backtest")
    metrics: List[BacktestMetrics] = Field(..., description="Metrics for each model")
    leaderboard: List[Dict[str, Any]] = Field(..., description="Ranked model performance")
    best_model: str = Field(..., description="Best performing model")
    plot_urls: List[str] = Field(default=[], description="URLs to generated plots")
    metadata: Dict[str, Any] = Field(..., description="Backtest metadata")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    models_available: List[str] = Field(..., description="Available forecasting models")
    memory_usage_mb: float = Field(..., description="Current memory usage in MB")


class SeriesConfig(BaseModel):
    """Configuration for series processing."""
    timezone: str = Field("Asia/Kolkata", description="Timezone for timestamp parsing")
    frequency: Optional[str] = Field(None, description="Data frequency hint")
    target_column: str = Field("value", description="Target variable column name")
    timestamp_column: str = Field("timestamp", description="Timestamp column name")
    meter_id_column: Optional[str] = Field(None, description="Meter ID column name")
    weather_columns: List[str] = Field(default=[], description="Weather variable columns")
    holiday_column: Optional[str] = Field(None, description="Holiday flag column name")


class RealTimeRequest(BaseModel):
    """Request model for real-time anomaly detection."""
    series_id: str = Field(..., description="Series identifier")
    value: float = Field(..., description="New observation value")
    timestamp: Optional[str] = Field(None, description="Observation timestamp")
    weather_data: Optional[Dict[str, float]] = Field(None, description="Current weather data")


class RealTimeResponse(BaseModel):
    """Response model for real-time detection."""
    is_anomaly: bool = Field(..., description="Whether observation is anomalous")
    score: float = Field(..., description="Anomaly score")
    label: Optional[str] = Field(None, description="Anomaly type if detected")
    message: Optional[str] = Field(None, description="Human-readable message")
    forecast: Optional[float] = Field(None, description="Expected value forecast")
    confidence_interval: Optional[List[float]] = Field(None, description="Prediction interval")


class ExportRequest(BaseModel):
    """Request model for data export."""
    series_id: str = Field(..., description="Series identifier")
    format: str = Field("csv", description="Export format (csv, json, excel)")
    include_forecasts: bool = Field(False, description="Include forecast results")
    include_anomalies: bool = Field(False, description="Include anomaly results")
    date_range: Optional[List[str]] = Field(None, description="Date range filter")


class WeatherData(BaseModel):
    """Weather data structure."""
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    humidity: Optional[float] = Field(None, description="Humidity percentage")
    pressure: Optional[float] = Field(None, description="Atmospheric pressure")
    wind_speed: Optional[float] = Field(None, description="Wind speed")
    weather_code: Optional[int] = Field(None, description="Weather condition code")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")
