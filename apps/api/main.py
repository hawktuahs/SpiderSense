"""
SmartSense FastAPI Backend

Main API server providing energy load forecasting and anomaly detection endpoints.
Designed for CPU-only deployment with <1GB memory usage and <10s cold start.
"""

import os
import sys
import logging
from pathlib import Path

# Add libs to path
sys.path.append(str(Path(__file__).parent.parent.parent / "libs"))

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import uuid
import json
from datetime import datetime, timedelta

from forecasting import BaseForecaster, NaiveSeasonalForecaster, ETSForecaster, ARIMAForecaster
from anomaly import AnomalyPipeline, create_anomaly_pipeline
from .models import (
    IngestResponse, ForecastResponse, AnomalyResponse, 
    BacktestResponse, HealthResponse, SeriesConfig
)
from .data_manager import DataManager
from .weather_client import WeatherClient
from .backtesting import BacktestEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SmartSense API",
    description="Scalable Energy Load Forecasting & Anomaly Detection for India",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
data_manager = DataManager()
weather_client = WeatherClient()
backtest_engine = BacktestEngine()

# Model registry
AVAILABLE_MODELS = {
    "naive_seasonal": NaiveSeasonalForecaster,
    "ets": ETSForecaster,
    "arima": ARIMAForecaster
}


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting SmartSense API...")
    
    # Initialize data directory
    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("data/cache", exist_ok=True)
    os.makedirs("data/exports", exist_ok=True)
    
    # Load demo data if available
    try:
        await data_manager.load_demo_data()
        logger.info("Demo data loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load demo data: {e}")
    
    logger.info("SmartSense API started successfully")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        models_available=list(AVAILABLE_MODELS.keys()),
        memory_usage_mb=_get_memory_usage()
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_data(
    file: UploadFile = File(...),
    timezone: str = Query("Asia/Kolkata", description="Timezone for timestamp parsing"),
    freq_hint: Optional[str] = Query(None, description="Frequency hint (H, D, etc.)")
):
    """
    Ingest CSV data for forecasting and anomaly detection.
    
    Expected CSV columns (flexible auto-detection):
    - timestamp: datetime column
    - value: target variable (energy load)
    - meter_id: optional meter identifier for multi-meter data
    - temp, humidity: optional weather variables
    - holiday_flag: optional holiday indicator
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read CSV data
        content = await file.read()
        df = pd.read_csv(pd.io.common.StringIO(content.decode('utf-8')))
        
        # Process and validate data
        series_info = await data_manager.ingest_dataframe(
            df=df,
            filename=file.filename,
            timezone=timezone,
            freq_hint=freq_hint
        )
        
        return IngestResponse(
            series_id=series_info["series_id"],
            filename=file.filename,
            n_records=series_info["n_records"],
            date_range=series_info["date_range"],
            frequency=series_info["frequency"],
            columns=series_info["columns"],
            schema_map=series_info["schema_map"],
            has_multiple_meters=series_info["has_multiple_meters"],
            meter_ids=series_info.get("meter_ids", [])
        )
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise HTTPException(status_code=400, detail=f"Data ingestion failed: {str(e)}")


@app.get("/forecast", response_model=ForecastResponse)
async def generate_forecast(
    series_id: str = Query(..., description="Series ID from ingestion"),
    h: int = Query(48, description="Forecast horizon (number of steps)"),
    model: str = Query("arima", description="Forecasting model"),
    meter_id: Optional[str] = Query(None, description="Specific meter ID for multi-meter data"),
    agg: str = Query("hourly", description="Aggregation level (hourly, daily)"),
    city: Optional[str] = Query(None, description="City for weather data"),
    include_weather: bool = Query(True, description="Include weather forecasts"),
    quantiles: str = Query("0.1,0.5,0.9", description="Comma-separated quantile levels")
):
    """Generate energy load forecasts with prediction intervals."""
    try:
        # Parse quantiles
        quantile_list = [float(q.strip()) for q in quantiles.split(",")]
        
        # Get series data
        series_data = await data_manager.get_series(series_id, meter_id)
        if series_data is None:
            raise HTTPException(status_code=404, detail="Series not found")
        
        # Initialize forecaster
        if model not in AVAILABLE_MODELS:
            raise HTTPException(status_code=400, detail=f"Model '{model}' not available")
        
        forecaster_class = AVAILABLE_MODELS[model]
        forecaster = forecaster_class(random_seed=42)
        
        # Prepare data
        df = series_data["dataframe"]
        freq = series_data["frequency"]
        
        # Get weather data if requested
        exog_future = None
        if include_weather and city:
            try:
                weather_data = await weather_client.get_forecast(city, h, freq)
                exog_future = weather_data
            except Exception as e:
                logger.warning(f"Weather data unavailable: {e}")
        
        # Fit model and generate forecasts
        forecaster.fit(df, freq=freq, target_col="value")
        forecast_result = forecaster.predict(h=h, exog_future=exog_future, quantiles=quantile_list)
        
        # Prepare response
        response_data = {
            "series_id": series_id,
            "model": model,
            "horizon": h,
            "timestamps": [ts.isoformat() for ts in forecast_result.timestamps],
            "forecasts": forecast_result.yhat.tolist(),
            "lower_bound": forecast_result.yhat_lower.tolist() if forecast_result.yhat_lower is not None else None,
            "upper_bound": forecast_result.yhat_upper.tolist() if forecast_result.yhat_upper is not None else None,
            "model_info": forecast_result.model_info,
            "weather_included": include_weather and exog_future is not None
        }
        
        return ForecastResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Forecast generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")


@app.get("/anomaly", response_model=AnomalyResponse)
async def detect_anomalies(
    series_id: str = Query(..., description="Series ID from ingestion"),
    meter_id: Optional[str] = Query(None, description="Specific meter ID"),
    window: int = Query(168, description="Analysis window (hours)"),
    detector: str = Query("robust", description="Anomaly detector (robust, ewma)"),
    threshold: float = Query(3.0, description="Detection threshold"),
    include_changepoints: bool = Query(True, description="Include changepoint detection")
):
    """Detect anomalies in energy load data."""
    try:
        # Get series data
        series_data = await data_manager.get_series(series_id, meter_id)
        if series_data is None:
            raise HTTPException(status_code=404, detail="Series not found")
        
        df = series_data["dataframe"]
        freq = series_data["frequency"]
        
        # Limit to analysis window
        if len(df) > window:
            df = df.tail(window)
        
        # Generate baseline forecast for residuals
        forecaster = NaiveSeasonalForecaster(random_seed=42)
        forecaster.fit(df, freq=freq, target_col="value")
        residuals = forecaster.get_residuals()
        
        # Initialize anomaly pipeline
        pipeline = create_anomaly_pipeline(
            detector_type=detector,
            seasonal_periods=forecaster.seasonal_periods,
            threshold=threshold,
            enable_changepoints=include_changepoints
        )
        
        # Run anomaly detection
        result = pipeline.detect(
            residuals=residuals,
            timestamps=df.index,
            original_data=df["value"].values
        )
        
        # Prepare response
        response_data = {
            "series_id": series_id,
            "detector": detector,
            "window_size": len(df),
            "anomaly_indices": result.anomalies.indices.tolist(),
            "anomaly_scores": result.anomalies.scores.tolist(),
            "anomaly_labels": result.anomalies.labels,
            "anomaly_messages": result.anomalies.messages,
            "threshold": result.anomalies.threshold,
            "changepoints": result.changepoints.changepoints if result.changepoints else [],
            "summary": result.summary
        }
        
        return AnomalyResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")


@app.post("/backtest", response_model=BacktestResponse)
async def run_backtest(
    series_id: str = Query(..., description="Series ID from ingestion"),
    models: str = Query("naive_seasonal,ets,arima", description="Comma-separated model list"),
    train_size: float = Query(0.8, description="Training data fraction"),
    horizon: int = Query(24, description="Forecast horizon for each test"),
    step_size: int = Query(12, description="Step size for sliding window"),
    export_plots: bool = Query(True, description="Export visualization plots")
):
    """Run backtesting comparison across multiple models."""
    try:
        # Parse model list
        model_list = [m.strip() for m in models.split(",")]
        invalid_models = [m for m in model_list if m not in AVAILABLE_MODELS]
        if invalid_models:
            raise HTTPException(status_code=400, detail=f"Invalid models: {invalid_models}")
        
        # Get series data
        series_data = await data_manager.get_series(series_id)
        if series_data is None:
            raise HTTPException(status_code=404, detail="Series not found")
        
        # Run backtest
        results = await backtest_engine.run_backtest(
            series_data=series_data,
            models=model_list,
            train_size=train_size,
            horizon=horizon,
            step_size=step_size
        )
        
        # Export plots if requested
        plot_urls = []
        if export_plots:
            plot_urls = await backtest_engine.export_plots(results, series_id)
        
        return BacktestResponse(
            series_id=series_id,
            models_tested=model_list,
            metrics=results["metrics"],
            leaderboard=results["leaderboard"],
            best_model=results["best_model"],
            plot_urls=plot_urls,
            metadata=results["metadata"]
        )
        
    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        raise HTTPException(status_code=500, detail=f"Backtesting failed: {str(e)}")


@app.get("/series/{series_id}/export")
async def export_series_data(
    series_id: str,
    format: str = Query("csv", description="Export format (csv, json)"),
    include_forecasts: bool = Query(False, description="Include latest forecasts")
):
    """Export series data and results."""
    try:
        # Get series data
        series_data = await data_manager.get_series(series_id)
        if series_data is None:
            raise HTTPException(status_code=404, detail="Series not found")
        
        # Export data
        export_path = await data_manager.export_series(
            series_id, format, include_forecasts
        )
        
        return FileResponse(
            path=export_path,
            filename=f"smartsense_{series_id}.{format}",
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.get("/demo/load")
async def load_demo_data():
    """Load demo datasets for testing."""
    try:
        demo_series = await data_manager.load_demo_data()
        return {
            "message": "Demo data loaded successfully",
            "series": demo_series
        }
    except Exception as e:
        logger.error(f"Demo data loading failed: {e}")
        raise HTTPException(status_code=500, detail=f"Demo data loading failed: {str(e)}")


def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
