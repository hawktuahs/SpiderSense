# SmartSense API Reference

Complete API documentation for energy load forecasting and anomaly detection endpoints.

## Base URL

```
Production: https://smartsense-api.onrender.com
Development: http://localhost:8000
```

## Authentication

No authentication required. All endpoints are publicly accessible for the open-source version.

## Rate Limits

- **Free Tier**: 100 requests/hour per IP
- **Self-hosted**: No limits

## Data Ingestion

### POST /ingest

Upload CSV data for forecasting and anomaly detection.

**Request:**
```bash
curl -X POST \
  -F "file=@energy_data.csv" \
  -F "timezone=Asia/Kolkata" \
  -F "freq_hint=H" \
  http://localhost:8000/ingest
```

**Parameters:**
- `file` (required): CSV file with energy data
- `timezone` (optional): Timezone for timestamp parsing (default: "Asia/Kolkata")
- `freq_hint` (optional): Frequency hint ("H", "D", etc.)

**Response:**
```json
{
  "series_id": "abc12345",
  "filename": "energy_data.csv",
  "n_records": 720,
  "date_range": ["2024-01-01T00:00:00", "2024-01-30T23:00:00"],
  "frequency": "H",
  "columns": ["timestamp", "value", "temperature", "humidity"],
  "schema_map": {
    "timestamp": "timestamp",
    "value": "value",
    "temperature": "temperature",
    "humidity": "humidity"
  },
  "has_multiple_meters": false,
  "meter_ids": []
}
```

**CSV Format:**
```csv
timestamp,value,temperature,humidity,holiday_flag
2024-01-01 00:00:00,85.2,22.1,68,1
2024-01-01 01:00:00,78.9,21.8,70,1
```

**Required columns:**
- `timestamp`: DateTime in any standard format
- `value`: Energy load/consumption values

**Optional columns:**
- `meter_id`: For multi-meter datasets
- `temperature`, `humidity`: Weather variables
- `holiday_flag`: Holiday indicator (0/1)

## Forecasting

### GET /forecast

Generate energy load forecasts with prediction intervals.

**Request:**
```bash
curl "http://localhost:8000/forecast?series_id=abc12345&h=48&model=arima&city=Mumbai&include_weather=true"
```

**Parameters:**
- `series_id` (required): Series ID from ingestion
- `h` (optional): Forecast horizon in steps (default: 48)
- `model` (optional): Forecasting model (default: "arima")
  - `"naive_seasonal"`: Seasonal naive baseline
  - `"ets"`: Exponential smoothing
  - `"arima"`: Auto-ARIMA
  - `"nhits_tiny"`: Neural forecasting (if available)
- `meter_id` (optional): Specific meter for multi-meter data
- `agg` (optional): Aggregation level ("hourly", "daily")
- `city` (optional): City for weather data
- `include_weather` (optional): Include weather forecasts (default: true)
- `quantiles` (optional): Comma-separated quantiles (default: "0.1,0.5,0.9")

**Response:**
```json
{
  "series_id": "abc12345",
  "model": "arima",
  "horizon": 48,
  "timestamps": [
    "2024-01-31T00:00:00",
    "2024-01-31T01:00:00"
  ],
  "forecasts": [89.4, 92.1],
  "lower_bound": [78.2, 80.5],
  "upper_bound": [100.6, 103.7],
  "model_info": {
    "model_type": "ARIMAForecaster",
    "arima_order": [1, 1, 1],
    "seasonal_order": [1, 1, 1, 24],
    "aic": 2847.3
  },
  "weather_included": true
}
```

## Anomaly Detection

### GET /anomaly

Detect anomalies in energy load data.

**Request:**
```bash
curl "http://localhost:8000/anomaly?series_id=abc12345&window=168&detector=robust&threshold=3.0"
```

**Parameters:**
- `series_id` (required): Series ID from ingestion
- `meter_id` (optional): Specific meter for multi-meter data
- `window` (optional): Analysis window in hours (default: 168)
- `detector` (optional): Anomaly detector (default: "robust")
  - `"robust"`: MAD-based robust z-score
  - `"ewma"`: Exponentially weighted moving average
- `threshold` (optional): Detection threshold (default: 3.0)
- `include_changepoints` (optional): Include changepoint detection (default: true)

**Response:**
```json
{
  "series_id": "abc12345",
  "detector": "robust",
  "window_size": 168,
  "anomaly_indices": [45, 123],
  "anomaly_scores": [3.2, -2.8],
  "anomaly_labels": ["spike", "drop"],
  "anomaly_messages": [
    "Spike: +3.2σ at 2024-01-02 21:00",
    "Drop: -2.8σ at 2024-01-05 03:00"
  ],
  "threshold": 3.0,
  "changepoints": [67, 145],
  "summary": {
    "total_points": 168,
    "anomaly_count": 2,
    "anomaly_rate": 0.012,
    "severity": "medium",
    "max_anomaly_score": 3.2
  }
}
```

## Backtesting

### POST /backtest

Run model comparison and backtesting.

**Request:**
```bash
curl -X POST "http://localhost:8000/backtest?series_id=abc12345&models=naive_seasonal,ets,arima&train_size=0.8&horizon=24"
```

**Parameters:**
- `series_id` (required): Series ID from ingestion
- `models` (optional): Comma-separated model list (default: "naive_seasonal,ets,arima")
- `train_size` (optional): Training data fraction (default: 0.8)
- `horizon` (optional): Forecast horizon for each test (default: 24)
- `step_size` (optional): Step size for sliding window (default: 12)
- `export_plots` (optional): Export visualization plots (default: true)

**Response:**
```json
{
  "series_id": "abc12345",
  "models_tested": ["naive_seasonal", "ets", "arima"],
  "metrics": [
    {
      "model": "arima",
      "mape": 8.2,
      "smape": 7.9,
      "mae": 12.4,
      "rmse": 18.7,
      "mase": 0.82,
      "n_forecasts": 15
    }
  ],
  "leaderboard": [
    {
      "rank": 1,
      "model": "arima",
      "composite_score": 0.23,
      "mape": 8.2,
      "smape": 7.9
    }
  ],
  "best_model": "arima",
  "plot_urls": [
    "/exports/model_comparison_abc12345.png",
    "/exports/metrics_heatmap_abc12345.png"
  ],
  "metadata": {
    "total_windows": 15,
    "train_size": 0.8,
    "horizon": 24
  }
}
```

## Utilities

### GET /health

Health check endpoint.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2024-01-31T12:00:00Z",
  "version": "1.0.0",
  "models_available": ["naive_seasonal", "ets", "arima"],
  "memory_usage_mb": 245.6
}
```

### GET /demo/load

Load demo datasets for testing.

**Request:**
```bash
curl http://localhost:8000/demo/load
```

**Response:**
```json
{
  "message": "Demo data loaded successfully",
  "series": [
    {
      "series_id": "demo_single",
      "name": "Demo Single Meter",
      "description": "Synthetic hourly energy load data"
    },
    {
      "series_id": "demo_multi",
      "name": "Demo Multi Meter", 
      "description": "Synthetic multi-meter campus data"
    }
  ]
}
```

### GET /series/{series_id}/export

Export series data and results.

**Request:**
```bash
curl "http://localhost:8000/series/abc12345/export?format=csv&include_forecasts=true"
```

**Parameters:**
- `format` (optional): Export format ("csv", "json") (default: "csv")
- `include_forecasts` (optional): Include latest forecasts (default: false)

**Response:**
File download with series data and metadata.

## Error Handling

All endpoints return standard HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (series not found)
- `422`: Validation Error (invalid input data)
- `500`: Internal Server Error

**Error Response Format:**
```json
{
  "detail": "Series not found",
  "error_code": "SERIES_NOT_FOUND",
  "timestamp": "2024-01-31T12:00:00Z"
}
```

## Rate Limiting

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1706702400
```

## Webhooks (Future)

Real-time anomaly alerts via webhooks:

```bash
curl -X POST http://localhost:8000/webhooks/register \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-app.com/anomaly-alert",
    "series_id": "abc12345",
    "threshold": 2.5
  }'
```

## SDKs and Examples

### Python SDK
```python
import requests

# Upload data
with open('energy_data.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/ingest',
        files={'file': f}
    )
series_id = response.json()['series_id']

# Generate forecast
forecast = requests.get(
    f'http://localhost:8000/forecast?series_id={series_id}&h=48'
).json()

# Detect anomalies
anomalies = requests.get(
    f'http://localhost:8000/anomaly?series_id={series_id}'
).json()
```

### JavaScript/Node.js
```javascript
const FormData = require('form-data');
const fs = require('fs');

// Upload data
const form = new FormData();
form.append('file', fs.createReadStream('energy_data.csv'));

const response = await fetch('http://localhost:8000/ingest', {
  method: 'POST',
  body: form
});
const { series_id } = await response.json();

// Generate forecast
const forecast = await fetch(
  `http://localhost:8000/forecast?series_id=${series_id}&h=48`
).then(r => r.json());
```

### cURL Examples
```bash
# Complete workflow
SERIES_ID=$(curl -s -X POST -F "file=@data.csv" http://localhost:8000/ingest | jq -r .series_id)
curl "http://localhost:8000/forecast?series_id=$SERIES_ID&h=48" | jq .
curl "http://localhost:8000/anomaly?series_id=$SERIES_ID" | jq .
```

## OpenAPI Specification

Interactive API documentation available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`
