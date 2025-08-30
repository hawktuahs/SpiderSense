"""
HuggingFace Spaces deployment for SmartSense.

Single-file deployment combining FastAPI backend with static frontend
for zero-config deployment on HuggingFace Spaces.
"""

import os
import sys
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "libs"))
sys.path.append(str(project_root / "apps" / "api"))

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

# Import SmartSense components
from main import app as api_app

# Create combined app
app = FastAPI(
    title="SmartSense - Energy Load Forecasting & Anomaly Detection",
    description="Scalable energy intelligence for India's Smart Cities Mission",
    version="1.0.0"
)

# Mount API routes
app.mount("/api", api_app)

# Serve static files (built Next.js app)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    """Serve the main application."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    else:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SmartSense - Energy Intelligence</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 2rem; background: #f8fafc; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
                .header { text-align: center; margin-bottom: 2rem; }
                .logo { width: 64px; height: 64px; background: linear-gradient(135deg, #ff9933, #ffffff, #138808); border-radius: 12px; margin: 0 auto 1rem; }
                h1 { color: #1f2937; margin: 0; }
                .subtitle { color: #6b7280; margin: 0.5rem 0 0; }
                .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 2rem 0; }
                .feature { padding: 1rem; border: 1px solid #e5e7eb; border-radius: 6px; }
                .feature h3 { margin: 0 0 0.5rem; color: #374151; }
                .feature p { margin: 0; color: #6b7280; font-size: 0.9rem; }
                .cta { text-align: center; margin-top: 2rem; }
                .btn { display: inline-block; padding: 0.75rem 1.5rem; background: #3b82f6; color: white; text-decoration: none; border-radius: 6px; font-weight: 500; }
                .btn:hover { background: #2563eb; }
                .api-link { margin-top: 1rem; }
                .api-link a { color: #3b82f6; text-decoration: none; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo"></div>
                    <h1>SmartSense</h1>
                    <p class="subtitle">Energy Load Forecasting & Anomaly Detection for India</p>
                </div>
                
                <div class="features">
                    <div class="feature">
                        <h3>🔮 Advanced Forecasting</h3>
                        <p>Multiple models including ARIMA, ETS, and NHITS-tiny with 20-30% improved accuracy</p>
                    </div>
                    <div class="feature">
                        <h3>🚨 Real-time Anomalies</h3>
                        <p>Robust statistical detection with seasonal awareness and changepoint analysis</p>
                    </div>
                    <div class="feature">
                        <h3>📊 Comprehensive Analytics</h3>
                        <p>Backtesting, model comparison, and exportable reports with visualizations</p>
                    </div>
                    <div class="feature">
                        <h3>🇮🇳 India-First Design</h3>
                        <p>Built for Smart Cities Mission with weather integration and holiday calendars</p>
                    </div>
                </div>
                
                <div class="cta">
                    <a href="/docs" class="btn">Explore API Documentation</a>
                    <div class="api-link">
                        <p>Direct API access: <a href="/api/health">/api/health</a></p>
                    </div>
                </div>
                
                <div style="margin-top: 2rem; padding-top: 2rem; border-top: 1px solid #e5e7eb; text-align: center; color: #6b7280; font-size: 0.9rem;">
                    <p>Open source under MIT License • CPU-optimized • Free deployment</p>
                    <p>Supporting India's Net-Zero 2070 goals</p>
                </div>
            </div>
        </body>
        </html>
        """)

@app.get("/health")
async def health():
    """Health check for the combined app."""
    return {"status": "ok", "message": "SmartSense HuggingFace deployment running"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # HF Spaces default port
    uvicorn.run(app, host="0.0.0.0", port=port)
