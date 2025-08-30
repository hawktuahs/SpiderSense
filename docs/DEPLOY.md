# SmartSense Deployment Guide

**Free-tier deployment options for energy forecasting platform**

## 🎯 Deployment Overview

SmartSense supports multiple deployment strategies optimized for free-tier constraints:
- **Memory**: <1GB usage
- **Cold Start**: <10 seconds
- **Cost**: $0/month on free tiers
- **Scalability**: Auto-scaling ready

## 🚀 Quick Deploy Options

### Option 1: HuggingFace Spaces (Recommended)
**Best for:** Demo, prototyping, single-user access

```bash
# 1. Clone repository
git clone https://github.com/your-org/smartsense
cd smartsense

# 2. Deploy to HuggingFace Spaces
cp deploy/huggingface/* .
git add .
git commit -m "Deploy SmartSense to HF Spaces"
git push origin main
```

**Features:**
- ✅ Zero configuration
- ✅ Automatic HTTPS
- ✅ Built-in authentication
- ✅ 16GB storage
- ⚠️ CPU-only (perfect for our models)

### Option 2: Vercel + Render
**Best for:** Production, multi-user, custom domain

**Frontend (Vercel):**
```bash
cd apps/web
npm install
vercel --prod
```

**Backend (Render):**
```bash
# Create render.yaml in project root
# Deploy via Render dashboard or CLI
```

**Features:**
- ✅ Global CDN
- ✅ Custom domains
- ✅ Environment variables
- ✅ Auto-scaling

### Option 3: Railway
**Best for:** Full-stack deployment, databases

```bash
railway login
railway init
railway up
```

## 📋 Prerequisites

### System Requirements
- **Python**: 3.9+ (3.11 recommended)
- **Node.js**: 18+ (for frontend)
- **Memory**: 512MB minimum, 1GB recommended
- **Storage**: 2GB for models and cache

### Environment Variables
```bash
# Required
PYTHONPATH=/app
TIMEZONE=Asia/Kolkata

# Optional (enhances functionality)
OWM_API_KEY=your_openweather_key
NEXT_PUBLIC_API_URL=https://your-api.onrender.com

# Production
ENVIRONMENT=production
LOG_LEVEL=INFO
```

## 🔧 Detailed Setup

### HuggingFace Spaces Deployment

#### Step 1: Prepare Repository
```bash
# Copy deployment files
cp deploy/huggingface/app.py .
cp deploy/huggingface/requirements.txt .
cp deploy/huggingface/README.md .

# Create .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo ".env" >> .gitignore
```

#### Step 2: Configure Space
```yaml
# In README.md header
---
title: SmartSense Energy Forecasting
emoji: ⚡
colorFrom: orange
colorTo: green
sdk: gradio
sdk_version: 4.7.1
app_file: app.py
pinned: false
license: mit
---
```

#### Step 3: Environment Setup
```bash
# In HF Spaces settings, add:
OWM_API_KEY=your_key_here
TIMEZONE=Asia/Kolkata
```

#### Step 4: Deploy
```bash
git add .
git commit -m "Initial SmartSense deployment"
git push origin main
```

### Vercel + Render Deployment

#### Frontend (Vercel)

**Step 1: Configure Next.js**
```javascript
// next.config.js
module.exports = {
  experimental: {
    appDir: true,
  },
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL}/:path*`,
      },
    ];
  },
};
```

**Step 2: Deploy to Vercel**
```bash
cd apps/web
npm install
vercel --prod

# Set environment variables in Vercel dashboard
NEXT_PUBLIC_API_URL=https://smartsense-api.onrender.com
```

#### Backend (Render)

**Step 1: Create render.yaml**
```yaml
services:
  - type: web
    name: smartsense-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn apps.api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHONPATH
        value: /opt/render/project/src
      - key: TIMEZONE
        value: Asia/Kolkata
      - key: OWM_API_KEY
        sync: false
```

**Step 2: Deploy to Render**
1. Connect GitHub repository
2. Select "Web Service"
3. Configure build and start commands
4. Add environment variables
5. Deploy

### Railway Deployment

**Step 1: Configure railway.json**
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn apps.api.main:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

**Step 2: Deploy**
```bash
railway login
railway init
railway up
```

## 🔒 Security Configuration

### Environment Variables
```bash
# Production security
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=your-domain.com,localhost
CORS_ORIGINS=https://your-frontend.vercel.app

# API Keys (optional)
OWM_API_KEY=your-openweather-key

# Database (if using)
DATABASE_URL=postgresql://user:pass@host:port/db
```

### CORS Setup
```python
# In main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📊 Monitoring & Health Checks

### Health Check Endpoint
```python
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "models_available": get_available_models(),
        "memory_usage_mb": get_memory_usage()
    }
```

### Logging Configuration
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("smartsense")
```

### Performance Monitoring
```python
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start_time
        logger.info(f"{func.__name__} took {duration:.2f}s")
        return result
    return wrapper
```

## 🧪 Testing Deployment

### Automated Tests
```bash
# Install test dependencies
pip install pytest httpx

# Run API tests
pytest tests/test_api.py -v

# Run end-to-end tests
pytest tests/test_e2e.py -v
```

### Manual Testing Checklist
```bash
# Health check
curl https://your-deployment.com/health

# Data ingestion
curl -X POST -F "file=@examples/demo_single_meter.csv" \
  https://your-deployment.com/ingest

# Forecasting
curl "https://your-deployment.com/forecast?series_id=demo123&h=48"

# Anomaly detection
curl "https://your-deployment.com/anomaly?series_id=demo123"

# Frontend access
open https://your-frontend.vercel.app
```

## 🔧 Troubleshooting

### Common Issues

#### Memory Errors
```bash
# Reduce model complexity
export ARIMA_MAX_ORDER=2
export ETS_TREND_DAMPED=true
export NHITS_DISABLE=true
```

#### Cold Start Timeout
```bash
# Optimize imports
export LAZY_IMPORTS=true
export PRELOAD_MODELS=false
```

#### CORS Errors
```python
# Update CORS origins
CORS_ORIGINS=["https://your-domain.com", "http://localhost:3000"]
```

### Performance Optimization

#### Model Caching
```python
from functools import lru_cache

@lru_cache(maxsize=10)
def get_cached_model(model_type: str):
    return load_model(model_type)
```

#### Response Compression
```python
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

## 📈 Scaling Considerations

### Horizontal Scaling
- **Load Balancer**: Use Cloudflare or similar
- **Database**: PostgreSQL for persistence
- **Cache**: Redis for model caching
- **Queue**: Celery for background tasks

### Vertical Scaling
- **Memory**: Upgrade to 2GB+ for larger datasets
- **CPU**: Multi-core for parallel processing
- **Storage**: SSD for faster model loading

## 🔄 CI/CD Pipeline

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy SmartSense
on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Test API
        run: |
          pip install -r requirements.txt
          pytest tests/ -v

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Render
        uses: render-deploy-action@v1
        with:
          service-id: ${{ secrets.RENDER_SERVICE_ID }}
          api-key: ${{ secrets.RENDER_API_KEY }}
```

## 📚 Additional Resources

### Documentation
- [API Reference](./API.md)
- [Impact Assessment](./IMPACT.md)
- [Model Documentation](./MODELS.md)

### Support
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Community support and questions
- **Email**: support@smartsense.energy (if available)

### Examples
- [Python SDK Examples](../examples/python_sdk.py)
- [JavaScript Examples](../examples/js_examples.js)
- [cURL Examples](../examples/curl_examples.sh)

---

**Deployment Summary:**
- **Free Tier**: HuggingFace Spaces (recommended for demos)
- **Production**: Vercel + Render (recommended for scale)
- **Enterprise**: Railway or self-hosted (recommended for control)

*Choose the deployment option that best fits your needs and scale requirements.*
