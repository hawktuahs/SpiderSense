@echo off
echo SmartSense Local Setup for Windows
echo ==================================

echo.
echo Step 1: Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install Python dependencies
    pause
    exit /b 1
)

echo.
echo Step 2: Installing Node.js dependencies...
cd apps\web
npm install
if %errorlevel% neq 0 (
    echo Failed to install Node.js dependencies
    pause
    exit /b 1
)

cd ..\..

echo.
echo Step 3: Creating environment file...
if not exist .env (
    copy .env.example .env
    echo Created .env file from .env.example
)

echo.
echo Setup complete! 
echo.
echo To run SmartSense:
echo 1. Start API: cd apps\api && python -m uvicorn main:app --reload --port 8000
echo 2. Start Web: cd apps\web && npm run dev
echo 3. Open: http://localhost:3000
echo.
pause
