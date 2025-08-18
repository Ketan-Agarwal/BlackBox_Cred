@echo off
REM Quick start script for CredTech Backend on Windows

echo ========================================
echo   CredTech Backend - Quick Start
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies if requirements.txt is newer than last install
if not exist "venv\.install_complete" (
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Error: Failed to install dependencies
        pause
        exit /b 1
    )
    echo. > venv\.install_complete
)

REM Check if .env exists
if not exist ".env" (
    echo Warning: .env file not found
    if exist ".env.example" (
        echo Copying .env.example to .env...
        copy .env.example .env
        echo Please edit .env file with your configuration before running the server.
        pause
    ) else (
        echo Error: .env.example file not found
        pause
        exit /b 1
    )
)

echo.
echo Starting CredTech Backend Server...
echo.
echo The server will be available at:
echo   - API Documentation: http://127.0.0.1:8000/docs
echo   - Health Check: http://127.0.0.1:8000/api/health
echo   - Status: http://127.0.0.1:8000/status
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

pause
