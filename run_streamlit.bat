@echo off
REM Eye Disease Detection - Streamlit Setup Script
REM This script sets up and runs the Streamlit application

echo.
echo ============================================================
echo  Eye Disease Detection - Streamlit Web Application
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org
    pause
    exit /b 1
)

echo [1/3] Checking Python installation...
python --version
echo OK - Python is installed
echo.

REM Check if requirements file exists
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found
    echo Please ensure you are in the correct directory
    pause
    exit /b 1
)

REM Check if model file exists
if not exist "eye_disease_model_boosted_drusen.h5" (
    echo WARNING: Model file not found!
    echo Expected: eye_disease_model_boosted_drusen.h5
    echo The app will fail to run without the model file
    pause
    exit /b 1
)

echo [2/3] Installing dependencies...
echo This may take a few minutes on first run...
echo.
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [3/3] Starting Streamlit application...
echo.
echo Application will open in your browser at: http://localhost:8501
echo Press Ctrl+C to stop the application
echo.

streamlit run streamlit_app.py

pause
