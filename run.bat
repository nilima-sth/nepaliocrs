@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

:: Check if Python is available
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

:: Check if virtual environment exists, create if it doesn't
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Activate the virtual environment
echo [INFO] Activating virtual environment...
call .\.venv\Scripts\activate.bat

:: Upgrade pip and install wheel/numpy first for safe compilation
echo [INFO] Ensuring base packaging tools are installed...
".\.venv\Scripts\python.exe" -m pip install --upgrade pip wheel numpy >nul 2>&1

:: Install requirements if not fully met
echo [INFO] Checking and installing dependencies from requirements.txt...
".\.venv\Scripts\pip.exe" install -r requirements.txt

:: Run the app
echo [INFO] Starting Flask App...
".\.venv\Scripts\python.exe" src\app.py

pause