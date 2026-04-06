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

if /I "%BOOTSTRAP%"=="1" (
    :: Upgrade pip and install wheel/numpy first for safe compilation
    echo [INFO] BOOTSTRAP=1 -> Ensuring base packaging tools are installed...
    ".\.venv\Scripts\python.exe" -m pip install --upgrade pip wheel numpy

    :: Install requirements
    echo [INFO] BOOTSTRAP=1 -> Installing dependencies from requirements.txt...
    ".\.venv\Scripts\pip.exe" install -r requirements.txt

    :: Download necessary models only during explicit bootstrap
    echo [INFO] BOOTSTRAP=1 -> Downloading OCR models...
    ".\.venv\Scripts\python.exe" src\download_models.py
) else (
    echo [INFO] Fast startup mode: skipping bootstrap tasks.
    echo [INFO] Set BOOTSTRAP=1 to reinstall dependencies and re-check model downloads.
)

:: Runtime safety defaults to avoid VS Code/host lag
set "FLASK_DEBUG=0"
set "OCR_DISABLE_RELOADER=1"
set "OCR_MAX_UPLOAD_MB=10"
set "OCR_MAX_IMAGE_SIDE=2200"
set "OCR_MAX_CONCURRENT_REQUESTS=1"
set "OCR_UNLOAD_MODELS_EACH_REQUEST=0"
set "OCR_ENABLE_PADDLE_FALLBACK=0"

:: Run the app
echo [INFO] Starting Flask App...
".\.venv\Scripts\python.exe" src\app.py

pause