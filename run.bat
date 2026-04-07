@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "MODE=%~1"
if "%MODE%"=="" set "MODE=serve"

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

if /I "%MODE%"=="serve" goto :run_serve
if /I "%MODE%"=="eval" goto :run_eval
if /I "%MODE%"=="kaggle" goto :run_kaggle

echo [ERROR] Unknown mode: %MODE%
echo [INFO] Valid modes: serve ^| eval ^| kaggle
pause
exit /b 1

:run_serve
echo [INFO] Mode: serve
echo [INFO] Starting Flask App...
".\.venv\Scripts\python.exe" src\app.py
goto :done

:run_eval
echo [INFO] Mode: eval
set "WORD_LIMIT=%WORD_LIMIT%"
if "%WORD_LIMIT%"=="" set "WORD_LIMIT=120"

set "DET_LIMIT=%DET_LIMIT%"
if "%DET_LIMIT%"=="" set "DET_LIMIT=30"

echo [INFO] Running word recognition benchmark...
".\.venv\Scripts\python.exe" src\evaluate_wordset.py --engines paddle,trocr,indic,malla --limit %WORD_LIMIT%
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Word benchmark failed.
    pause
    exit /b 1
)

echo [INFO] Running detection benchmark...
".\.venv\Scripts\python.exe" src\evaluate_detection_testset.py --detectors tesseract,paddle,dbnet --limit %DET_LIMIT%
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Detection benchmark failed.
    pause
    exit /b 1
)

echo [DONE] Evaluation runs completed. Check results\eval_wordset and results\eval_detection.
goto :done

:run_kaggle
echo [INFO] Mode: kaggle
".\.venv\Scripts\python.exe" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('tensorflow') else 1)"
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] tensorflow not found. Installing tensorflow-cpu and h5py...
    ".\.venv\Scripts\pip.exe" install tensorflow-cpu h5py
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install tensorflow runtime.
        pause
        exit /b 1
    )
)

if "%PORT%"=="" set "PORT=5000"
echo [INFO] Starting Flask App with kaggle-ready runtime on port %PORT%...
".\.venv\Scripts\python.exe" src\app.py

:done
pause