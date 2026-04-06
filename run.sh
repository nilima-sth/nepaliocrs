#!/bin/bash
set -e


# Change to the directory of the script
cd "$(dirname "$0")"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3."
    exit 1
fi

# Check if virtual environment exists, create if it doesn't
if [ ! -f ".venv/bin/activate" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment."
        exit 1
    fi
fi

# Activate the virtual environment
echo "[INFO] Activating virtual environment..."
source .venv/bin/activate

# Bootstrap is now opt-in to keep startup responsive
if [ "${BOOTSTRAP:-0}" = "1" ]; then
    echo "[INFO] BOOTSTRAP=1 -> Ensuring base packaging tools are installed..."
    .venv/bin/python3 -m pip install --upgrade pip wheel numpy

    echo "[INFO] BOOTSTRAP=1 -> Installing dependencies from requirements.txt..."
    .venv/bin/pip3 install -r requirements.txt

    echo "[INFO] BOOTSTRAP=1 -> Downloading OCR models..."
    .venv/bin/python3 src/download_models.py
else
    echo "[INFO] Fast startup mode: skipping bootstrap tasks."
    echo "[INFO] Set BOOTSTRAP=1 to reinstall dependencies and re-check model downloads."
fi

# Runtime safety defaults to avoid editor/system lag
export FLASK_DEBUG=0
export OCR_DISABLE_RELOADER=1
export OCR_MAX_UPLOAD_MB=10
export OCR_MAX_IMAGE_SIDE=2200
export OCR_MAX_CONCURRENT_REQUESTS=1
export OCR_UNLOAD_MODELS_EACH_REQUEST=0
export OCR_ENABLE_PADDLE_FALLBACK=0

# Run the app
echo "[INFO] Starting Flask App..."
.venv/bin/python3 src/app.py
