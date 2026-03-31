#!/bin/bash
set -e

echo "==================================================="
echo "  Nepali License Plate OCR App Initialization"
echo "==================================================="

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

# Upgrade pip and install wheel/numpy first for safe compilation
echo "[INFO] Ensuring base packaging tools are installed..."
python3 -m pip install --upgrade pip wheel numpy

# Install requirements
echo "[INFO] Checking and installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Run the app
echo "[INFO] Starting Gradio App..."
python3 src/gradio_app.py
