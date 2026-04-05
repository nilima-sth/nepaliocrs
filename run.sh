#!/bin/bash
set -e

MODE="${1:-serve}"

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

case "$MODE" in
    serve)
        echo "[INFO] Mode: serve"
        echo "[INFO] Starting Flask App..."
        .venv/bin/python3 src/app.py
        ;;

    eval)
        echo "[INFO] Mode: eval"
        WORD_LIMIT="${WORD_LIMIT:-120}"
        DET_LIMIT="${DET_LIMIT:-30}"

        echo "[INFO] Running word recognition benchmark..."
        .venv/bin/python3 src/evaluate_wordset.py --engines paddle,trocr,indic --limit "${WORD_LIMIT}"

        echo "[INFO] Running detection benchmark..."
        .venv/bin/python3 src/evaluate_detection_testset.py --detectors tesseract,paddle,dbnet --limit "${DET_LIMIT}"

        echo "[DONE] Evaluation runs completed. Check results/eval_wordset and results/eval_detection."
        ;;

    kaggle)
        echo "[INFO] Mode: kaggle"
        if ! .venv/bin/python3 - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("tensorflow") else 1)
PY
        then
            echo "[INFO] tensorflow not found. Installing tensorflow-cpu and h5py..."
            .venv/bin/pip3 install tensorflow-cpu h5py
        fi

        export PORT="${PORT:-5000}"
        echo "[INFO] Starting Flask App with kaggle-ready runtime on port ${PORT}..."
        .venv/bin/python3 src/app.py
        ;;

    *)
        echo "[ERROR] Unknown mode: ${MODE}"
        echo "[INFO] Valid modes: serve | eval | kaggle"
        exit 1
        ;;
esac
