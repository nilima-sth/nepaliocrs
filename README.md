# Nepali OCRs

Nepali OCRs is a Flask-based OCR service with method-wise evaluation scripts for Nepali document text extraction.

## Requirements

- Python 3.10
- pip

## Setup

Install dependencies manually:

```bash
pip install -r requirements.txt
```

Or use bootstrap in run scripts:

- Windows: set BOOTSTRAP=1 and run run.bat
- Linux/macOS: BOOTSTRAP=1 ./run.sh

## Main run commands

Two entry scripts are used at repository root:

- run.bat (Windows)
- run.sh (Linux/macOS)

Supported modes:

- serve: start Flask OCR API/UI
- eval: run word and detection benchmarks
- kaggle: start app with TensorFlow check for kaggle model mode

Examples:

```bash
# Linux/macOS
./run.sh serve
./run.sh eval
./run.sh kaggle
```

```bat
REM Windows
run.bat serve
run.bat eval
run.bat kaggle
```

Default mode is serve when no mode is passed.

## API endpoint

Base URL:

- http://127.0.0.1:5000

Protected machine endpoint:

- POST /api/v1/extract

Headers:

- X-API-Token: <token>
- or Authorization: Bearer <token>

Form fields:

- file: image file (required)
- engine: indic, malla, paddle, trocr, or kaggle (optional, default paddle)
- include_debug: true|false (optional)

Example:

```bash
curl -X POST "http://127.0.0.1:5000/api/v1/extract" \
  -H "X-API-Token: replace-with-a-long-random-token" \
  -F "engine=indic" \
  -F "include_debug=false" \
  -F "file=@/absolute/path/to/image.jpg"
```

## Method-wise evaluation

Mode eval runs both:

- Word recognition benchmark: src/evaluate_wordset.py
- Detection benchmark: src/evaluate_detection_testset.py

Default word benchmark engines in eval mode:

- paddle
- trocr
- indic
- malla

Environment variables:

- WORD_LIMIT (default 120)
- DET_LIMIT (default 30)

Outputs:

- results/eval_wordset/<run_id>/summary.md
- results/eval_wordset/<run_id>/summary.json
- results/eval_detection/<run_id>/summary.json

## Model bootstrap

Model helper script:

- src/download_models.py

It currently ensures availability of:

- PaddleOCR-VL source files
- Indic-HTR repository
- TrOCR snapshot
- DBNet repository

Important:

- DBNet and Indic-HTR still require trained checkpoint files (.pth/.ckpt) for real inference.

## Docker

Build:

```bash
docker build -t nepali-ocr-app .
```

Run:

```bash
docker run -p 5000:5000 nepali-ocr-app
```