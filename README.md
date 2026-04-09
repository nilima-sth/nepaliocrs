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
- select: rank core OCR models and recommend fine-tune target
- kaggle: start app with TensorFlow check for kaggle model mode

Examples:

```bash
# Linux/macOS
./run.sh serve
./run.sh eval
./run.sh select
./run.sh kaggle
```

```bat
REM Windows
run.bat serve
run.bat eval
run.bat select
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

- file: image or PDF file (required). For PDF, first page is used.
- engine: indic, malla, paddle, trocr, or kaggle (optional, default paddle)
- include_debug: true|false (optional)
- include_segments: true|false (optional, default false)
- reference_text: optional expected text for token-level comparison against detected segments

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

- malla
- indic
- paddle

Environment variables:

- WORD_LIMIT (default 120)
- DET_LIMIT (default 30)

Outputs:

- results/eval_wordset/<run_id>/summary.md
- results/eval_wordset/<run_id>/summary.json
- results/eval_detection/<run_id>/summary.json

## Dashboard benchmark runner

The Flask dashboard now includes a Run Test Benchmark section.

It can execute in one run:

- Word-level k-fold evaluation (exact match, CER, confidence)
- Document-level extraction health metrics (word/sentence/paragraph counts, non-empty rate)
- Detection k-fold evaluation (precision, recall, hmean)

API route used by dashboard:

- POST /api/benchmark

Benchmark reports are stored at:

- results/dashboard_benchmark/<run_id>/report.json

Note:

- Document-level section is health-oriented because train/test page folders do not include paragraph-level ground-truth transcripts.

## Segmentation inspection in dashboard

In the main upload panel:

- Upload image or PDF (first page is automatically rendered for OCR).
- Core profile is shown by default (malla, indic, paddle).
- Optional toggle shows non-core research engines (trocr, kaggle).
- Optional: paste expected text in Ground Truth Text box.

Enable non-core profile from environment when needed:

- `OCR_ENABLE_NON_CORE_PROFILE=1`

The result panel now includes:

- Segmentation overlay view (word boxes with prediction labels).
- Segment table with bbox, predicted token, confidence, source, and optional match/mismatch.
- Token comparison summary when reference text is provided.

## Fast model selection

Use `select` mode for quick ranking with k-fold comparison:

```bash
./run.sh select
```

```bat
run.bat select
```

Outputs:

- `results/model_selection/<run_id>/ranking.json`
- `results/model_selection/<run_id>/ranking.md`

This gives:

- ranked core models by selection score
- recommended model to fine-tune next
- fine-tune hint for the winner

## CI/CD (GitHub Actions)

Two workflows are included:

- CI: `.github/workflows/ci.yml`
  - Trigger: push to `main`/`Gradio`, pull requests
  - Uses lightweight dependencies from `requirements-ci.txt`
  - Runs source compile check and unit tests

- CD: `.github/workflows/cd.yml`
  - Trigger: manual (`workflow_dispatch`)
  - Builds and pushes Docker image to GHCR
  - Supports `include_models=true|false` to control heavy model download during image build

Notes:

- CI is intentionally fast and does not download large model checkpoints.
- CD can build lean images (`include_models=false`) or full images (`include_models=true`).

## De-fluff checklist

If the project feels noisy, keep only one clear path for each concern:

- Inference path: keep one default engine chain and mark alternatives as optional.
- Evaluation path: one benchmark command, one output folder schema.
- Deployment path: one Dockerfile, one CD target (GHCR).
- Dependencies: keep runtime deps in `requirements.txt`, CI deps in `requirements-ci.txt`.
- Experiments: keep notebooks/experimental scripts out of runtime CI checks.

## Runtime vs research split

- Runtime boundary notes: `runtime/BOUNDARY.md`
- Research boundary notes: `research/BOUNDARY.md`

Research directories moved under `research/`:

- `research/MallaNet/`
- `research/TraificNPR/`
- `research/tools/`

## Branch protection checklist

Use and enforce:

- `.github/BRANCH_PROTECTION_CHECKLIST.md`

Key policy:

- require CI `quality` check
- no direct push to protected branches
- PR + approval before merge

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