# OCR Model Findings - 2026-04-07

## Scope

This report summarizes the current OCR benchmark state after adding MallaNet-backed evaluation in the main pipeline.

## Run Configuration

Word benchmark:

- command: python src/evaluate_wordset.py --engines paddle,trocr,indic,malla --limit 80
- output: results/eval_wordset/20260407_213433

Detection benchmark:

- command: python src/evaluate_detection_testset.py --detectors tesseract,paddle,dbnet --limit 20
- output: results/eval_detection/20260407_214220

## Word Recognition Results (80 samples)

| Engine | Exact Match | Avg CER | Avg Confidence | Errors |
|---|---:|---:|---:|---:|
| paddle | 0.0500 | 0.8034 | 0.4239 | 0 |
| trocr | 0.0375 | 0.9237 | 0.0000 | 0 |
| indic | 0.0500 | 0.8034 | 0.4239 | 0 |
| malla | 0.0500 | 0.8506 | 0.6178 | 0 |

## Detection Results (20 images)

| Detector | Precision | Recall | HMean | Errors | First Error |
|---|---:|---:|---:|---:|---|
| tesseract | 0.2655 | 0.4250 | 0.3268 | 0 | |
| paddle | 0.0000 | 0.0000 | 0.0000 | 20 | ConvertPirAttribute2RuntimeAttribute not support pir::ArrayAttribute<pir::DoubleAttribute> |
| dbnet | 0.0000 | 0.0000 | 0.0000 | 20 | Missing checkpoint models/Nepali-Text-Detection-DBnet/models/nepali/nepali_td_best.pth |

## Key Findings

1. Word-level quality is currently low across all engines on this dataset split, with CER still high.
2. Malla-backed pipeline is now functional in benchmark flow, but does not yet outperform paddle/indic on CER.
3. TrOCR is stable but currently weakest on CER in this run.
4. Detection stack is bottlenecked by backend/runtime and missing checkpoints:
- Paddle detection fails at runtime in current environment.
- DBNet cannot run without trained checkpoint.
5. Tesseract is currently the only reliable detector baseline.

## Proposed Solutions

## Immediate (today to next 1-2 days)

1. Stabilize detection by keeping tesseract as primary detector and gating paddle/dbnet behind health checks.
2. Acquire and register DBNet checkpoint path, then re-run detection benchmark.
3. Pin Paddle and PaddleOCR versions known to be compatible on Windows runtime to avoid PIR conversion failure.

## Short-Term (this week)

1. Add dataset-specific preprocessing variants (denoise, threshold, deskew) and run ablation with fixed limits.
2. Add per-class or per-length error slices from predictions CSV to identify failure clusters.
3. Introduce weighted engine routing:
- use malla fallback only on low-confidence words,
- keep strong sequence extraction from document pipeline.

## Mid-Term (next sprint)

1. Add confidence calibration for TrOCR and Malla outputs so confidence values are comparable.
2. Expand benchmark set size and lock deterministic split for trend tracking.
3. Add daily benchmark report generation script to keep streak-friendly, repeatable experiment logs.

## Current Recommendation

Keep the production default on the stable document pipeline path with tesseract detection baseline, while using malla as fallback and improving detection checkpoint/runtime readiness before switching detector defaults.
