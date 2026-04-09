from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from ocr_engines import build_text_engine, levenshtein_distance, normalize_text


@dataclass
class Sample:
    image_path: Path
    label: str


@dataclass
class Prediction:
    engine: str
    image_path: str
    ground_truth: str
    predicted: str
    confidence: float
    cer: float
    exact_match: int
    status: str
    error: str


def parse_labels(labels_csv: Path, images_dir: Path) -> List[Sample]:
    samples: List[Sample] = []

    lines: List[str] = []
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            with labels_csv.open("r", encoding=enc) as f:
                lines = list(f)
            break
        except UnicodeError:
            continue

    if not lines:
        raise RuntimeError(f"Unable to decode labels file with supported encodings: {labels_csv}")

    for raw in lines:
            line = raw.strip()
            if not line:
                continue

            filename: Optional[str] = None
            label: Optional[str] = None

            if "," in line:
                first, rest = line.split(",", 1)
                filename = first.strip()
                label = rest.strip()
            else:
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    filename, label = parts[0].strip(), parts[1].strip()

            if not filename or label is None:
                continue

            image_path = images_dir / filename
            if image_path.exists():
                samples.append(Sample(image_path=image_path, label=label))

    return samples


def evaluate_engine(engine_name: str, samples: List[Sample], trocr_model_id: str) -> Tuple[Dict[str, float], List[Prediction]]:
    engine = build_text_engine(engine_name, trocr_model_id=trocr_model_id)

    preds: List[Prediction] = []
    total_cer = 0.0
    total_conf = 0.0
    conf_count = 0
    exact = 0
    ok_count = 0

    for sample in samples:
        gt = normalize_text(sample.label)

        try:
            image = Image.open(sample.image_path).convert("RGB")
            result = engine.predict(image)
        except Exception as err:
            preds.append(
                Prediction(
                    engine=engine_name,
                    image_path=str(sample.image_path),
                    ground_truth=gt,
                    predicted="",
                    confidence=0.0,
                    cer=1.0,
                    exact_match=0,
                    status="error",
                    error=str(err),
                )
            )
            continue

        if result.status != "ok":
            preds.append(
                Prediction(
                    engine=engine_name,
                    image_path=str(sample.image_path),
                    ground_truth=gt,
                    predicted="",
                    confidence=0.0,
                    cer=1.0,
                    exact_match=0,
                    status="error",
                    error=result.error or "unknown engine error",
                )
            )
            continue

        pred = normalize_text(result.text)
        distance = levenshtein_distance(pred, gt)
        cer = distance / max(1, len(gt))
        is_exact = int(pred == gt)

        total_cer += cer
        total_conf += float(result.conf)
        conf_count += 1
        exact += is_exact
        ok_count += 1

        preds.append(
            Prediction(
                engine=engine_name,
                image_path=str(sample.image_path),
                ground_truth=gt,
                predicted=pred,
                confidence=float(result.conf),
                cer=cer,
                exact_match=is_exact,
                status="ok",
                error="",
            )
        )

    total_samples = len(samples)
    metrics = {
        "engine": engine_name,
        "samples": total_samples,
        "ok": ok_count,
        "errors": total_samples - ok_count,
        "exact_match_rate": (exact / max(1, ok_count)),
        "avg_cer": (total_cer / max(1, ok_count)),
        "avg_conf": (total_conf / max(1, conf_count)),
    }

    return metrics, preds


def write_predictions_csv(path: Path, rows: List[Prediction]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "engine",
            "image_path",
            "ground_truth",
            "predicted",
            "confidence",
            "cer",
            "exact_match",
            "status",
            "error",
        ])
        for row in rows:
            writer.writerow([
                row.engine,
                row.image_path,
                row.ground_truth,
                row.predicted,
                f"{row.confidence:.6f}",
                f"{row.cer:.6f}",
                row.exact_match,
                row.status,
                row.error,
            ])


def write_summary_markdown(path: Path, metrics: List[Dict[str, float]]) -> None:
    lines = [
        "# OCR Wordset Evaluation Summary",
        "",
        "| Engine | Samples | OK | Errors | Exact Match | Avg CER | Avg Confidence |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for m in metrics:
        lines.append(
            "| {engine} | {samples} | {ok} | {errors} | {exact:.4f} | {cer:.4f} | {conf:.4f} |".format(
                engine=m["engine"],
                samples=int(m["samples"]),
                ok=int(m["ok"]),
                errors=int(m["errors"]),
                exact=float(m["exact_match_rate"]),
                cer=float(m["avg_cer"]),
                conf=float(m["avg_conf"]),
            )
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCR engines on labeled Nepali word crops.")
    parser.add_argument("--labels-csv", default="datas/archivelabelled/labels.csv")
    parser.add_argument("--images-dir", default="datas/archivelabelled/crops")
    parser.add_argument("--engines", default="malla,indic,paddle")
    parser.add_argument("--trocr-model", default="paudelanil/trocr-devanagari-2")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", default="results/eval_wordset")
    args = parser.parse_args()

    labels_csv = Path(args.labels_csv)
    images_dir = Path(args.images_dir)

    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    samples = parse_labels(labels_csv, images_dir)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    if not samples:
        raise RuntimeError("No valid labeled samples found. Check labels.csv and image paths.")

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    if not engines:
        raise RuntimeError("No engines selected. Use --engines malla,indic,paddle")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[Dict[str, float]] = []

    for engine_name in engines:
        metrics, rows = evaluate_engine(engine_name, samples, trocr_model_id=args.trocr_model)
        all_metrics.append(metrics)
        write_predictions_csv(out_dir / f"predictions_{engine_name}.csv", rows)

    (out_dir / "summary.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    write_summary_markdown(out_dir / "summary.md", all_metrics)

    print(f"[DONE] Evaluation completed. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
