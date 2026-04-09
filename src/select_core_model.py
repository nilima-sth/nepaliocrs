from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dashboard_benchmark import run_word_kfold


def _bounded(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _selection_score(metrics: Dict[str, float]) -> float:
    exact = _bounded(metrics.get("mean_exact_match", 0.0), 0.0, 1.0)
    cer = _bounded(metrics.get("mean_avg_cer", 1.0), 0.0, 2.0)
    conf = _bounded(metrics.get("mean_avg_conf", 0.0), 0.0, 1.0)
    std_cer = _bounded(metrics.get("std_avg_cer", 0.0), 0.0, 2.0)

    # Higher exact/conf is better, lower CER and lower CER variance are better.
    # CER is weighted most because exact-match rates can remain low across OCR engines.
    score = (0.25 * exact) + (0.65 * (1.0 - min(1.0, cer))) + (0.10 * conf) - (0.05 * std_cer)
    return round(float(score), 6)


def _fine_tune_hint(engine_name: str) -> str:
    if engine_name == "malla":
        return "Fine-tune character model with harder low-confidence crops and tune fallback threshold in document pipeline."
    if engine_name == "indic":
        return "Fine-tune sequence OCR on normalized document crops and improve line segmentation consistency."
    if engine_name == "paddle":
        return "Fine-tune Paddle recognition with Nepali-focused text lines and robust preprocessing augmentations."
    if engine_name == "trocr":
        return "Fine-tune decoder with domain text and stronger handwriting augmentation to reduce CER drift."
    return "Collect failure cases first, then fine-tune on cleaned train/validation splits."


def _write_markdown(
    path: Path,
    ranking: List[Dict[str, object]],
    winner: str,
    run_meta: Dict[str, object],
    analysis_notes: List[str],
) -> None:
    lines = [
        "# Core Model Selection",
        "",
        f"- Run ID: {run_meta['run_id']}",
        f"- Timestamp: {run_meta['timestamp']}",
        f"- Sample count: {run_meta['sample_count']}",
        f"- K folds: {run_meta['k_folds']}",
        f"- Recommended engine: **{winner}**",
        "",
        "| Rank | Engine | Selection Score | Mean Exact | Mean CER | Mean Conf | CER Std |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]

    for row in ranking:
        lines.append(
            "| {rank} | {engine} | {score:.4f} | {exact:.4f} | {cer:.4f} | {conf:.4f} | {std_cer:.4f} |".format(
                rank=row["rank"],
                engine=row["engine"],
                score=float(row["selection_score"]),
                exact=float(row["mean_exact_match"]),
                cer=float(row["mean_avg_cer"]),
                conf=float(row["mean_avg_conf"]),
                std_cer=float(row["std_avg_cer"]),
            )
        )

    lines.extend([
        "",
        "## Analysis Notes",
        "",
    ])

    if analysis_notes:
        for note in analysis_notes:
            lines.append(f"- {note}")
    else:
        lines.append("- No additional notes.")

    lines.extend([
        "",
        "## Fine-tune Plan",
        "",
        _fine_tune_hint(winner),
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run core-model ranking from k-fold word benchmark and recommend a fine-tune target.")
    parser.add_argument("--engines", default="malla,indic,paddle")
    parser.add_argument("--labels-csv", default="datas/archivelabelled/labels.csv")
    parser.add_argument("--images-dir", default="datas/archivelabelled/crops")
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trocr-model", default="paudelanil/trocr-devanagari-2")
    parser.add_argument("--output-dir", default="results/model_selection")
    args = parser.parse_args()

    engines = [item.strip().lower() for item in str(args.engines).split(",") if item.strip()]
    if not engines:
        raise RuntimeError("No engines selected. Example: --engines malla,indic,paddle")

    labels_csv = Path(args.labels_csv)
    images_dir = Path(args.images_dir)
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    report = run_word_kfold(
        engines=engines,
        k_folds=args.k_folds,
        sample_limit=args.limit,
        seed=args.seed,
        labels_csv=labels_csv,
        images_dir=images_dir,
        trocr_model_id=args.trocr_model,
    )

    per_engine = report.get("engines", {})
    ranking: List[Dict[str, object]] = []
    for engine_name, metrics in per_engine.items():
        row = {
            "engine": engine_name,
            "selection_score": _selection_score(metrics),
            "mean_exact_match": float(metrics.get("mean_exact_match", 0.0)),
            "mean_avg_cer": float(metrics.get("mean_avg_cer", 1.0)),
            "mean_avg_conf": float(metrics.get("mean_avg_conf", 0.0)),
            "std_avg_cer": float(metrics.get("std_avg_cer", 0.0)),
        }
        ranking.append(row)

    ranking.sort(key=lambda row: (-float(row["selection_score"]), float(row["mean_avg_cer"]), -float(row["mean_exact_match"])))
    for idx, row in enumerate(ranking, start=1):
        row["rank"] = idx

    if not ranking:
        raise RuntimeError("No engine metrics produced. Cannot rank models.")

    analysis_notes: List[str] = []
    signatures: Dict[tuple, List[str]] = {}
    for row in ranking:
        signature = (
            round(float(row["mean_exact_match"]), 6),
            round(float(row["mean_avg_cer"]), 6),
            round(float(row["mean_avg_conf"]), 6),
        )
        signatures.setdefault(signature, []).append(str(row["engine"]))

    for engines_with_same_metrics in signatures.values():
        if len(engines_with_same_metrics) > 1:
            joined = ", ".join(sorted(engines_with_same_metrics))
            analysis_notes.append(
                f"Engines with identical metric signature: {joined}. Validate whether they currently share the same inference path."
            )

    winner = str(ranking[0]["engine"])
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "engines": engines,
            "k_folds": int(report.get("k_folds", args.k_folds)),
            "sample_count": int(report.get("sample_count", 0)),
            "limit": int(args.limit),
            "seed": int(args.seed),
            "labels_csv": str(labels_csv),
            "images_dir": str(images_dir),
        },
        "recommended_engine": winner,
        "fine_tune_hint": _fine_tune_hint(winner),
        "analysis_notes": analysis_notes,
        "ranking": ranking,
        "raw_word_kfold": report,
    }

    (out_dir / "ranking.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    _write_markdown(
        out_dir / "ranking.md",
        ranking=ranking,
        winner=winner,
        run_meta={
            "run_id": run_id,
            "timestamp": output["timestamp"],
            "sample_count": output["config"]["sample_count"],
            "k_folds": output["config"]["k_folds"],
        },
        analysis_notes=analysis_notes,
    )

    print(f"[DONE] Core model ranking saved to: {out_dir}")
    print(f"[RECOMMENDED] {winner}")


if __name__ == "__main__":
    main()
