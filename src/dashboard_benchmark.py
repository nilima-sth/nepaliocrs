from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from sklearn.model_selection import KFold

from evaluate_detection_testset import collect_pairs, match_boxes, parse_xml_boxes
from ocr_engines import (
    DBNetDetector,
    PaddleTextDetector,
    TesseractTextDetector,
    build_text_engine,
    levenshtein_distance,
    normalize_text,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class WordSample:
    image_path: Path
    label: str


def _safe_float(value: float, digits: int = 6) -> float:
    return float(round(float(value), digits))


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = _mean(values)
    var = sum((float(v) - avg) ** 2 for v in values) / len(values)
    return float(var ** 0.5)


def _list_document_images(doc_dirs: Iterable[Path]) -> List[Path]:
    images: List[Path] = []
    for directory in doc_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                images.append(path)
    return images


def _load_word_samples(labels_csv: Path, images_dir: Path) -> List[WordSample]:
    lines: List[str] = []
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            lines = labels_csv.read_text(encoding=enc).splitlines()
            break
        except UnicodeError:
            continue

    if not lines:
        raise RuntimeError(f"Unable to decode labels file: {labels_csv}")

    samples: List[WordSample] = []
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
            samples.append(WordSample(image_path=image_path, label=label))

    return samples


def _evaluate_word_subset(engine, samples: Sequence[WordSample]) -> Dict[str, float]:
    total = len(samples)
    ok = 0
    errors = 0
    exact = 0
    cer_sum = 0.0
    conf_sum = 0.0

    for sample in samples:
        gt = normalize_text(sample.label)
        try:
            image = Image.open(sample.image_path).convert("RGB")
            result = engine.predict(image)
        except Exception:
            errors += 1
            continue

        if result.status != "ok":
            errors += 1
            continue

        pred = normalize_text(result.text)
        dist = levenshtein_distance(pred, gt)
        cer = dist / max(1, len(gt))

        ok += 1
        exact += int(pred == gt)
        cer_sum += cer
        conf_sum += float(result.conf)

    return {
        "samples": total,
        "ok": ok,
        "errors": errors,
        "exact_match": exact / max(1, ok),
        "avg_cer": cer_sum / max(1, ok),
        "avg_conf": conf_sum / max(1, ok),
    }


def run_word_kfold(
    engines: Sequence[str],
    k_folds: int,
    sample_limit: int,
    seed: int,
    labels_csv: Path,
    images_dir: Path,
    trocr_model_id: str,
) -> Dict[str, object]:
    samples = _load_word_samples(labels_csv=labels_csv, images_dir=images_dir)
    if sample_limit > 0:
        samples = samples[:sample_limit]

    if len(samples) < 2:
        raise RuntimeError("Not enough word samples for k-fold evaluation.")

    k = max(2, min(int(k_folds), len(samples)))
    splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
    splits = list(splitter.split(samples))

    report: Dict[str, object] = {
        "sample_count": len(samples),
        "k_folds": k,
        "engines": {},
    }

    for engine_name in engines:
        engine = build_text_engine(engine_name, trocr_model_id=trocr_model_id)
        fold_rows: List[Dict[str, object]] = []

        for fold_idx, (_, test_idx) in enumerate(splits, start=1):
            test_samples = [samples[i] for i in test_idx]
            metrics = _evaluate_word_subset(engine, test_samples)
            row = {
                "fold": fold_idx,
                "test_size": len(test_samples),
                "exact_match": _safe_float(metrics["exact_match"]),
                "avg_cer": _safe_float(metrics["avg_cer"]),
                "avg_conf": _safe_float(metrics["avg_conf"]),
                "errors": int(metrics["errors"]),
            }
            fold_rows.append(row)

        exact_vals = [float(r["exact_match"]) for r in fold_rows]
        cer_vals = [float(r["avg_cer"]) for r in fold_rows]
        conf_vals = [float(r["avg_conf"]) for r in fold_rows]

        report["engines"][engine_name] = {
            "mean_exact_match": _safe_float(_mean(exact_vals)),
            "std_exact_match": _safe_float(_std(exact_vals)),
            "mean_avg_cer": _safe_float(_mean(cer_vals)),
            "std_avg_cer": _safe_float(_std(cer_vals)),
            "mean_avg_conf": _safe_float(_mean(conf_vals)),
            "std_avg_conf": _safe_float(_std(conf_vals)),
            "folds": fold_rows,
        }

    return report


def _count_text_features(text: str) -> Tuple[int, int, int, int]:
    raw = text or ""
    words = re.findall(r"\S+", raw)
    sentences = [s for s in re.split(r"[।.!?]+", raw) if s.strip()]
    paragraphs = [p for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if not paragraphs and raw.strip():
        paragraphs = [raw.strip()]
    return len(raw.strip()), len(words), len(sentences), len(paragraphs)


def run_document_health(
    engines: Sequence[str],
    image_limit: int,
    doc_dirs: Sequence[Path],
    trocr_model_id: str,
) -> Dict[str, object]:
    images = _list_document_images(doc_dirs)
    if image_limit > 0:
        images = images[:image_limit]

    if not images:
        raise RuntimeError("No document images found for document-level benchmark.")

    report: Dict[str, object] = {
        "image_count": len(images),
        "engines": {},
        "note": "Document-level section reports extraction health metrics. Ground-truth text labels are not available in archive train/test.",
    }

    for engine_name in engines:
        engine = build_text_engine(engine_name, trocr_model_id=trocr_model_id)

        ok = 0
        errors = 0
        non_empty = 0
        conf_sum = 0.0
        char_sum = 0
        word_sum = 0
        sentence_sum = 0
        paragraph_sum = 0

        for image_path in images:
            try:
                image = Image.open(image_path).convert("RGB")
                result = engine.predict(image)
            except Exception:
                errors += 1
                continue

            if result.status != "ok":
                errors += 1
                continue

            ok += 1
            conf_sum += float(result.conf)
            char_count, word_count, sentence_count, paragraph_count = _count_text_features(result.text)
            if char_count > 0:
                non_empty += 1

            char_sum += char_count
            word_sum += word_count
            sentence_sum += sentence_count
            paragraph_sum += paragraph_count

        report["engines"][engine_name] = {
            "ok_images": ok,
            "error_images": errors,
            "non_empty_rate": _safe_float(non_empty / max(1, ok)),
            "avg_conf": _safe_float(conf_sum / max(1, ok)),
            "avg_chars": _safe_float(char_sum / max(1, ok)),
            "avg_words": _safe_float(word_sum / max(1, ok)),
            "avg_sentences": _safe_float(sentence_sum / max(1, ok)),
            "avg_paragraphs": _safe_float(paragraph_sum / max(1, ok)),
        }

    return report


def _build_detector(name: str, dbnet_checkpoint: Optional[str]):
    key = (name or "").strip().lower()
    if key == "tesseract":
        return TesseractTextDetector(lang_chain="nep+hin+eng")
    if key == "paddle":
        return PaddleTextDetector(lang="hi", use_gpu=False)
    if key == "dbnet":
        return DBNetDetector(checkpoint_path=dbnet_checkpoint, use_gpu=False)
    raise ValueError(f"Unsupported detector: {name}")


def _evaluate_detection_subset(detector, pairs: Sequence[Tuple[Path, Path]], iou_threshold: float) -> Dict[str, object]:
    tp_total = 0
    fp_total = 0
    fn_total = 0
    ok_images = 0
    errors = 0
    first_error = ""

    for image_path, xml_path in pairs:
        gt_boxes = parse_xml_boxes(xml_path)

        try:
            image = Image.open(image_path).convert("RGB")
            pred_boxes = detector.detect_boxes(image)
            tp, fp, fn = match_boxes(gt_boxes, pred_boxes, threshold=iou_threshold)
            ok_images += 1
        except Exception as err:
            tp, fp, fn = 0, 0, len(gt_boxes)
            errors += 1
            if not first_error:
                first_error = str(err)

        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision = tp_total / max(1, tp_total + fp_total)
    recall = tp_total / max(1, tp_total + fn_total)
    hmean = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

    return {
        "tp": int(tp_total),
        "fp": int(fp_total),
        "fn": int(fn_total),
        "ok_images": int(ok_images),
        "errors": int(errors),
        "first_error": first_error,
        "precision": _safe_float(precision),
        "recall": _safe_float(recall),
        "hmean": _safe_float(hmean),
    }


def run_detection_kfold(
    detectors: Sequence[str],
    k_folds: int,
    pair_limit: int,
    seed: int,
    iou_threshold: float,
    train_dir: Path,
    test_dir: Path,
    dbnet_checkpoint: Optional[str],
) -> Dict[str, object]:
    all_pairs = collect_pairs(train_dir, train_dir) + collect_pairs(test_dir, test_dir)
    if pair_limit > 0:
        all_pairs = all_pairs[:pair_limit]

    if len(all_pairs) < 2:
        raise RuntimeError("Not enough annotated document pairs for detection k-fold.")

    k = max(2, min(int(k_folds), len(all_pairs)))
    splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
    splits = list(splitter.split(all_pairs))

    report: Dict[str, object] = {
        "pair_count": len(all_pairs),
        "k_folds": k,
        "detectors": {},
    }

    for detector_name in detectors:
        detector = _build_detector(detector_name, dbnet_checkpoint=dbnet_checkpoint)
        fold_rows: List[Dict[str, object]] = []

        for fold_idx, (_, test_idx) in enumerate(splits, start=1):
            fold_pairs = [all_pairs[i] for i in test_idx]
            metrics = _evaluate_detection_subset(detector, fold_pairs, iou_threshold=iou_threshold)
            row = {
                "fold": fold_idx,
                "test_size": len(fold_pairs),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "hmean": metrics["hmean"],
                "errors": metrics["errors"],
                "first_error": metrics["first_error"],
            }
            fold_rows.append(row)

        p_vals = [float(r["precision"]) for r in fold_rows]
        r_vals = [float(r["recall"]) for r in fold_rows]
        h_vals = [float(r["hmean"]) for r in fold_rows]

        report["detectors"][detector_name] = {
            "mean_precision": _safe_float(_mean(p_vals)),
            "std_precision": _safe_float(_std(p_vals)),
            "mean_recall": _safe_float(_mean(r_vals)),
            "std_recall": _safe_float(_std(r_vals)),
            "mean_hmean": _safe_float(_mean(h_vals)),
            "std_hmean": _safe_float(_std(h_vals)),
            "folds": fold_rows,
        }

    return report


def run_dashboard_benchmark(
    engines: Sequence[str],
    detectors: Sequence[str],
    k_folds: int,
    word_limit: int,
    doc_limit: int,
    detection_limit: int,
    seed: int,
    iou_threshold: float,
    trocr_model_id: str,
    dbnet_checkpoint: Optional[str],
    root_dir: Path,
) -> Dict[str, object]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    word_report = run_word_kfold(
        engines=engines,
        k_folds=k_folds,
        sample_limit=word_limit,
        seed=seed,
        labels_csv=root_dir / "datas" / "archivelabelled" / "labels.csv",
        images_dir=root_dir / "datas" / "archivelabelled" / "crops",
        trocr_model_id=trocr_model_id,
    )

    document_report = run_document_health(
        engines=engines,
        image_limit=doc_limit,
        doc_dirs=[root_dir / "datas" / "archive" / "train", root_dir / "datas" / "archive" / "test"],
        trocr_model_id=trocr_model_id,
    )

    detection_report = run_detection_kfold(
        detectors=detectors,
        k_folds=k_folds,
        pair_limit=detection_limit,
        seed=seed,
        iou_threshold=iou_threshold,
        train_dir=root_dir / "datas" / "archive" / "train",
        test_dir=root_dir / "datas" / "archive" / "test",
        dbnet_checkpoint=dbnet_checkpoint,
    )

    report: Dict[str, object] = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "engines": list(engines),
            "detectors": list(detectors),
            "k_folds": int(k_folds),
            "word_limit": int(word_limit),
            "doc_limit": int(doc_limit),
            "detection_limit": int(detection_limit),
            "seed": int(seed),
            "iou_threshold": float(iou_threshold),
            "trocr_model_id": trocr_model_id,
            "dbnet_checkpoint": dbnet_checkpoint or "",
        },
        "word_kfold": word_report,
        "document_level": document_report,
        "detection_kfold": detection_report,
    }

    out_dir = root_dir / "results" / "dashboard_benchmark" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report
