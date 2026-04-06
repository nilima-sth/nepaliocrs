from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from ocr_engines import DBNetDetector, PaddleTextDetector, TesseractTextDetector


BBox = Tuple[int, int, int, int]


@dataclass
class DetectionRow:
    detector: str
    image_path: str
    gt_count: int
    pred_count: int
    tp: int
    fp: int
    fn: int


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def parse_xml_boxes(xml_path: Path) -> List[BBox]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes: List[BBox] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name != "text":
            continue

        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        try:
            xmin = int(float(bnd.findtext("xmin", "0")))
            ymin = int(float(bnd.findtext("ymin", "0")))
            xmax = int(float(bnd.findtext("xmax", "0")))
            ymax = int(float(bnd.findtext("ymax", "0")))
        except Exception:
            continue

        if xmax > xmin and ymax > ymin:
            boxes.append((xmin, ymin, xmax, ymax))

    return boxes


def match_boxes(gt_boxes: List[BBox], pred_boxes: List[BBox], threshold: float) -> Tuple[int, int, int]:
    matched_gt = set()
    matched_pred = set()

    candidates: List[Tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            score = iou(gt, pred)
            if score >= threshold:
                candidates.append((score, gi, pi))

    candidates.sort(reverse=True, key=lambda x: x[0])

    tp = 0
    for _, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        tp += 1

    fp = max(0, len(pred_boxes) - tp)
    fn = max(0, len(gt_boxes) - tp)
    return tp, fp, fn


def evaluate_detector(detector_name: str, pairs: List[Tuple[Path, Path]], iou_threshold: float, dbnet_checkpoint: Optional[str]) -> Tuple[Dict[str, float], List[DetectionRow]]:
    if detector_name == "paddle":
        detector = PaddleTextDetector(lang="hi", use_gpu=False)
    elif detector_name == "tesseract":
        detector = TesseractTextDetector(lang_chain="nep+hin+eng")
    elif detector_name == "dbnet":
        detector = DBNetDetector(checkpoint_path=dbnet_checkpoint, use_gpu=False)
    else:
        raise ValueError(f"Unsupported detector: {detector_name}")

    rows: List[DetectionRow] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    ok_images = 0
    errors = 0
    first_error = ""

    for img_path, xml_path in pairs:
        gt_boxes = parse_xml_boxes(xml_path)

        try:
            image = Image.open(img_path).convert("RGB")
            pred_boxes = detector.detect_boxes(image)
            tp, fp, fn = match_boxes(gt_boxes, pred_boxes, threshold=iou_threshold)
            ok_images += 1
        except Exception:
            pred_boxes = []
            tp, fp, fn = 0, 0, len(gt_boxes)
            errors += 1
            if not first_error:
                try:
                    raise
                except Exception as err:
                    first_error = str(err)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        rows.append(
            DetectionRow(
                detector=detector_name,
                image_path=str(img_path),
                gt_count=len(gt_boxes),
                pred_count=len(pred_boxes),
                tp=tp,
                fp=fp,
                fn=fn,
            )
        )

    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    hmean = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

    metrics = {
        "detector": detector_name,
        "images": len(pairs),
        "ok_images": ok_images,
        "errors": errors,
        "first_error": first_error,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "hmean": hmean,
    }
    return metrics, rows


def write_rows(path: Path, rows: List[DetectionRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["detector", "image_path", "gt_count", "pred_count", "tp", "fp", "fn"])
        for r in rows:
            writer.writerow([r.detector, r.image_path, r.gt_count, r.pred_count, r.tp, r.fp, r.fn])


def collect_pairs(images_dir: Path, annotations_dir: Path) -> List[Tuple[Path, Path]]:
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    pairs: List[Tuple[Path, Path]] = []

    for img in sorted(images_dir.iterdir()):
        if not img.is_file() or img.suffix.lower() not in image_exts:
            continue
        xml = annotations_dir / f"{img.stem}.xml"
        if xml.exists():
            pairs.append((img, xml))

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate text detection on XML-annotated test set.")
    parser.add_argument("--images-dir", default="datas/archive/test")
    parser.add_argument("--annotations-dir", default="datas/archive/test")
    parser.add_argument("--detectors", default="tesseract,paddle,dbnet")
    parser.add_argument("--dbnet-checkpoint", default="")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", default="results/eval_detection")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    annotations_dir = Path(args.annotations_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations dir not found: {annotations_dir}")

    pairs = collect_pairs(images_dir, annotations_dir)
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    if not pairs:
        raise RuntimeError("No image/xml pairs found for detection evaluation.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[Dict[str, float]] = []
    detector_names = [d.strip() for d in args.detectors.split(",") if d.strip()]

    for detector_name in detector_names:
        try:
            metrics, rows = evaluate_detector(
                detector_name=detector_name,
                pairs=pairs,
                iou_threshold=args.iou_threshold,
                dbnet_checkpoint=args.dbnet_checkpoint or None,
            )
        except Exception as err:
            metrics = {
                "detector": detector_name,
                "images": len(pairs),
                "ok_images": 0,
                "errors": len(pairs),
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "hmean": 0.0,
                "error": str(err),
            }
            rows = []

        all_metrics.append(metrics)
        if rows:
            write_rows(out_dir / f"rows_{detector_name}.csv", rows)

    (out_dir / "summary.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    print(f"[DONE] Detection evaluation completed. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
