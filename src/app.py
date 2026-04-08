import gc
import json
import os
import sys
import base64
import re
import secrets
from threading import BoundedSemaphore, Lock
from io import BytesIO
from pathlib import Path

from flask import Flask, request, jsonify, render_template
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageDraw, ImageFont
from torchvision import transforms
from document_ocr import proprietary_document_pipeline
from ocr_engines import TrOCREngine
from dashboard_benchmark import run_dashboard_benchmark

cv2 = None
cv2_import_error = None
try:
    import cv2  # type: ignore[assignment]
except Exception as err:
    cv2 = None
    cv2_import_error = str(err)


def _env_flag(name, default="0"):
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
torch.set_num_threads(max(1, min(2, os.cpu_count() or 2)))
if cv2 is not None:
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = ROOT / "MallaNet" / "models" / "best_model.pth"
KAGGLE_MODEL_PATH = ROOT / "kaggle-model" / "model" / "model.h5"
MAX_UPLOAD_MB = max(1, int(os.getenv("OCR_MAX_UPLOAD_MB", "10")))
MAX_IMAGE_SIDE = max(640, int(os.getenv("OCR_MAX_IMAGE_SIDE", "2200")))
MAX_CONCURRENT_REQUESTS = max(1, int(os.getenv("OCR_MAX_CONCURRENT_REQUESTS", "1")))
UNLOAD_MODELS_EACH_REQUEST = _env_flag("OCR_UNLOAD_MODELS_EACH_REQUEST", "0")
ENABLE_PADDLE_FALLBACK = _env_flag("OCR_ENABLE_PADDLE_FALLBACK", "0")
ENABLE_TRAIFIC_ENGINE = _env_flag("OCR_ENABLE_TRAIFIC_ENGINE", "0")
TROCR_MODEL_ID = os.getenv("TROCR_MODEL_ID", "paudelanil/trocr-devanagari-2")
INFERENCE_SEMAPHORE = BoundedSemaphore(MAX_CONCURRENT_REQUESTS)

TRAIFIC_APP_DIR = ROOT / "TraificNPR" / "application"
if ENABLE_TRAIFIC_ENGINE and str(TRAIFIC_APP_DIR) not in sys.path:
    sys.path.append(str(TRAIFIC_APP_DIR))

if ENABLE_TRAIFIC_ENGINE:
    try:
        from model_loader import load_models as load_traific_models
        from image_processing import process_frame as process_traific_frame
        TRAIFIC_AVAILABLE = True
        TRAIFIC_ERROR = None
    except ImportError as e:
        TRAIFIC_AVAILABLE = False
        TRAIFIC_ERROR = str(e)
else:
    TRAIFIC_AVAILABLE = False
    TRAIFIC_ERROR = "Traific engine disabled via OCR_ENABLE_TRAIFIC_ENGINE=0"

_TRAIFIC_MODELS = None
_TRAIFIC_MODEL_LOCK = Lock()
_BENCHMARK_LOCK = Lock()


def get_traific_models():
    global _TRAIFIC_MODELS
    if _TRAIFIC_MODELS is not None:
        return _TRAIFIC_MODELS

    with _TRAIFIC_MODEL_LOCK:
        if _TRAIFIC_MODELS is None:
            if not TRAIFIC_AVAILABLE:
                raise RuntimeError(f"TraificNPR failed to import: {TRAIFIC_ERROR}")
            _TRAIFIC_MODELS = load_traific_models()
    return _TRAIFIC_MODELS

def unload_traific_models():
    global _TRAIFIC_MODELS
    if _TRAIFIC_MODELS is not None:
        _TRAIFIC_MODELS = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_KAGGLE_MODEL = None
_KAGGLE_MODEL_ERROR = None
_KAGGLE_MODEL_LOCK = Lock()


def _load_kaggle_model_legacy_h5(model_path, tf):
    import h5py  # Lazy import so base startup remains light.

    with h5py.File(str(model_path), "r") as h5_file:
        raw_cfg = h5_file.attrs.get("model_config")

    if raw_cfg is None:
        raise RuntimeError("Legacy load failed: model_config missing in .h5 file.")

    if isinstance(raw_cfg, bytes):
        cfg_text = raw_cfg.decode("utf-8")
    elif hasattr(raw_cfg, "decode"):
        cfg_text = raw_cfg.decode("utf-8")
    else:
        cfg_text = str(raw_cfg)

    parsed = json.loads(cfg_text)
    config_root = parsed.get("config", {})
    if isinstance(config_root, dict):
        layer_specs = config_root.get("layers", [])
        model_name = config_root.get("name", "sequential")
    else:
        layer_specs = config_root
        model_name = "sequential"

    model = tf.keras.Sequential(name=model_name)
    has_input = False

    for spec in layer_specs:
        class_name = spec.get("class_name")
        layer_cfg = dict(spec.get("config", {}))
        trainable = bool(layer_cfg.pop("trainable", True))
        batch_shape = layer_cfg.pop("batch_input_shape", None)

        if batch_shape and not has_input:
            model.add(tf.keras.layers.Input(shape=tuple(batch_shape[1:])))
            has_input = True

        if class_name == "Conv2D":
            layer = tf.keras.layers.Conv2D(**layer_cfg)
        elif class_name == "MaxPooling2D":
            layer = tf.keras.layers.MaxPooling2D(**layer_cfg)
        else:
            raise RuntimeError(f"Legacy load failed: unsupported layer {class_name!r}.")

        layer.trainable = trainable
        model.add(layer)

    model.load_weights(str(model_path))
    return model


def get_kaggle_model():
    global _KAGGLE_MODEL, _KAGGLE_MODEL_ERROR
    if _KAGGLE_MODEL is not None:
        return _KAGGLE_MODEL
    if _KAGGLE_MODEL_ERROR is not None:
        return None

    with _KAGGLE_MODEL_LOCK:
        if _KAGGLE_MODEL is not None:
            return _KAGGLE_MODEL
        if _KAGGLE_MODEL_ERROR is not None:
            return None

        if not KAGGLE_MODEL_PATH.exists():
            _KAGGLE_MODEL_ERROR = f"Kaggle model not found: {KAGGLE_MODEL_PATH}"
            return None

        try:
            # Lazy import so base startup remains fast unless this engine is selected.
            import tensorflow as tf  # type: ignore

            _KAGGLE_MODEL = tf.keras.models.load_model(str(KAGGLE_MODEL_PATH))
            return _KAGGLE_MODEL
        except Exception as err:
            # Fallback for some legacy Sequential .h5 formats that Keras 3 cannot deserialize directly.
            try:
                import tensorflow as tf  # type: ignore

                _KAGGLE_MODEL = _load_kaggle_model_legacy_h5(KAGGLE_MODEL_PATH, tf)
                _KAGGLE_MODEL_ERROR = None
                return _KAGGLE_MODEL
            except Exception as legacy_err:
                _KAGGLE_MODEL_ERROR = f"{err} | Legacy fallback failed: {legacy_err}"
                return None


def unload_kaggle_model():
    global _KAGGLE_MODEL
    if _KAGGLE_MODEL is not None:
        _KAGGLE_MODEL = None
        gc.collect()


# ---------------------------------------------------------
# MallaNet architecture (minimal inference-only copy)
# ---------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = out + identity
        out = self.relu(out)
        return out


class HFCLayer(nn.Module):
    def __init__(self, num_classes, d_b):
        super().__init__()
        self.num_classes = num_classes
        self.V = nn.Parameter(torch.randn(num_classes, d_b))
        self.bn = nn.BatchNorm1d(num_classes * d_b)

    def forward(self, x):
        u_b = x.sum(dim=1)
        u_b_exp = u_b.unsqueeze(1)
        v_exp = self.V.unsqueeze(0)
        t_b = u_b_exp * v_exp
        batch_size = t_b.size(0)
        t_b_flat = t_b.view(batch_size, -1)
        t_b_bn = self.bn(t_b_flat)
        t_b_bn = t_b_bn.view(batch_size, self.num_classes, -1)
        t_b_relu = F.relu(t_b_bn)
        logits = t_b_relu.sum(dim=2)
        return logits


class MergingLayer(nn.Module):
    def __init__(self, num_branches=3):
        super().__init__()
        self.w = nn.Parameter(torch.ones(num_branches) / num_branches)

    def forward(self, inputs):
        weights = F.softmax(self.w, dim=0)
        return sum(weight * logit for weight, logit in zip(weights, inputs))


class BMCNNBase(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            ResidualBlock(1, 128, stride=1, dropout_rate=dropout_rate),
            ResidualBlock(128, 128, stride=1, dropout_rate=dropout_rate),
            ResidualBlock(128, 128, stride=1, dropout_rate=dropout_rate),
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv_block2 = nn.Sequential(
            ResidualBlock(128, 256, stride=1, dropout_rate=dropout_rate),
            ResidualBlock(256, 256, stride=1, dropout_rate=dropout_rate),
            ResidualBlock(256, 256, stride=1, dropout_rate=dropout_rate),
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv_block3 = nn.Sequential(
            ResidualBlock(256, 512, stride=1, dropout_rate=dropout_rate),
            ResidualBlock(512, 512, stride=1, dropout_rate=dropout_rate),
            ResidualBlock(512, 512, stride=1, dropout_rate=dropout_rate),
        )

    def forward(self, x):
        x1 = self.conv_block1(x)
        x = self.pool1(x1)
        x2 = self.conv_block2(x)
        x = self.pool2(x2)
        x3 = self.conv_block3(x)
        return x1, x2, x3


class EnhancedBMCNNwHFCs(BMCNNBase):
    def __init__(self, num_classes=46, dropout_rate=0.0):
        super().__init__(dropout_rate=dropout_rate)
        self.hfc1 = HFCLayer(num_classes, d_b=32 * 32)
        self.hfc2 = HFCLayer(num_classes, d_b=16 * 16)
        self.hfc3 = HFCLayer(num_classes, d_b=8 * 8)
        self.merging = MergingLayer(num_branches=3)

    def forward(self, x):
        x1, x2, x3 = super().forward(x)
        x1_reshaped = x1.view(x1.size(0), x1.size(1), -1)
        x2_reshaped = x2.view(x2.size(0), x2.size(1), -1)
        x3_reshaped = x3.view(x3.size(0), x3.size(1), -1)
        logit1 = self.hfc1(x1_reshaped)
        logit2 = self.hfc2(x2_reshaped)
        logit3 = self.hfc3(x3_reshaped)
        return self.merging((logit1, logit2, logit3))


# ---------------------------------------------------------
# Labels (DHCD: 10 digits + 36 consonants)
# ---------------------------------------------------------
DEVANAGARI_DIGITS = [chr(cp) for cp in range(0x0966, 0x0970)]
DEVANAGARI_CONSONANTS = [
    "\u0915", "\u0916", "\u0917", "\u0918", "\u0919", "\u091A", "\u091B", "\u091C", "\u091D", "\u091E",
    "\u091F", "\u0920", "\u0921", "\u0922", "\u0923", "\u0924", "\u0925", "\u0926", "\u0927", "\u0928",
    "\u092A", "\u092B", "\u092C", "\u092D", "\u092E", "\u092F", "\u0930", "\u0932", "\u0935", "\u0936",
    "\u0937", "\u0938", "\u0939", "\u0915\u094d\u0937", "\u0924\u094d\u0930", "\u091C\u094d\u091E",
]
DEVANAGARI_CLASSES = DEVANAGARI_DIGITS + DEVANAGARI_CONSONANTS
DEVANAGARI_LABELS = {i: DEVANAGARI_CLASSES[i] for i in range(46)}
NEPALI_TO_ASCII_DIGIT = {digit: str(idx) for idx, digit in enumerate(DEVANAGARI_DIGITS)}


_MODEL = None
_MODEL_ERROR = None
_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def _clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def get_devanagari_model():
    global _MODEL, _MODEL_ERROR
    if _MODEL is not None:
        return _MODEL
    if _MODEL_ERROR is not None:
        return None

    if not CHECKPOINT_PATH.exists():
        _MODEL_ERROR = f"Checkpoint not found: {CHECKPOINT_PATH}"
        return None

    try:
        loaded = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        state = loaded["model_state_dict"] if isinstance(loaded, dict) and "model_state_dict" in loaded else loaded
        model = EnhancedBMCNNwHFCs(num_classes=46, dropout_rate=0.0).to(DEVICE)
        try:
            model.load_state_dict(state)
        except Exception:
            model.load_state_dict(_clean_state_dict(state))
        model.eval()
        _MODEL = model
        return _MODEL
    except Exception as err:
        _MODEL_ERROR = str(err)
        return None


def unload_model():
    global _MODEL
    if _MODEL is not None:
        _MODEL = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _predict_char(gray_patch):
    model = get_devanagari_model()
    if model is None:
        return "?", 0.0

    prep = _prepare_char_for_model(gray_patch)
    if prep is None:
        return "?", 0.0

    arr = np.array(prep)
    variants = [prep, Image.fromarray(255 - arr, mode="L")]
    best_label = "?"
    best_conf = 0.0

    with torch.no_grad():
        for variant in variants:
            x = _TRANSFORM(variant).unsqueeze(0).to(DEVICE)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            conf_val = float(conf.item())
            pred_idx = int(pred.item())
            label = DEVANAGARI_LABELS.get(pred_idx, str(pred_idx))
            if conf_val > best_conf:
                best_conf = conf_val
                best_label = label

    if best_conf < 0.30:
        best_label = "?"
    return best_label, best_conf


def _normalize_plate(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(gray)
    return cv2.bilateralFilter(norm, 5, 60, 60)


def _auto_crop_plate(rgb):
    h_img, w_img = rgb.shape[:2]
    if h_img < 80 or w_img < 150:
        return rgb, False

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(blur, 80, 200)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:40]

    best = None
    best_score = -1.0
    full_area = float(h_img * w_img)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        area_ratio = (w * h) / full_area
        if area_ratio < 0.04:
            continue
        aspect = w / float(h)
        if aspect < 1.5 or aspect > 7.0:
            continue
        score = area_ratio + 0.2 * (1.0 - abs((x + w / 2) - (w_img / 2)) / (w_img / 2 + 1e-6))
        if score > best_score:
            best = (x, y, w, h)
            best_score = score

    if best is None:
        return rgb, False

    x, y, w, h = best
    px = max(4, int(0.04 * w))
    py = max(4, int(0.08 * h))
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(w_img, x + w + px)
    y2 = min(h_img, y + h + py)
    return rgb[y1:y2, x1:x2], True


def _binary_candidates(gray):
    norm = _normalize_plate(gray)
    _, otsu_inv = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cands = [otsu_inv, adaptive]
    cleaned = []
    for cand in cands:
        c = cv2.morphologyEx(cand, cv2.MORPH_OPEN, kernel, iterations=1)
        c = cv2.morphologyEx(c, cv2.MORPH_CLOSE, kernel, iterations=1)
        cleaned.append(c)
    return cleaned


def _extract_boxes(binary):
    h_img, w_img = binary.shape[:2]
    area = float(h_img * w_img)
    min_area = max(35, int(0.00045 * area))
    max_area = int(0.25 * area)
    boxes = []
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, n_labels):
        x, y, w, h, a = stats[i]
        if a < min_area or a > max_area:
            continue
        if h < int(0.20 * h_img) or h > int(0.98 * h_img):
            continue
        if w < int(0.012 * w_img) or w > int(0.75 * w_img):
            continue
        aspect = w / float(h)
        if aspect < 0.08 or aspect > 2.2:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def _select_best_boxes(gray):
    best_boxes = []
    best_score = -1.0
    width = gray.shape[1]
    for binary in _binary_candidates(gray):
        boxes = _extract_boxes(binary)
        if not boxes:
            continue
        heights = np.array([b[3] for b in boxes], dtype=np.float32)
        n = len(boxes)
        count_score = 1.0 if 4 <= n <= 16 else max(0.0, 1.0 - abs(n - 9) / 9.0)
        height_score = max(0.0, 1.0 - float(np.std(heights)) / (float(np.mean(heights)) + 1e-6))
        coverage = min(1.0, float(np.sum([b[2] for b in boxes])) / (width + 1e-6))
        score = 0.5 * count_score + 0.3 * height_score + 0.2 * coverage
        if score > best_score:
            best_boxes = boxes
            best_score = score
    return best_boxes


def _group_rows(boxes):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[1] + b[3] / 2.0)
    median_h = float(np.median([b[3] for b in boxes]))
    y_thresh = max(10.0, 0.45 * median_h)
    rows = []
    for box in boxes:
        cy = box[1] + box[3] / 2.0
        assigned = False
        for row in rows:
            row_cy = float(np.mean([r[1] + r[3] / 2.0 for r in row]))
            if abs(cy - row_cy) <= y_thresh:
                row.append(box)
                assigned = True
                break
        if not assigned:
            rows.append([box])
    rows.sort(key=lambda row: float(np.mean([r[1] for r in row])))
    for row in rows:
        row.sort(key=lambda b: b[0])
    return rows


def _prepare_char_for_model(gray_patch):
    if gray_patch is None or gray_patch.size == 0:
        return None
    blur = cv2.GaussianBlur(gray_patch, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(bw) < 127:
        bw = cv2.bitwise_not(bw)
    fg = cv2.bitwise_not(bw)
    pts = cv2.findNonZero(fg)
    if pts is None:
        return None
    x, y, w, h = cv2.boundingRect(pts)
    char = bw[y:y + h, x:x + w]
    side = max(w, h) + 8
    canvas = np.full((side, side), 255, dtype=np.uint8)
    xo = (side - w) // 2
    yo = (side - h) // 2
    canvas[yo:yo + h, xo:xo + w] = char
    resized = cv2.resize(canvas, (32, 32), interpolation=cv2.INTER_AREA)
    return Image.fromarray(resized, mode="L")


def _ascii_digits_only(text):
    out = []
    for ch in text:
        if ch in NEPALI_TO_ASCII_DIGIT:
            out.append(NEPALI_TO_ASCII_DIGIT[ch])
        elif ch.isdigit():
            out.append(ch)
    return "".join(out)


def extract_plate_details(pil_image):
    if cv2 is None:
        return {"status": "error", "message": f"OpenCV failed to import: {cv2_import_error}"}

    model = get_devanagari_model()
    if model is None:
        return {"status": "error", "message": f"MallaNet load failed: {_MODEL_ERROR}"}
        
    try:
        traific_models = get_traific_models()
        plate_model, seg_model, _, _, _ = traific_models
    except Exception as e:
        return {"status": "error", "message": f"Could not load YOLO models for segmentation: {str(e)}"}

    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h, w = cv_image.shape[:2]
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
        cv_image = cv2.resize(cv_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = cv_image.shape[:2]

    # 1. YOLO Plate Detection (fallback to full image)
    plate_results = plate_model.predict(cv_image, verbose=False, conf=0.15)
    if not plate_results or not len(plate_results[0].boxes):
        plate_img = cv_image
    else:
        x1, y1, x2, y2 = map(int, plate_results[0].boxes[0].xyxy[0].tolist())
        plate_img = cv_image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

    # 2. YOLO Char Segmentation
    seg_results = seg_model.predict(plate_img, verbose=False, conf=0.15)
    if not seg_results or not len(seg_results[0].boxes):
        return {"status": "error", "message": "No characters detected by YOLO Segmenter."}

    boxes = []
    for box in seg_results[0].boxes:
        bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
        boxes.append((bx1, by1, bx2 - bx1, by2 - by1))

    rows = _group_rows(boxes)
    if not rows:
        return {"status": "error", "message": "Could not group YOLO detected characters."}

    plate_h, plate_w = plate_img.shape[:2]
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    row_texts = []
    confs = []
    
    # Debug image (RGB for rendering)
    debug_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
    
    for row in rows:
        chars = []
        for x, y, bw, bh in row:
            cv2.rectangle(debug_img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            
            pad = max(2, int(0.08 * max(bw, bh)))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(plate_w, x + bw + pad)
            y2 = min(plate_h, y + bh + pad)
            patch = gray[y1:y2, x1:x2]
            ch, conf = _predict_char(patch)
            
            if ch != "?":
                cv2.putText(debug_img, f"{conf:.2f}", (x, max(10, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            chars.append(ch)
            confs.append(conf)
        row_texts.append("".join(chars))

    plate_text = "".join(row_texts)
    return {
        "status": "ok",
        "plate_text": plate_text,
        "digits_ascii": _ascii_digits_only(plate_text),
        "avg_conf": float(np.mean(confs)) if confs else 0.0,
        "debug_img": Image.fromarray(debug_img)
    }

def extract_traific_details(pil_image):
    try:
        models = get_traific_models()
    except Exception as e:
        return {"status": "error", "message": str(e)}

    plate_model, seg_model, recog_model, device, ocr_font = models
    
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    results = process_traific_frame(
        frame=cv_image, 
        frame_number=0, 
        filename_prefix="gradio_upload",
        plate_model=plate_model,
        seg_model=seg_model,
        recog_model=recog_model,
        device=device,
        ocr_font=ocr_font
    )
    
    if not results:
        return {"status": "error", "message": "TraificNPR could not process plates in this image."}
    
    best_res = results[0]
    if 'error' in best_res:
        return {"status": "error", "message": best_res['error']}

    plate_text = best_res.get('final_text', '')
    conf = best_res.get('confidence', 0.0)
    
    debug_pil_img = pil_image.copy()
    if 'deskewed_plate' in best_res and best_res['deskewed_plate']:
        try:
            b64_data = best_res['deskewed_plate']
            if "," in b64_data:
                b64_data = b64_data.split(",")[1]
            img_data = base64.b64decode(b64_data)
            debug_pil_img = Image.open(BytesIO(img_data)).convert("RGB")
        except Exception as e:
            pass

    return {
        "status": "ok",
        "plate_text": plate_text,
        "digits_ascii": _ascii_digits_only(plate_text),
        "avg_conf": conf,
        "debug_img": debug_pil_img
    }


def _kaggle_prepare_input(pil_image, input_shape):
    shape = input_shape[0] if isinstance(input_shape, list) else input_shape
    if not isinstance(shape, tuple) or len(shape) != 4:
        raise ValueError(f"Unsupported model input shape: {shape}")

    _, d1, d2, d3 = shape
    if d1 in (1, 3) and d3 not in (1, 3):
        # Channels-first: (N, C, H, W)
        channels = int(d1)
        height = int(d2)
        width = int(d3)
        channels_first = True
    else:
        # Channels-last: (N, H, W, C)
        height = int(d1)
        width = int(d2)
        channels = int(d3)
        channels_first = False

    mode = "L" if channels == 1 else "RGB"
    resized = pil_image.convert(mode).resize((width, height), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0

    if channels == 1 and arr.ndim == 2:
        arr = arr[..., np.newaxis]

    if channels_first:
        arr = np.transpose(arr, (2, 0, 1))

    return np.expand_dims(arr, axis=0), resized.convert("RGB")


def extract_kaggle_details(pil_image):
    model = get_kaggle_model()
    if model is None:
        return {"status": "error", "message": _KAGGLE_MODEL_ERROR or "Unable to load kaggle model."}

    try:
        x, preview_img = _kaggle_prepare_input(pil_image, model.input_shape)
        pred = np.asarray(model.predict(x, verbose=0))

        if pred.ndim == 2 and pred.shape[1] > 1:
            class_idx = int(np.argmax(pred[0]))
            conf = float(np.max(pred[0]))
            label = f"class_{class_idx}"
            digits_ascii = str(class_idx)
        else:
            score = float(pred.reshape(-1)[0])
            is_handwritten = score >= 0.5
            label = "handwritten" if is_handwritten else "not_handwritten"
            conf = score if is_handwritten else 1.0 - score
            digits_ascii = "1" if is_handwritten else "0"

        w, h = pil_image.size
        segments = [
            {
                "word_index": 0,
                "bbox": [0, 0, int(w), int(h)],
                "predicted_text": label,
                "raw_text": label,
                "confidence": round(float(conf), 4),
                "source": "kaggle_classifier",
                "used_fallback": False,
            }
        ]

        return {
            "status": "ok",
            "plate_text": label,
            "digits_ascii": digits_ascii,
            "avg_conf": conf,
            "debug_img": preview_img,
            "segments": segments,
        }
    except Exception as err:
        return {"status": "error", "message": f"Kaggle inference failed: {err}"}

# ==============================================================================
# FLASK APPLICATION EXPOSURE
# ==============================================================================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@app.errorhandler(413)
def payload_too_large(_err):
    return jsonify({"error": f"Uploaded file is too large. Max allowed size is {MAX_UPLOAD_MB} MB."}), 413


def _pil_to_base64_jpeg(pil_image, quality=85):
    if pil_image is None:
        return None
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG", quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


def _load_pdf_first_page(raw_file_bytes):
    try:
        import pypdfium2 as pdfium  # type: ignore
    except Exception as err:
        raise RuntimeError("PDF upload requires pypdfium2. Install with 'pip install pypdfium2'.") from err

    document = None
    page = None
    try:
        document = pdfium.PdfDocument(raw_file_bytes)
        if len(document) < 1:
            raise RuntimeError("Uploaded PDF has no pages.")
        page = document[0]
        bitmap = page.render(scale=2.5)
        pil_image = bitmap.to_pil().convert("RGB")
        return pil_image
    except Exception as err:
        raise RuntimeError(f"Unable to render first page from PDF: {err}") from err
    finally:
        if page is not None:
            try:
                page.close()
            except Exception:
                pass
        if document is not None:
            try:
                document.close()
            except Exception:
                pass


def _normalize_input_image(file_stream):
    raw_file = file_stream.read()
    if not raw_file:
        raise ValueError("Uploaded file is empty.")

    try:
        if raw_file.startswith(b"%PDF"):
            pil_image = _load_pdf_first_page(raw_file)
        else:
            pil_image = Image.open(BytesIO(raw_file))
        pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
    except Exception as err:
        raise ValueError(f"Unable to read uploaded file as image/PDF: {err}") from err

    width, height = pil_image.size
    max_side = max(width, height)

    if max_side <= MAX_IMAGE_SIDE:
        return pil_image

    scale = MAX_IMAGE_SIDE / float(max_side)
    target_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    return pil_image.resize(target_size, resampling)


def _clean_token_for_compare(token):
    compact = re.sub(r"\s+", "", str(token or ""), flags=re.UNICODE)
    compact = re.sub(r"[^\w]+", "", compact, flags=re.UNICODE)
    return compact.lower().strip()


def _align_segments_with_reference(segments, reference_text):
    if not isinstance(segments, list):
        return [], None

    reference_tokens = []
    if isinstance(reference_text, str) and reference_text.strip():
        reference_tokens = [tok for tok in re.split(r"\s+", reference_text.strip()) if tok]

    aligned = []
    matched = 0
    compared = 0

    for idx, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue

        record = dict(segment)
        predicted = str(record.get("predicted_text") or record.get("text") or "").strip()
        record["predicted_text"] = predicted

        bbox = record.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                record["bbox"] = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            except Exception:
                record["bbox"] = None
        else:
            record["bbox"] = None

        actual = reference_tokens[idx] if idx < len(reference_tokens) else ""
        record["actual_text"] = actual

        if actual:
            compared += 1
            is_match = _clean_token_for_compare(predicted) == _clean_token_for_compare(actual)
            record["is_match"] = bool(is_match)
            if is_match:
                matched += 1
        else:
            record["is_match"] = None

        aligned.append(record)

    summary = {
        "predicted_token_count": len(aligned),
        "reference_token_count": len(reference_tokens),
        "compared_count": compared,
        "matched_count": matched,
        "word_accuracy": (matched / compared) if compared else None,
        "extra_predictions": max(0, len(aligned) - len(reference_tokens)),
        "missed_reference_tokens": max(0, len(reference_tokens) - len(aligned)),
    }
    return aligned, summary


def _render_segment_overlay(pil_image, segments):
    if pil_image is None:
        return None

    canvas = pil_image.convert("RGB").copy()
    if not isinstance(segments, list) or not segments:
        return canvas

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        bbox = segment.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue

        try:
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        except Exception:
            continue

        is_match = segment.get("is_match")
        if is_match is True:
            color = (34, 197, 94)
        elif is_match is False:
            color = (239, 68, 68)
        else:
            color = (37, 99, 235)

        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)

        predicted_text = str(segment.get("predicted_text") or "").strip()
        actual_text = str(segment.get("actual_text") or "").strip()
        confidence = segment.get("confidence")

        label_parts = []
        if predicted_text:
            label_parts.append(predicted_text)
        if isinstance(confidence, (int, float)):
            label_parts.append(f"{int(round(float(confidence) * 100.0))}%")
        if actual_text:
            label_parts.append(f"GT:{actual_text}")

        if not label_parts:
            continue

        label = " | ".join(label_parts)
        if len(label) > 60:
            label = f"{label[:57]}..."

        tx = max(0, x1)
        ty = max(0, y1 - 14)
        try:
            text_bounds = draw.textbbox((tx, ty), label, font=font)
            tw = max(1, int(text_bounds[2] - text_bounds[0]))
            th = max(1, int(text_bounds[3] - text_bounds[1]))
        except Exception:
            tw, th = (max(16, 6 * len(label)), 10)
        draw.rectangle([(tx, ty), (tx + tw + 4, ty + th + 2)], fill=color)
        draw.text((tx + 2, ty + 1), label, fill=(255, 255, 255), font=font)

    return canvas


def _poly_to_bbox(poly):
    try:
        arr = np.asarray(poly, dtype=np.float32)
        if arr.size == 0:
            return None
        xs = arr[:, 0]
        ys = arr[:, 1]
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    except Exception:
        return None


def _extract_paddle_segments(raw_result):
    segments = []
    if not isinstance(raw_result, list):
        return segments

    for page in raw_result:
        if not isinstance(page, list):
            continue
        for item in page:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue

            bbox = _poly_to_bbox(item[0])
            if bbox is None:
                continue

            rec = item[1]
            predicted_text = ""
            confidence = 0.0
            if isinstance(rec, (list, tuple)) and len(rec) >= 1:
                predicted_text = str(rec[0]).strip()
                if len(rec) > 1:
                    try:
                        confidence = float(rec[1])
                    except Exception:
                        confidence = 0.0

            segments.append(
                {
                    "word_index": len(segments),
                    "bbox": bbox,
                    "predicted_text": predicted_text,
                    "raw_text": predicted_text,
                    "confidence": round(float(confidence), 4),
                    "source": "paddleocr",
                    "used_fallback": False,
                }
            )

    return segments


def _maybe_release_resources(engine_choice):
    if not UNLOAD_MODELS_EACH_REQUEST:
        return

    if engine_choice == "kaggle":
        unload_kaggle_model()
    elif engine_choice in {"indic", "malla"}:
        unload_model()
    elif engine_choice == "traific":
        unload_traific_models()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _api_token_is_valid(req):
    expected = os.getenv("OCR_API_TOKEN", "").strip()
    if not expected:
        return False

    supplied = (req.headers.get("X-API-Token") or "").strip()
    if not supplied:
        auth_header = (req.headers.get("Authorization") or "").strip()
        if auth_header.lower().startswith("bearer "):
            supplied = auth_header[7:].strip()

    if not supplied:
        return False

    return secrets.compare_digest(supplied, expected)


def _run_ocr(engine_choice, pil_image):
    engine = (engine_choice or "paddle").strip().lower()

    if engine == 'paddle':
        return extract_paddle_details(pil_image)
    if engine == 'trocr':
        return extract_trocr_details(pil_image)
    if engine == 'indic':
        return extract_indic_details(pil_image)
    if engine == 'malla':
        return extract_malla_details(pil_image)
    if engine == 'kaggle':
        return extract_kaggle_details(pil_image)
    return {"status": "error", "message": "Unknown engine selected. Use 'paddle', 'trocr', 'indic', 'malla', or 'kaggle'."}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/extract', methods=['POST'])
def api_extract():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    engine_choice = request.form.get('engine', 'paddle')
    reference_text = request.form.get('reference_text', '')
    include_segments = request.form.get('include_segments', 'true').lower() in {'1', 'true', 'yes'}
    acquired = INFERENCE_SEMAPHORE.acquire(blocking=False)
    if not acquired:
        return jsonify({"error": "OCR engine is busy. Please retry in a few seconds."}), 429

    try:
        pil_image = _normalize_input_image(file.stream)

        result = _run_ocr(engine_choice, pil_image)

        if result['status'] != 'ok':
            return jsonify({"error": result['message']}), 200

        aligned_segments = []
        comparison_summary = None
        if include_segments:
            aligned_segments, comparison_summary = _align_segments_with_reference(result.get('segments', []), reference_text)

        debug_pil = result.get('debug_img')
        if include_segments and aligned_segments:
            debug_pil = _render_segment_overlay(pil_image, aligned_segments)

        debug_b64 = _pil_to_base64_jpeg(debug_pil, quality=85)

        response_data = {
            "status": "ok",
            "engine": engine_choice,
            "extracted_text": result.get('extracted_text', result.get('plate_text', '')),
            "avg_conf": round(result.get('avg_conf', 0.0), 4),
            "debug_img": debug_b64,
            "segment_count": len(aligned_segments),
        }

        if include_segments:
            response_data["segments"] = aligned_segments
            response_data["comparison_summary"] = comparison_summary

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
    finally:
        if acquired:
            INFERENCE_SEMAPHORE.release()
        _maybe_release_resources(engine_choice)


@app.route('/api/v1/extract', methods=['POST'])
def api_extract_v1():
    if not _api_token_is_valid(request):
        return jsonify({"error": "Unauthorized"}), 401

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    engine_choice = request.form.get('engine', 'paddle')
    if engine_choice not in {'paddle', 'trocr', 'indic', 'kaggle', 'malla'}:
        return jsonify({"error": "Invalid engine. Use 'paddle', 'trocr', 'indic', 'malla', or 'kaggle'."}), 400

    include_debug = request.form.get('include_debug', 'false').lower() in {'1', 'true', 'yes'}
    include_segments = request.form.get('include_segments', 'false').lower() in {'1', 'true', 'yes'}
    reference_text = request.form.get('reference_text', '')
    acquired = INFERENCE_SEMAPHORE.acquire(blocking=False)
    if not acquired:
        return jsonify({"error": "OCR engine is busy. Please retry in a few seconds."}), 429

    try:
        pil_image = _normalize_input_image(file.stream)
        result = _run_ocr(engine_choice, pil_image)

        if result['status'] != 'ok':
            return jsonify({"error": result['message']}), 422

        response_data = {
            "status": "ok",
            "engine": engine_choice,
            "extracted_text": result.get('extracted_text', result.get('plate_text', '')),
            "avg_conf": round(result.get('avg_conf', 0.0), 4),
        }

        aligned_segments = []
        comparison_summary = None
        if include_segments:
            aligned_segments, comparison_summary = _align_segments_with_reference(result.get('segments', []), reference_text)
            response_data['segments'] = aligned_segments
            response_data['comparison_summary'] = comparison_summary

        if include_debug:
            debug_pil = result.get('debug_img')
            if include_segments and aligned_segments:
                debug_pil = _render_segment_overlay(pil_image, aligned_segments)
            if debug_pil is not None:
                response_data['debug_img'] = _pil_to_base64_jpeg(debug_pil, quality=85)

        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
    finally:
        if acquired:
            INFERENCE_SEMAPHORE.release()
        _maybe_release_resources(engine_choice)


def _parse_int(value, default_value, min_value, max_value):
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default_value)
    return max(min_value, min(max_value, parsed))


def _parse_float(value, default_value, min_value, max_value):
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default_value)
    return max(min_value, min(max_value, parsed))


def _parse_csv_list(value, default_csv):
    raw = value if isinstance(value, str) and value.strip() else default_csv
    return [item.strip().lower() for item in raw.split(',') if item.strip()]


@app.route('/api/benchmark', methods=['POST'])
def api_benchmark():
    payload = request.get_json(silent=True) or request.form.to_dict() or {}

    engines = _parse_csv_list(payload.get('engines', ''), 'paddle,trocr,indic,malla')
    detectors = _parse_csv_list(payload.get('detectors', ''), 'tesseract,paddle,dbnet')

    allowed_engines = {'paddle', 'trocr', 'indic', 'malla'}
    allowed_detectors = {'tesseract', 'paddle', 'dbnet'}

    engines = [e for e in engines if e in allowed_engines]
    detectors = [d for d in detectors if d in allowed_detectors]

    if not engines:
        engines = ['paddle', 'trocr', 'indic', 'malla']
    if not detectors:
        detectors = ['tesseract', 'paddle', 'dbnet']

    k_folds = _parse_int(payload.get('k_folds', 5), 5, 2, 10)
    word_limit = _parse_int(payload.get('word_limit', 120), 120, 10, 5000)
    doc_limit = _parse_int(payload.get('doc_limit', 60), 60, 10, 2000)
    detection_limit = _parse_int(payload.get('detection_limit', 60), 60, 10, 2000)
    seed = _parse_int(payload.get('seed', 42), 42, 1, 10_000_000)
    iou_threshold = _parse_float(payload.get('iou_threshold', 0.5), 0.5, 0.1, 0.95)
    dbnet_checkpoint = (payload.get('dbnet_checkpoint') or '').strip() or None

    if not _BENCHMARK_LOCK.acquire(blocking=False):
        return jsonify({'error': 'A benchmark run is already in progress. Please wait for it to finish.'}), 429

    try:
        report = run_dashboard_benchmark(
            engines=engines,
            detectors=detectors,
            k_folds=k_folds,
            word_limit=word_limit,
            doc_limit=doc_limit,
            detection_limit=detection_limit,
            seed=seed,
            iou_threshold=iou_threshold,
            trocr_model_id=TROCR_MODEL_ID,
            dbnet_checkpoint=dbnet_checkpoint,
            root_dir=ROOT,
        )
        return jsonify({'status': 'ok', 'report': report})
    except Exception as err:
        return jsonify({'error': f'Benchmark failed: {err}'}), 500
    finally:
        _BENCHMARK_LOCK.release()


_PADDLE_OCR = None
_PADDLE_OCR_LOCK = Lock()
_TROCR_ENGINE = None
_TROCR_ENGINE_LOCK = Lock()


def get_trocr_engine():
    global _TROCR_ENGINE
    if _TROCR_ENGINE is not None:
        return _TROCR_ENGINE

    with _TROCR_ENGINE_LOCK:
        if _TROCR_ENGINE is not None:
            return _TROCR_ENGINE
        _TROCR_ENGINE = TrOCREngine(model_id=TROCR_MODEL_ID)
        return _TROCR_ENGINE


def extract_trocr_details(pil_image):
    try:
        engine = get_trocr_engine()
        result = engine.predict(pil_image)
        if result.status != 'ok':
            return {'status': 'error', 'message': result.error or 'TrOCR inference failed'}
        w, h = pil_image.size
        trocr_text = str(result.text or '').strip()
        segments = []
        if trocr_text:
            segments.append(
                {
                    'word_index': 0,
                    'bbox': [0, 0, int(w), int(h)],
                    'predicted_text': trocr_text,
                    'raw_text': trocr_text,
                    'confidence': float(result.conf),
                    'source': 'trocr_full_image',
                    'used_fallback': False,
                }
            )
        return {
            'status': 'ok',
            'extracted_text': trocr_text,
            'avg_conf': float(result.conf),
            'debug_img': result.debug_img or pil_image,
            'segments': segments,
        }
    except Exception as err:
        return {'status': 'error', 'message': f'TrOCR inference failed: {err}'}


def extract_indic_details(pil_image):
    try:
        result = proprietary_document_pipeline(
            pil_image,
            char_predictor=None,
            warmup_model=None,
            include_segments=True,
        )
        if result.get('status') == 'ok':
            result['engine'] = 'indic_document_pipeline'
        return result
    except Exception as err:
        return {'status': 'error', 'message': f'Indic pipeline failed: {err}'}


def extract_malla_details(pil_image):
    try:
        result = proprietary_document_pipeline(
            pil_image,
            char_predictor=_predict_char,
            warmup_model=get_devanagari_model,
            include_segments=True,
        )
        if result.get('status') == 'ok':
            result['engine'] = 'malla_document_pipeline'
        return result
    except Exception as err:
        return {'status': 'error', 'message': f'Malla pipeline failed: {err}'}


def get_paddle_ocr():
    global _PADDLE_OCR
    if _PADDLE_OCR is not None:
        return _PADDLE_OCR

    with _PADDLE_OCR_LOCK:
        if _PADDLE_OCR is not None:
            return _PADDLE_OCR
        try:
            from paddleocr import PaddleOCR
            # PaddleOCR does not expose a 'devanagari' language key; Hindi model covers Devanagari script.
            _PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang='hi', use_gpu=False)
            return _PADDLE_OCR
        except Exception as e:
            print('PaddleOCR load error:', e)
            return None

def _parse_paddle_output(result):
    texts = []
    confs = []

    if result is None:
        return texts, confs

    # Common output: list[page] where page is list[[box, [text, conf]]]
    if isinstance(result, list):
        for page in result:
            if isinstance(page, list):
                for item in page:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        rec = item[1]
                        if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                            text = str(rec[0]).strip()
                            if text:
                                texts.append(text)
                                try:
                                    confs.append(float(rec[1]))
                                except Exception:
                                    confs.append(0.0)
            elif isinstance(page, dict):
                text = str(page.get("rec_text", "")).strip()
                if text:
                    texts.append(text)
                    try:
                        confs.append(float(page.get("rec_score", 0.0)))
                    except Exception:
                        confs.append(0.0)

    return texts, confs


def extract_paddle_details(pil_image):
    ocr = get_paddle_ocr()
    if not ocr:
        return {'status': 'error', 'message': 'PaddleOCR not loaded properly'}

    img_arr = np.array(pil_image.convert("RGB"))

    try:
        result = ocr.ocr(img_arr)
    except Exception as err:
        fallback = proprietary_document_pipeline(
            pil_image,
            char_predictor=None,
            warmup_model=None,
            include_segments=True,
        )
        if fallback.get('status') == 'ok':
            fallback['engine'] = 'paddle_fallback_document_pipeline'
            return fallback
        return {'status': 'error', 'message': f'PaddleOCR inference failed: {err}'}
    
    texts, confs = _parse_paddle_output(result)
    segments = _extract_paddle_segments(result)

    # Fall back to Tesseract+layout salvage path only when explicitly enabled.
    if not texts:
        if not ENABLE_PADDLE_FALLBACK:
            return {
                'status': 'ok',
                'extracted_text': '',
                'avg_conf': 0.0,
                'debug_img': pil_image,
                'segments': segments,
            }

        fallback = proprietary_document_pipeline(
            pil_image,
            char_predictor=_predict_char,
            warmup_model=get_devanagari_model,
            include_segments=True,
        )
        if fallback.get('status') == 'ok':
            fallback['engine'] = 'paddle_fallback_document_pipeline'
        return fallback

    extracted = "\n".join(texts)
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return {
        'status': 'ok',
        'extracted_text': extracted,
        'avg_conf': avg_conf,
        'debug_img': pil_image,
        'segments': segments,
    }


if __name__ == '__main__':
    # Make sure templates folder exists relative to app.py
    template_dir = Path(__file__).parent / 'templates'
    template_dir.mkdir(exist_ok=True)
    app_port = int(os.getenv('PORT', '5000'))
    app_debug = _env_flag('FLASK_DEBUG', '0')
    disable_reloader = _env_flag('OCR_DISABLE_RELOADER', '1')
    app.run(
        host='0.0.0.0',
        port=app_port,
        debug=app_debug,
        use_reloader=(app_debug and not disable_reloader),
        threaded=True,
    )

