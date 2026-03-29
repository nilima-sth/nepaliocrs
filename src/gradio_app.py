import gc
import os
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

cv2 = None
cv2_import_error = None
try:
    import cv2  # type: ignore[assignment]
except Exception as err:
    cv2 = None
    cv2_import_error = str(err)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
torch.set_num_threads(max(1, min(2, os.cpu_count() or 2)))

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = ROOT / "MallaNet" / "models" / "best_model.pth"


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

    rgb = np.array(pil_image.convert("RGB"))
    h, w = rgb.shape[:2]
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    plate_rgb, _ = _auto_crop_plate(rgb)
    gray = cv2.cvtColor(plate_rgb, cv2.COLOR_RGB2GRAY)
    boxes = _select_best_boxes(gray)
    rows = _group_rows(boxes)
    if not rows:
        return {"status": "error", "message": "No characters detected. Use a closer/cropped plate image."}

    plate_h, plate_w = gray.shape[:2]
    row_texts = []
    confs = []
    for row in rows:
        chars = []
        for x, y, bw, bh in row:
            pad = max(2, int(0.08 * max(bw, bh)))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(plate_w, x + bw + pad)
            y2 = min(plate_h, y + bh + pad)
            patch = gray[y1:y2, x1:x2]
            ch, conf = _predict_char(patch)
            chars.append(ch)
            confs.append(conf)
        row_texts.append("".join(chars))

    plate_text = "".join(row_texts)
    return {
        "status": "ok",
        "plate_text": plate_text,
        "digits_ascii": _ascii_digits_only(plate_text),
        "avg_conf": float(np.mean(confs)) if confs else 0.0,
    }


def predict_single(image):
    if image is None:
        return "Please upload an image."
    result = extract_plate_details(image)
    try:
        if result["status"] != "ok":
            return result["message"]
        return (
            f"Plate text: {result['plate_text']}\n"
            f"Digits (ASCII): {result['digits_ascii'] or '(none)'}\n"
            f"Avg confidence: {result['avg_conf']:.3f}"
        )
    finally:
        unload_model()


def _coerce_path(file_obj):
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict) and "name" in file_obj:
        return file_obj["name"]
    if hasattr(file_obj, "name"):
        return file_obj.name
    return str(file_obj)


def predict_batch(files):
    if not files:
        return "Please upload one or more images.", []
    if not isinstance(files, list):
        files = [files]

    rows = []
    success = 0
    for idx, fobj in enumerate(files[:10], start=1):
        path = _coerce_path(fobj)
        name = Path(path).name if path else f"image_{idx}"
        try:
            with Image.open(path) as img:
                result = extract_plate_details(img.convert("RGB"))
        except Exception as err:
            rows.append([name, "-", "-", "0.000", f"Error: {err}"])
            continue

        if result["status"] == "ok":
            success += 1
            rows.append(
                [
                    name,
                    result["plate_text"] if result["plate_text"] else "-",
                    result["digits_ascii"] if result["digits_ascii"] else "-",
                    f"{result['avg_conf']:.3f}",
                    "OK",
                ]
            )
        else:
            rows.append([name, "-", "-", "0.000", result["message"]])

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = f"Processed {len(rows)} image(s). Successful reads: {success}/{len(rows)}."
    unload_model()
    return summary, rows


def create_app():
    with gr.Blocks(title="Lightweight Nepali License Plate OCR") as demo:
        gr.Markdown(
            """
            # Lightweight Nepali License Plate OCR
            This build is optimized to reduce crashes: it only loads the plate OCR model when you click extract.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Single Plate Image")
                single_btn = gr.Button("Extract Single Plate", variant="primary")
                file_input = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                    label="Batch Upload (up to 10 images)",
                )
                batch_btn = gr.Button("Extract Batch Plates", variant="primary")

            with gr.Column(scale=1):
                single_output = gr.Markdown(label="Single Result")
                batch_summary = gr.Markdown()
                batch_table = gr.Dataframe(
                    headers=["Image", "Plate Text", "Digits (ASCII)", "Avg Confidence", "Status"],
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False,
                )

        single_btn.click(fn=predict_single, inputs=[image_input], outputs=[single_output])
        batch_btn.click(fn=predict_batch, inputs=[file_input], outputs=[batch_summary, batch_table])

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False, show_error=True)
