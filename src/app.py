import gc
import os
import sys
import base64
from io import BytesIO
from pathlib import Path

from flask import Flask, request, jsonify, render_template
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

TRAIFIC_APP_DIR = ROOT / "TraificNPR" / "application"
if str(TRAIFIC_APP_DIR) not in sys.path:
    sys.path.append(str(TRAIFIC_APP_DIR))

try:
    from model_loader import load_models as load_traific_models
    from image_processing import process_frame as process_traific_frame
    TRAIFIC_AVAILABLE = True
    TRAIFIC_ERROR = None
except ImportError as e:
    TRAIFIC_AVAILABLE = False
    TRAIFIC_ERROR = str(e)

_TRAIFIC_MODELS = None
def get_traific_models():
    global _TRAIFIC_MODELS
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

# ==============================================================================
# FLASK APPLICATION EXPOSURE
# ==============================================================================
app = Flask(__name__)

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

    engine_choice = request.form.get('engine', 'malla')
    
    try:
        pil_image = Image.open(file.stream).convert('RGB')
        
        if engine_choice == 'traific':
            result = extract_traific_details(pil_image)
        else:
            result = extract_plate_details(pil_image)

        if result['status'] != 'ok':
            return jsonify({"error": result['message']}), 400
            
        # Convert the debug image to base64 for the frontend
        debug_pil = result.get('debug_img')
        debug_b64 = None
        if debug_pil:
            buffered = BytesIO()
            debug_pil.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            debug_b64 = f"data:image/jpeg;base64,{img_str}"
        
        response_data = {
            "status": "ok",
            "plate_text": result['plate_text'],
            "digits_ascii": result['digits_ascii'],
            "avg_conf": round(result['avg_conf'], 4),
            "debug_img": debug_b64
        }
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
    finally:
        # Free memory aggressively after each request
        if engine_choice == 'traific':
            unload_traific_models()
        else:
            unload_model()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    # Make sure templates folder exists relative to app.py
    template_dir = Path(__file__).parent / 'templates'
    template_dir.mkdir(exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
