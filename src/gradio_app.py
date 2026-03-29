import os
import sys
from pathlib import Path
import gradio as gr
from PIL import Image
import pytesseract
import easyocr
import cv2
import logging
logging.disable(logging.WARNING)

import numpy as np
import torch
from torchvision import transforms
from importlib.util import spec_from_file_location, module_from_spec
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

# ---------------------------------------------------------
# ENV / HELPERS
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MALLANET_ROOT = ROOT / "MallaNet"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if MALLANET_ROOT.exists():
    sys.path.insert(0, str(MALLANET_ROOT))

# EasyOCR reader placeholder (Lazy loaded)
reader = None

paddle_ocr = None
def get_paddle():
    global paddle_ocr
    if paddle_ocr is None and PaddleOCR is not None:
        try:
            paddle_ocr = PaddleOCR(use_textline_orientation=True, lang='en')
        except Exception as e:
            print('PaddleOCR load failed:', e)
    return paddle_ocr

# load module helper (for non-package structure)
def load_module_from_path(name, path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find module file: {path}")
    spec = spec_from_file_location(name, str(path))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# -------------
# Load custom model classes
# -------------
english_model_cls = None
devanagari_model_cls = None

try:
    eng_mod = load_module_from_path('english_model', MALLANET_ROOT / 'experiments' / 'english' / 'one_model' / 'english.py')
    english_model_cls = getattr(eng_mod, 'EnhancedBMCNNwHFCs', None)
except Exception as e:
    print('English model module load failed:', e)

try:
    # Changed from HVC to HFC devanagari script since the best_model.pth uses HFCs
    dev_mod = load_module_from_path('devanagari_model', MALLANET_ROOT / 'experiments' / 'devanagari' / 'ensemble' / 'devanagari_ensemble.py')
    devanagari_model_cls = getattr(dev_mod, 'EnhancedBMCNNwHFCs', None)
except Exception as e:
    print('Devanagari model module load failed:', e)

# -------------
# Checkpoint
# -------------
checkpoint_path = ROOT / "MallaNet" / "models" / "best_model.pth"
if checkpoint_path.exists():
    try:
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
    except Exception as e:
        print('Unable to load checkpoint file', e)
        ckpt = None
else:
    ckpt = None

english_model = None
devanagari_model = None

def _clean_state_dict(state_dict):
    """Strip module prefixes and convert any cpu/cuda mismatched keys."""
    if not isinstance(state_dict, dict):
        return state_dict
    clean = {}
    for k, v in state_dict.items():
        new_key = k
        if k.startswith('module.'):
            new_key = k[len('module.'):]
        clean[new_key] = v
    return clean


def _load_model_weights(model, state_dict):
    try:
        r = model.load_state_dict(state_dict)
        if r.missing_keys:
            print('State dict missing keys:', r.missing_keys)
        if r.unexpected_keys:
            print('State dict unexpected keys:', r.unexpected_keys)
        return True
    except RuntimeError as e:
        print('RuntimeError during load_state_dict:', e)
        # try stripping module prefix fallback
        clean_state = _clean_state_dict(state_dict)
        try:
            r = model.load_state_dict(clean_state)
            if r.missing_keys:
                print('Fallback missing keys:', r.missing_keys)
            if r.unexpected_keys:
                print('Fallback unexpected keys:', r.unexpected_keys)
            return True
        except Exception as e2:
            print('Fallback load_state_dict also failed:', type(e2).__name__, e2)
            return False
    except Exception as e:
        print('Error during load_state_dict:', type(e).__name__, e)
        return False


if english_model_cls is not None and ckpt is not None:
    for num_classes in [46, 10]:
        try:
            m = english_model_cls(num_classes=num_classes)
            if _load_model_weights(m, ckpt):
                m.to(DEVICE).eval()
                english_model = m
                print(f'English custom model loaded successfully with num_classes={num_classes}')
                break
        except Exception as e:
            print(f'English model instantiate failed for num_classes={num_classes}:', type(e).__name__, e)
    else:
        print('English custom model could not be loaded')

if devanagari_model_cls is not None and ckpt is not None:
    for num_classes in [46, 10]:
        try:
            d = devanagari_model_cls(num_classes=num_classes)
            if _load_model_weights(d, ckpt):
                d.to(DEVICE).eval()
                devanagari_model = d
                print(f'Devanagari custom model loaded successfully with num_classes={num_classes}')
                break
        except Exception as e:
            print(f'Devanagari model instantiate failed for num_classes={num_classes}:', type(e).__name__, e)
    else:
        print('Devanagari custom model could not be loaded')

ENGLISH_LABELS = {i: str(i) for i in range(10)}
NEPALI_CHARS = [
    'क','ख','ग','घ','ङ','च','छ','ज','झ','ञ','ट','ठ','ड','ढ','ण','त','थ','द','ध','न',
    'प','फ','ब','भ','म','य','र','ल','व','श','ष','स','ह','क्ष','त्र','ज्ञ','ळ','क्ष',
    'श्र','ऋ','ए','ऐ','ओ','औ'  # approx 46 classes sample, adjust to actual class maps
]
DEVANAGARI_LABELS = {i: NEPALI_CHARS[i] if i < len(NEPALI_CHARS) else f"char_{i}" for i in range(46)}

# ---------------------------------------------------------
# MODEL FUNCTIONS
# ---------------------------------------------------------

def tesseract_ocr(image):
    try:
        text = pytesseract.image_to_string(image, lang='nep+eng')
        if not text.strip():
            text = pytesseract.image_to_string(image, lang='eng')
        return text if text.strip() else '(No text detected)'
    except Exception as e:
        return f"Tesseract error: {e}"


def easyocr_extraction(image):
    global reader
    if reader is None:
        reader = easyocr.Reader(['en', 'ne'], gpu=torch.cuda.is_available())
    img_np = np.array(image)
    results = reader.readtext(img_np, detail=0)
    return ' '.join(results) if results else '(No text detected)'


def _custom_model_predict(image, model, label_map, image_size=(32,32)):
    if model is None:
        return '(No model loaded)'
    if image.mode != 'L':
        image = image.convert('L')
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    x = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        pred = int(logits.argmax(dim=1).item())
    label = label_map.get(pred, str(pred))
    return f"Predicted class {pred} -> {label}"


def devanagari_custom_model(image):
    return _custom_model_predict(image, devanagari_model, DEVANAGARI_LABELS, image_size=(64,64))


def paddle_extraction(image):
    p_ocr = get_paddle()
    if p_ocr is None:
        return "(PaddleOCR not installed or failed to load)"
    img_np = np.array(image.convert('RGB')) # ensure paddle gets 3-channel RGB  
    # paddle_ocr.predict expects standard arguments.
    results = p_ocr.ocr(img_np, cls=True)
    if not results or not results[0]:
        return "(No text detected)"
    # results[0] is a list of [box, (text, score)]
    text = " ".join([res[1][0] for res in results[0]])
    return text

def english_custom_model(image):
    # the english custom model uses 28x28 (standard MNIST dimension) or 32x32.
    return _custom_model_predict(image, english_model, ENGLISH_LABELS, image_size=(32,32))

def devanagari_custom_model(image):
    # the devanagari model likely uses 32x32 based on the 1024 tensor mismatch 
    # (128 channels * 8 * 8 = 8192 or similar. 4096 vs 1024 means image dimension needs adjusting)
    return _custom_model_predict(image, devanagari_model, DEVANAGARI_LABELS, image_size=(32,32))
def license_plate_pipeline(image):
    if devanagari_model is None:
        return "(MallaNet Devanagari model not loaded. Check checkpoints.)"
    
    # 1. Convert to OpenCV format & Grayscale
    img_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # 2. Thresholding / Binarization (Detect black text on white/red background)
    # Using Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Find Character Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and wrap in bounding boxes
    rects = []
    h_img, w_img = img_np.shape[:2]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        # Heuristics: Ignore tiny specks and the massive outer border of the plate itself
        if area > 80 and h > 15 and h < h_img * 0.9: 
            rects.append((x, y, w, h))
            
    if not rects:
        return "No characters could be isolated by OpenCV."
        
    # 4. Sort contours top-to-bottom, then left-to-right (for multi-line plates)
    rects.sort(key=lambda b: b[1])  # Sort purely by Y coordinate first
    
    rows = []
    current_row = [rects[0]]
    for r in rects[1:]:
        # If Y is within 25 pixels of current row's average, keep it in the same row
        avg_y = sum(b[1] for b in current_row) / len(current_row)
        if abs(r[1] - avg_y) < 25: 
            current_row.append(r)
        else:
            rows.append(current_row)
            current_row = [r]
    rows.append(current_row)
    
    # Sort each row horizontally (X-axis)
    for row in rows:
        row.sort(key=lambda b: b[0])
        
    # 5. Extract and Predict each character via MallaNet
    results = []
    for i, row in enumerate(rows):
        row_text = []
        for (x, y, w, h) in row:
            # Pad the crop slightly so MallaNet isn't zoomed in too tight
            pad = 4
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(w_img, x+w+pad), min(h_img, y+h+pad)
            char_img = gray[y1:y2, x1:x2]
            
            # Convert crop back to PIL structure exactly like MallaNet expects
            char_pil = Image.fromarray(char_img)
            
            # MallaNet standard inference (32x32 size)
            res_str = _custom_model_predict(char_pil, devanagari_model, DEVANAGARI_LABELS, image_size=(32,32))
            
            # The function returns "Predicted class X -> [CHAR]", we just want the [CHAR]
            predicted_char = res_str.split("->")[-1].strip()
            row_text.append(predicted_char)
            
        results.append("".join(row_text))
        
    return "\n".join(results)
# Dictionary mapping Model Names to their inference functions.
AVAILABLE_MODELS = {
    'License Plate Pipeline (OpenCV + MallaNet)': license_plate_pipeline,
    'Tesseract OCR': tesseract_ocr,
    'EasyOCR': easyocr_extraction,
    'PaddleOCR': paddle_extraction,
    'Devanagari Custom Model': devanagari_custom_model,
    'English Custom Model': english_custom_model,
}

# ---------------------------------------------------------
# INFERENCE LOGIC
# ---------------------------------------------------------

def process_extraction(image, selected_models):
    if image is None:
        return '⚠️ Please upload an image first.'
    if not selected_models:
        return '⚠️ Please select at least one model.'

    outputs = []
    for model_name in selected_models:
        func = AVAILABLE_MODELS.get(model_name)
        if not func:
            outputs.append(f"### {model_name}\n\n(Model not found)\n\n---\n")
            continue
        try:
            text = func(image)
        except Exception as e:
            text = f'Error: {e}'
        outputs.append(f"### {model_name}\n\n{text}\n\n---\n")

    return '\n'.join(outputs)

# ---------------------------------------------------------
# GRADIO UI
# ---------------------------------------------------------

def create_app():
    with gr.Blocks(title='OCR Model Comparator') as demo:
        gr.Markdown('''
            # OCR Model Comparator
            Upload an image containing text, select the models/libraries to use, and compare extracted text in one screen.
        ''')

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type='pil', label='Input Image')
                model_checkboxes = gr.CheckboxGroup(
                    choices=list(AVAILABLE_MODELS.keys()),
                    label='Select Models for Extraction',
                    value=list(AVAILABLE_MODELS.keys())[:2]
                )
                submit_btn = gr.Button('Extract Text', variant='primary')

            with gr.Column(scale=1):
                output_display = gr.Markdown(label='Extraction Results')

        submit_btn.click(fn=process_extraction, inputs=[image_input, model_checkboxes], outputs=[output_display])

    return demo


if __name__ == '__main__':
    app = create_app()
    app.launch(share=False)

