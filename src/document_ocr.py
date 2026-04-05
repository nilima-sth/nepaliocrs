import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image
from pathlib import Path

# Prefer the default Windows install path if present, otherwise use PATH-based resolution.
_DEFAULT_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if _DEFAULT_TESSERACT.exists():
    pytesseract.pytesseract.tesseract_cmd = str(_DEFAULT_TESSERACT)

def proprietary_document_pipeline(pil_image, char_predictor=None, warmup_model=None):
    """
    PROPRIETARY "MALLA-ENSEMBLE" ARCHITECTURE
    1. OpenCV Adaptive Denoising tailored for old/yellowed documents
    2. Tesseract sequence extraction for layout and high-confidence structure
    3. Low-confidence interception -> YOLO/OpenCV bounding box -> MallaNet single character fallback verification
    """
    if warmup_model is not None:
        try:
            warmup_model()  # Heat up optional character model in RAM.
        except Exception:
            # Keep going; base OCR should still run even without optional fallback.
            pass

    try:
        # Convert PIL to BGR OpenCV format
        open_cv_image = np.array(pil_image.convert('RGB'))
        open_cv_image = open_cv_image[:, :, ::-1].copy()
    except Exception as e:
        return {"status": "error", "message": f"Image format error: {e}"}

    # Step 1: Document Pre-processing
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    
    # Blur to remove paper noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive threshold to save faded ink texts from old documents
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    
    # Tesseract expects black text on white background
    tesseract_ready = cv2.bitwise_not(thresh)

    # Prepare return variables
    debug_image = open_cv_image.copy()
    document_lines = {}
    
    # Step 2: Extract layout and base probabilities via Tesseract sequence OCR.
    # Try Nepali+Hindi+English first, then gracefully fall back.
    configs = [r'-l nep+hin+eng --psm 6', r'-l hin+eng --psm 6', r'-l eng --psm 6']
    data = None
    for custom_config in configs:
        try:
            data = pytesseract.image_to_data(tesseract_ready, output_type=Output.DICT, config=custom_config)
            break
        except pytesseract.pytesseract.TesseractNotFoundError:
            return {"status": "error", "message": "Tesseract binary not found! Please check C:\\Program Files\\Tesseract-OCR\\tesseract.exe"}
        except pytesseract.pytesseract.TesseractError:
            continue

    if data is None:
        return {"status": "error", "message": "Tesseract OCR failed for all configured language sets."}
    
    n_boxes = len(data['text'])
    avg_conf_list = []

    # Step 3: Run the Verification Pipeline
    for i in range(n_boxes):
        word_text = data['text'][i].strip()
        try:
            conf = float(str(data['conf'][i]))
        except Exception:
            conf = -1.0
        
        # -1 confidence indicates a structural block (like a paragraph boundary) rather than a word
        if conf > -1 and len(word_text) > 0:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
            # Expand bounding box slightly for safer cropping
            pad = 2
            y1, y2 = max(0, y-pad), min(open_cv_image.shape[0], y+h+pad)
            x1, x2 = max(0, x-pad), min(open_cv_image.shape[1], x+w+pad)
            
            # Optional low-confidence salvage path via external character predictor.
            if conf < 55 and char_predictor is not None:
                # Draw a RED box indicating MallaNet took over this messy word
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.putText(debug_image, str(conf), (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                
                # Extract Grayscale Patch of the failed word 
                word_region_gray = gray[y1:y2, x1:x2]
                word_region_thresh = thresh[y1:y2, x1:x2] # inverted for contour finding
                
                # Cut the word into individual pieces (Vertical Segmentation)
                contours, _ = cv2.findContours(word_region_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                char_boxes = [cv2.boundingRect(c) for c in contours]
                if len(char_boxes) > 0:
                    # Sort completely from left to right to build the word properly
                    char_boxes = sorted(char_boxes, key=lambda b: b[0])
                    
                    verified_characters = []
                    for bx, by, bw, bh in char_boxes:
                        # Filter out tiny noise dots
                        if bw * bh > 15:
                            # Isolate the exact character
                            char_patch = word_region_gray[max(0, by-1):by+bh+1, max(0, bx-1):bx+bw+1]
                            
                            # Run it through the provided fallback character recognizer.
                            char_pred, char_conf = char_predictor(char_patch)
                            
                            if char_pred != "?":
                                verified_characters.append(char_pred)
                                avg_conf_list.append(char_conf * 100) # scale to 100 max
                    
                    # If MallaNet successfully salvaged chars, override the Sequence OCR output completely
                    if verified_characters:
                        word_text = "".join(verified_characters)
                
            else:
                # Draw a GREEN box indicating Sequence OCR handled it fine natively
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                avg_conf_list.append(conf)

            # Store the word dynamically into its correct line hierarchy
            block_num = data['block_num'][i]
            line_num = data['line_num'][i]
            par_num = data['par_num'][i]
            
            line_id = f"{block_num}_{par_num}_{line_num}"
            if line_id not in document_lines:
                document_lines[line_id] = []
            
            document_lines[line_id].append(word_text)

    # Re-assemble the document text formatting
    final_text_blocks = []
    # Using natural python sort will order keys like '1_1_1', '1_1_2'
    for line_id in sorted(document_lines.keys()):
        final_text_blocks.append(" ".join(document_lines[line_id]))
        
    final_compiled_text = "\n".join(final_text_blocks)
    
    total_avg_conf = np.mean(avg_conf_list) if len(avg_conf_list) > 0 else 0.0

    # Ensure debug image is returned as a proper object
    debug_image_rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
    
    return {
        "status": "ok",
        "extracted_text": final_compiled_text,
        "avg_conf": float(total_avg_conf) / 100.0, # Normalizing back to 0-1 for UI parity
        "debug_img": Image.fromarray(debug_image_rgb)
    }