from __future__ import annotations

import importlib.util
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from document_ocr import proprietary_document_pipeline


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


@dataclass
class EngineResult:
    status: str
    text: str = ""
    conf: float = 0.0
    error: Optional[str] = None
    debug_img: Optional[Image.Image] = None


def normalize_text(value: str) -> str:
    return " ".join((value or "").strip().split())


def levenshtein_distance(a: str, b: str) -> int:
    a = a or ""
    b = b or ""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


class PaddleTextEngine:
    _model = None
    _lock = Lock()

    def __init__(self, lang: str = "hi", use_gpu: bool = False) -> None:
        self.lang = lang
        self.use_gpu = use_gpu

    @classmethod
    def _parse_result(cls, result) -> Tuple[List[str], List[float]]:
        texts: List[str] = []
        confs: List[float] = []

        if result is None:
            return texts, confs

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

    def _load(self):
        if self.__class__._model is not None:
            return self.__class__._model

        with self.__class__._lock:
            if self.__class__._model is not None:
                return self.__class__._model

            from paddleocr import PaddleOCR

            self.__class__._model = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
            )
            return self.__class__._model

    def predict(self, pil_image: Image.Image) -> EngineResult:
        try:
            model = self._load()
            img_arr = np.array(pil_image.convert("RGB"))
            result = model.ocr(img_arr)
            texts, confs = self._parse_result(result)
            text = "\n".join(texts)
            conf = float(sum(confs) / len(confs)) if confs else 0.0
            return EngineResult(status="ok", text=text, conf=conf, debug_img=pil_image)
        except Exception as err:
            try:
                fallback = proprietary_document_pipeline(
                    pil_image,
                    char_predictor=None,
                    warmup_model=None,
                )
                if fallback.get("status") == "ok":
                    return EngineResult(
                        status="ok",
                        text=str(fallback.get("extracted_text", "")),
                        conf=float(fallback.get("avg_conf", 0.0)),
                        debug_img=fallback.get("debug_img") or pil_image,
                    )
            except Exception:
                pass
            return EngineResult(status="error", error=f"Paddle OCR failed: {err}")


class DocumentPipelineEngine:
    def __init__(self, allow_char_fallback: bool = False):
        self.allow_char_fallback = allow_char_fallback

    def predict(self, pil_image: Image.Image) -> EngineResult:
        try:
            if self.allow_char_fallback:
                out = proprietary_document_pipeline(pil_image)
            else:
                out = proprietary_document_pipeline(
                    pil_image,
                    char_predictor=None,
                    warmup_model=None,
                )
            if out.get("status") != "ok":
                return EngineResult(status="error", error=out.get("message", "Unknown document pipeline error"))
            return EngineResult(
                status="ok",
                text=str(out.get("extracted_text", "")),
                conf=float(out.get("avg_conf", 0.0)),
                debug_img=out.get("debug_img"),
            )
        except Exception as err:
            return EngineResult(status="error", error=f"Indic/document pipeline failed: {err}")


class MallaPipelineEngine:
    _bindings: Optional[Tuple[object, object]] = None
    _lock = Lock()

    @classmethod
    def _load_bindings(cls) -> Tuple[object, object]:
        if cls._bindings is not None:
            return cls._bindings

        with cls._lock:
            if cls._bindings is not None:
                return cls._bindings

            # Imported lazily to avoid import cycles when app.py imports ocr_engines.
            from app import _predict_char, get_devanagari_model

            cls._bindings = (_predict_char, get_devanagari_model)
            return cls._bindings

    def predict(self, pil_image: Image.Image) -> EngineResult:
        try:
            char_predictor, warmup_model = self._load_bindings()
            out = proprietary_document_pipeline(
                pil_image,
                char_predictor=char_predictor,
                warmup_model=warmup_model,
            )
            if out.get("status") != "ok":
                return EngineResult(status="error", error=out.get("message", "Unknown Malla pipeline error"))
            return EngineResult(
                status="ok",
                text=str(out.get("extracted_text", "")),
                conf=float(out.get("avg_conf", 0.0)),
                debug_img=out.get("debug_img"),
            )
        except Exception as err:
            return EngineResult(status="error", error=f"Malla pipeline failed: {err}")


class TrOCREngine:
    _bundles: Dict[str, object] = {}
    _lock = Lock()

    def __init__(self, model_id: str = "paudelanil/trocr-devanagari-2") -> None:
        self.model_id = model_id

    @staticmethod
    def _preprocess_image(image: Image.Image) -> Image.Image:
        target_w, target_h = 224, 224
        w, h = image.size
        if w <= 0 or h <= 0:
            return image.resize((target_w, target_h))

        ratio = w / float(h)
        if ratio > 1:
            new_w = target_w
            new_h = max(1, int(target_w / ratio))
        else:
            new_h = target_h
            new_w = max(1, int(target_h * ratio))

        resized = image.resize((new_w, new_h))
        canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
        return canvas

    def _load(self):
        if self.model_id in self.__class__._bundles:
            return self.__class__._bundles[self.model_id]

        with self.__class__._lock:
            if self.model_id in self.__class__._bundles:
                return self.__class__._bundles[self.model_id]

            import torch
            from transformers import (
                AutoImageProcessor,
                AutoTokenizer,
                VisionEncoderDecoderModel,
            )
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()

            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = VisionEncoderDecoderModel.from_pretrained(self.model_id)
            image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            bundle = {
                "image_processor": image_processor,
                "tokenizer": tokenizer,
                "model": model,
                "device": device,
            }
            self.__class__._bundles[self.model_id] = bundle
            return bundle

    def predict(self, pil_image: Image.Image) -> EngineResult:
        try:
            import torch

            bundle = self._load()
            model = bundle["model"]
            image_processor = bundle["image_processor"]
            tokenizer = bundle["tokenizer"]
            device = bundle["device"]

            image = self._preprocess_image(pil_image.convert("RGB"))
            pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)

            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_new_tokens=128)

            text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return EngineResult(status="ok", text=str(text).strip(), conf=0.0, debug_img=pil_image)
        except Exception as err:
            return EngineResult(status="error", error=f"TrOCR inference failed: {err}")


class PaddleTextDetector:
    _detector = None
    _lock = Lock()

    def __init__(self, lang: str = "hi", use_gpu: bool = False) -> None:
        self.lang = lang
        self.use_gpu = use_gpu

    @staticmethod
    def _poly_to_bbox(poly: Sequence[Sequence[float]]) -> Optional[Tuple[int, int, int, int]]:
        try:
            arr = np.asarray(poly, dtype=np.float32)
            if arr.size == 0:
                return None
            xs = arr[:, 0]
            ys = arr[:, 1]
            return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        except Exception:
            return None

    def _load(self):
        if self.__class__._detector is not None:
            return self.__class__._detector

        with self.__class__._lock:
            if self.__class__._detector is not None:
                return self.__class__._detector

            from paddleocr import PaddleOCR

            self.__class__._detector = PaddleOCR(
                use_angle_cls=False,
                lang=self.lang,
            )
            return self.__class__._detector

    def detect_boxes(self, pil_image: Image.Image) -> List[Tuple[int, int, int, int]]:
        model = self._load()
        img_arr = np.array(pil_image.convert("RGB"))
        raw = model.ocr(img_arr)

        boxes: List[Tuple[int, int, int, int]] = []
        if isinstance(raw, list):
            for page in raw:
                if isinstance(page, list):
                    for item in page:
                        poly = item[0] if isinstance(item, (list, tuple)) and item else item
                        bbox = self._poly_to_bbox(poly)
                        if bbox is not None:
                            boxes.append(bbox)

        return boxes


class TesseractTextDetector:
    def __init__(self, lang_chain: str = "nep+hin+eng") -> None:
        self.lang_chain = lang_chain

    def detect_boxes(self, pil_image: Image.Image) -> List[Tuple[int, int, int, int]]:
        import cv2
        import pytesseract
        from pytesseract import Output

        img = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        data = pytesseract.image_to_data(
            gray,
            output_type=Output.DICT,
            config=f"-l {self.lang_chain} --psm 6",
        )

        boxes: List[Tuple[int, int, int, int]] = []
        count = len(data.get("text", []))
        for i in range(count):
            text = str(data["text"][i]).strip()
            if not text:
                continue

            try:
                conf = float(str(data["conf"][i]))
            except Exception:
                conf = -1.0

            if conf < 0:
                continue

            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])

            if w > 0 and h > 0:
                boxes.append((x, y, x + w, y + h))

        return boxes


class DBNetDetector:
    _model = None
    _lock = Lock()

    def __init__(self, checkpoint_path: Optional[str] = None, use_gpu: bool = False) -> None:
        self.repo_dir = MODELS_DIR / "Nepali-Text-Detection-DBnet"
        default_ckpt = self.repo_dir / "models" / "nepali" / "nepali_td_best.pth"
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else default_ckpt
        self.use_gpu = use_gpu

    @staticmethod
    def _poly_to_bbox(poly: Sequence[Sequence[float]]) -> Optional[Tuple[int, int, int, int]]:
        try:
            arr = np.asarray(poly, dtype=np.float32)
            if arr.size == 0:
                return None
            xs = arr[:, 0]
            ys = arr[:, 1]
            return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        except Exception:
            return None

    def _load(self):
        if self.__class__._model is not None:
            return self.__class__._model

        if not self.repo_dir.exists():
            raise RuntimeError(f"DBNet repo missing: {self.repo_dir}")
        if not self.checkpoint_path.exists():
            raise RuntimeError(
                f"DBNet checkpoint missing: {self.checkpoint_path}. "
                "Train DBNet first or provide --dbnet-checkpoint."
            )

        with self.__class__._lock:
            if self.__class__._model is not None:
                return self.__class__._model

            module_path = self.repo_dir / "inference.py"
            spec = importlib.util.spec_from_file_location("dbnet_inference", module_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Unable to load DBNet inference module: {module_path}")
            module = importlib.util.module_from_spec(spec)

            import sys

            repo_str = str(self.repo_dir)
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)

            spec.loader.exec_module(module)

            class _Args:
                pass

            args = _Args()
            args.model_path = str(self.checkpoint_path)
            args.device = "cuda" if self.use_gpu else "cpu"
            args.thresh = 0.25
            args.box_thresh = 0.5
            args.unclip_ratio = 1.5
            args.prob_thred = 0.5
            args.alpha = 0.6

            self.__class__._model = module.Inference(args)
            return self.__class__._model

    def detect_boxes(self, pil_image: Image.Image) -> List[Tuple[int, int, int, int]]:
        model = self._load()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = Path(tmp.name)
            pil_image.save(temp_path)

        try:
            _, _, box_list, _ = model(img_path=str(temp_path), poly_only=True)
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

        boxes: List[Tuple[int, int, int, int]] = []
        for poly in box_list:
            bbox = self._poly_to_bbox(poly)
            if bbox is not None:
                boxes.append(bbox)

        return boxes


def build_text_engine(name: str, trocr_model_id: str = "paudelanil/trocr-devanagari-2"):
    key = (name or "").strip().lower()
    if key in {"paddle", "paddleocr"}:
        return PaddleTextEngine()
    if key in {"indic", "indic_pipeline", "document", "doc"}:
        return DocumentPipelineEngine(allow_char_fallback=False)
    if key in {"malla", "mallanet", "malla_pipeline"}:
        return MallaPipelineEngine()
    if key in {"trocr", "trocr_devanagari"}:
        return TrOCREngine(model_id=trocr_model_id)
    raise ValueError(f"Unsupported text engine: {name}")


def available_text_engines() -> List[str]:
    return ["paddle", "indic", "malla", "trocr"]
