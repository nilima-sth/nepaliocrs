from PIL import Image
import pytest

import document_ocr


def _mock_tesseract_data(word="नेपाल", conf="87"):
    return {
        "text": [word],
        "conf": [conf],
        "left": [10],
        "top": [12],
        "width": [60],
        "height": [24],
        "block_num": [1],
        "par_num": [1],
        "line_num": [1],
    }


def test_pipeline_returns_segments_when_enabled(monkeypatch):
    monkeypatch.setattr(
        document_ocr.pytesseract,
        "image_to_data",
        lambda *args, **kwargs: _mock_tesseract_data(),
    )

    image = Image.new("RGB", (200, 80), "white")
    result = document_ocr.proprietary_document_pipeline(image, include_segments=True)

    assert result["status"] == "ok"
    assert "segments" in result
    assert len(result["segments"]) == 1

    segment = result["segments"][0]
    assert segment["predicted_text"] == "नेपाल"
    assert segment["source"] in {"tesseract", "malla_fallback"}
    assert segment["confidence"] == pytest.approx(0.87, abs=0.03)


def test_pipeline_omits_segments_when_disabled(monkeypatch):
    monkeypatch.setattr(
        document_ocr.pytesseract,
        "image_to_data",
        lambda *args, **kwargs: _mock_tesseract_data(word="hello", conf="65"),
    )

    image = Image.new("RGB", (180, 64), "white")
    result = document_ocr.proprietary_document_pipeline(image, include_segments=False)

    assert result["status"] == "ok"
    assert "segments" not in result
