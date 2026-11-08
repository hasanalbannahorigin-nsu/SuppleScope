from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from app.services.ocr import google_vision_provider, tesseract_provider
from app.services.ocr.google_vision_provider import GoogleVisionOcrProvider
from app.services.ocr.tesseract_provider import TesseractOcrProvider
from app.services.ocr.preprocess import preprocess_for_ocr


def test_tesseract_adapter_preserves_bangla_unicode_and_confidence(tmp_path, monkeypatch):
    sample_image = tmp_path / "sample_bangla_ad.png"
    Image.new("RGB", (320, 100), color="white").save(sample_image)
    data = {
        "text": ["ইমিউনিটি", "Boost"],
        "conf": ["90", "80"],
        "block_num": [1, 1],
        "line_num": [1, 1],
        "left": [10, 160],
        "top": [20, 20],
        "width": [130, 80],
        "height": [30, 30],
    }
    monkeypatch.setattr(tesseract_provider, "preprocess_for_ocr", lambda _: np.zeros((100, 320, 3), dtype=np.uint8))
    monkeypatch.setattr(tesseract_provider.TesseractOcrProvider, "unavailable_reason", lambda _: None)
    monkeypatch.setattr(
        "pytesseract.image_to_data",
        lambda *args, **kwargs: data,
    )

    output = TesseractOcrProvider().recognize(Path(sample_image))

    assert output.raw_text == "ইমিউনিটি Boost"
    assert output.cleaned_text == "ইমিউনিটি Boost"
    assert output.average_confidence == 0.85
    assert output.bounding_boxes[0]["text"] == "ইমিউনিটি"
    assert output.language_hints == ["bn", "en"]


def test_sample_bangla_ad_runs_real_ocr_when_tesseract_is_configured():
    import pytest

    sample_dir = Path(__file__).parent / "fixtures"
    sample_image = sample_dir / "bangla_ad.png"
    if not TesseractOcrProvider().available():
        pytest.skip("Tesseract with ben and eng language data is not installed")

    from app.services.ocr.metrics import evaluate_transcription

    reference = (sample_dir / "bangla_ad.txt").read_text(encoding="utf-8")
    output = TesseractOcrProvider().recognize(sample_image)
    quality = evaluate_transcription(reference, output.cleaned_text)

    assert output.cleaned_text
    assert 0 <= quality["cer"]
    assert 0 <= quality["wer"]
    print(f"Sample Bangla OCR quality: CER={quality['cer']:.4f}, WER={quality['wer']:.4f}")


def test_tesseract_reports_missing_bengali_language_data(monkeypatch):
    import pytesseract

    monkeypatch.setattr(
        tesseract_provider,
        "get_settings",
        lambda: SimpleNamespace(tesseract_cmd="", tessdata_prefix=""),
    )
    monkeypatch.setattr(pytesseract, "get_tesseract_version", lambda: "5.0")
    monkeypatch.setattr(pytesseract, "get_languages", lambda: ["eng"])

    reason = TesseractOcrProvider().unavailable_reason()

    assert reason is not None
    assert "Bengali" in reason
    assert "ben" in reason


def test_google_vision_reports_missing_credentials(monkeypatch):
    monkeypatch.setattr(
        google_vision_provider,
        "get_settings",
        lambda: SimpleNamespace(google_application_credentials=""),
    )

    reason = GoogleVisionOcrProvider().unavailable_reason()

    assert reason is not None
    assert "GOOGLE_APPLICATION_CREDENTIALS" in reason


def test_image_preprocessing_returns_valid_bgr_image(tmp_path):
    sample_image = tmp_path / "preprocess.png"
    image = Image.new("RGB", (320, 100), color="white")
    image.save(sample_image)

    processed = preprocess_for_ocr(sample_image)

    assert processed.shape == (100, 320, 3)
    assert np.isfinite(processed).all()
