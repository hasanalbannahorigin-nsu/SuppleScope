from __future__ import annotations

import logging
from pathlib import Path

from app.config import get_settings
from app.services.ocr.base import OcrOutput, OcrProvider
from app.services.ocr.google_vision_provider import GoogleVisionOcrProvider
from app.services.ocr.tesseract_provider import TesseractOcrProvider

logger = logging.getLogger(__name__)

PROVIDERS: dict[str, type[OcrProvider]] = {
    "google_vision": GoogleVisionOcrProvider,
    "tesseract": TesseractOcrProvider,
}


def get_provider(name: str) -> OcrProvider:
    cls = PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown OCR provider: {name}")
    return cls()


def run_ocr_on_image(image_path: Path, *, compare: bool = True) -> list[OcrOutput]:
    settings = get_settings()
    results: list[OcrOutput] = []
    failures: list[str] = []
    ordered = [settings.ocr_primary_provider, settings.ocr_secondary_provider]
    seen: set[str] = set()
    for name in ordered:
        if name in seen:
            continue
        seen.add(name)
        provider = get_provider(name)
        reason = provider.unavailable_reason()
        if reason:
            logger.info("OCR provider %s not available: %s", name, reason)
            failures.append(f"{name}: {reason}")
            continue
        try:
            result = provider.recognize(image_path)
            results.append(result)
            if not compare:
                break
        except Exception as exc:
            logger.exception("OCR provider %s failed", name)
            failures.append(f"{name}: {exc}")
    if not results:
        detail = " ".join(failures) or "No OCR providers are configured."
        raise RuntimeError(f"No OCR provider succeeded. {detail}")
    return results
