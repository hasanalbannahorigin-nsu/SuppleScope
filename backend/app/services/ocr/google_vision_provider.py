from __future__ import annotations

import logging
import os
from pathlib import Path

from app.config import get_settings
from app.services.ocr.base import OcrOutput, OcrProvider
from app.services.ocr.normalize import clean_ocr_text
from app.services.ocr.preprocess import encode_png, preprocess_for_ocr

logger = logging.getLogger(__name__)


class GoogleVisionOcrProvider(OcrProvider):
    name = "google_vision"

    def unavailable_reason(self) -> str | None:
        settings = get_settings()
        creds = settings.google_application_credentials
        if not creds:
            return (
                "Google Cloud Vision is not configured. Set "
                "GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON file."
            )
        path = Path(creds)
        if not path.is_file():
            return f"Google Vision credentials file not found: {path}"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)
        try:
            from google.cloud import vision  # noqa: F401

            return None
        except Exception as exc:  # pragma: no cover
            return f"Google Cloud Vision SDK is unavailable: {exc}"

    def available(self) -> bool:
        reason = self.unavailable_reason()
        if reason:
            logger.info("Google Vision unavailable: %s", reason)
            return False
        return True

    def recognize(self, image_path: Path) -> OcrOutput:
        from google.cloud import vision

        processed = preprocess_for_ocr(image_path)
        content = encode_png(processed)
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=content)
        image_context = vision.ImageContext(language_hints=["bn", "en"])
        response = client.document_text_detection(image=image, image_context=image_context)
        if response.error.message:
            raise RuntimeError(response.error.message)
        annotation = response.full_text_annotation
        raw = annotation.text if annotation else ""
        boxes = []
        confidences: list[float] = []
        if annotation:
            for page in annotation.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            text = "".join(symbol.text for symbol in word.symbols)
                            conf = float(word.confidence) if word.confidence is not None else None
                            if conf is not None:
                                confidences.append(conf)
                            vertices = word.bounding_box.vertices
                            xs = [v.x for v in vertices if v.x is not None]
                            ys = [v.y for v in vertices if v.y is not None]
                            boxes.append(
                                {
                                    "text": text,
                                    "confidence": conf,
                                    "x": min(xs) if xs else None,
                                    "y": min(ys) if ys else None,
                                    "width": (max(xs) - min(xs)) if xs else None,
                                    "height": (max(ys) - min(ys)) if ys else None,
                                }
                            )
        avg = sum(confidences) / len(confidences) if confidences else None
        return OcrOutput(
            provider=self.name,
            raw_text=raw,
            cleaned_text=clean_ocr_text(raw),
            average_confidence=avg,
            bounding_boxes=boxes,
            language_hints=["bn", "en"],
            extra={"engine": "google_vision"},
        )
