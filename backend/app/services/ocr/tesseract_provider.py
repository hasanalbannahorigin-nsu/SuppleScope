from __future__ import annotations

import logging
import os
from pathlib import Path

from app.config import get_settings
from app.services.ocr.base import OcrOutput, OcrProvider
from app.services.ocr.normalize import clean_ocr_text
from app.services.ocr.preprocess import preprocess_for_ocr

logger = logging.getLogger(__name__)


class TesseractOcrProvider(OcrProvider):
    name = "tesseract"

    def unavailable_reason(self) -> str | None:
        try:
            import pytesseract
        except ImportError:
            return "Install the backend dependency pytesseract to use local OCR."

        settings = get_settings()
        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
        if settings.tessdata_prefix:
            os.environ["TESSDATA_PREFIX"] = settings.tessdata_prefix
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:
            return (
                "Tesseract executable is unavailable. Install Tesseract and set "
                f"TESSERACT_CMD if needed. Details: {exc}"
            )
        try:
            languages = set(pytesseract.get_languages())
        except Exception as exc:
            return f"Could not read Tesseract language data. Check TESSDATA_PREFIX. Details: {exc}"
        missing = {"ben", "eng"} - languages
        if missing:
            language_names = {
                "ben": "Bengali",
                "eng": "English",
            }
            missing_names = ", ".join(language_names[lang] for lang in sorted(missing))
            return (
                f"Tesseract language data missing: {missing_names}. Install the "
                "corresponding traineddata files (ben and eng) and configure "
                "TESSDATA_PREFIX if they are outside the default tessdata directory."
            )
        return None

    def available(self) -> bool:
        reason = self.unavailable_reason()
        if reason:
            logger.info("Tesseract unavailable: %s", reason)
            return False
        return True

    def recognize(self, image_path: Path) -> OcrOutput:
        import pytesseract
        from PIL import Image
        import cv2

        settings = get_settings()
        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
        if settings.tessdata_prefix:
            os.environ["TESSDATA_PREFIX"] = settings.tessdata_prefix
        reason = self.unavailable_reason()
        if reason:
            raise RuntimeError(reason)

        processed = preprocess_for_ocr(image_path)
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        # ben = Bengali, eng = English. Do not translate.
        config = "--oem 1 --psm 6"
        data = pytesseract.image_to_data(
            pil_image, lang="ben+eng", config=config, output_type=pytesseract.Output.DICT
        )
        lines: dict[tuple[int, int], list[str]] = {}
        boxes = []
        confidences: list[float] = []
        n = len(data.get("text", []))
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1
            if conf >= 0:
                confidences.append(conf)
            key = (int(data["block_num"][i]), int(data["line_num"][i]))
            lines.setdefault(key, []).append(text)
            boxes.append(
                {
                    "text": text,
                    "confidence": conf if conf >= 0 else None,
                    "x": int(data["left"][i]),
                    "y": int(data["top"][i]),
                    "width": int(data["width"][i]),
                    "height": int(data["height"][i]),
                }
            )
        ordered = [" ".join(words) for _, words in sorted(lines.items())]
        raw = "\n".join(ordered)
        avg = sum(confidences) / len(confidences) / 100 if confidences else None
        return OcrOutput(
            provider=self.name,
            raw_text=raw,
            cleaned_text=clean_ocr_text(raw),
            average_confidence=avg,
            bounding_boxes=boxes,
            language_hints=["bn", "en"],
            extra={"engine": "tesseract", "lang": "ben+eng"},
        )
