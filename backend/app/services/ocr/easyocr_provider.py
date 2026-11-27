"""EasyOCR (Bengali + English) as the second local OCR engine next to Tesseract.

The Bengali fake-news study among the related works (P6) used EasyOCR for Bangla text in images. It needs
`pip install easyocr` (pulls PyTorch; see requirements-ml.txt) and downloads its bn/en models once.
Set OCR_SECONDARY_PROVIDER=easyocr (the default) and EASYOCR_GPU=true when a CUDA GPU is present.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from app.services.ocr.base import OcrOutput, OcrProvider
from app.services.ocr.normalize import clean_ocr_text

logger = logging.getLogger(__name__)
LANGS = ("bn", "en")


@lru_cache(maxsize=1)
def _reader(gpu: bool):
    import easyocr

    return easyocr.Reader(list(LANGS), gpu=gpu, verbose=False)


def group_lines(results: list) -> tuple[list[str], list[dict], list[float]]:
    """EasyOCR (bbox, text, conf) items -> reading-order lines, boxes and confidences."""
    items = []
    for bbox, text, conf in results:
        text = (text or "").strip()
        if not text:
            continue
        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
        items.append({"text": text, "confidence": float(conf), "x": int(min(xs)), "y": int(min(ys)),
                      "width": int(max(xs) - min(xs)), "height": int(max(ys) - min(ys))})
    items.sort(key=lambda b: (b["y"] + b["height"] / 2, b["x"]))
    lines: list[list[dict]] = []
    for b in items:
        cy = b["y"] + b["height"] / 2
        if lines:
            last = lines[-1]
            ly = sum(x["y"] + x["height"] / 2 for x in last) / len(last)
            lh = max(x["height"] for x in last)
            if abs(cy - ly) <= max(lh, b["height"]) * 0.5:
                last.append(b)
                continue
        lines.append([b])
    text_lines = [" ".join(x["text"] for x in sorted(line, key=lambda x: x["x"])) for line in lines]
    return text_lines, items, [b["confidence"] for b in items]


class EasyOcrProvider(OcrProvider):
    name = "easyocr"

    def unavailable_reason(self) -> str | None:
        try:
            import easyocr  # noqa: F401
        except ImportError:
            return "EasyOCR is not installed (pip install -r requirements-ml.txt)."
        return None

    def available(self) -> bool:
        return self.unavailable_reason() is None

    def recognize(self, image_path: Path) -> OcrOutput:
        reason = self.unavailable_reason()
        if reason:
            raise RuntimeError(reason)
        gpu = os.environ.get("EASYOCR_GPU", "false").lower() in ("1", "true", "yes")
        results = _reader(gpu).readtext(str(image_path), detail=1, paragraph=False)
        lines, boxes, confs = group_lines(results)
        raw = "\n".join(lines)
        return OcrOutput(provider=self.name, raw_text=raw, cleaned_text=clean_ocr_text(raw),
                         average_confidence=(sum(confs) / len(confs)) if confs else None, bounding_boxes=boxes,
                         language_hints=["bn", "en"], extra={"engine": "easyocr", "lang": "+".join(LANGS), "gpu": gpu})
