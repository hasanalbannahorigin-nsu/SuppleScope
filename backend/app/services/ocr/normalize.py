from __future__ import annotations

import re
import unicodedata


def normalize_unicode(text: str) -> str:
    # NFC preserves Bangla conjuncts more reliably than NFKC for display text.
    return unicodedata.normalize("NFC", text or "")


def clean_ocr_text(text: str) -> str:
    text = normalize_unicode(text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def reading_order_join(lines: list[str]) -> str:
    return "\n".join(clean_ocr_text(line) for line in lines if clean_ocr_text(line))
