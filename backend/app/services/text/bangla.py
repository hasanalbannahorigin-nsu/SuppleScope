"""Deterministic Bangla text normalisation shared by extraction and source validation."""
from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

BN_DIGITS = "০১২৩৪৫৬৭৮৯"
_TO_ASCII = str.maketrans(BN_DIGITS, "0123456789")

# Unicode variants OCR engines and keyboards produce for the same glyph.
_FIXES = [
    ("ড়", "ড়"),  # ড + nukta -> ড়
    ("ঢ়", "ঢ়"),  # ঢ + nukta -> ঢ়
    ("য়", "য়"),  # য + nukta -> য়
    ("ো", "ো"),  # ে + া -> ো
    ("ৌ", "ৌ"),  # ে + ৗ -> ৌ
    ("‌", ""), ("‍", ""),
]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    for a, b in _FIXES:
        text = text.replace(a, b)
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def to_ascii_digits(text: str) -> str:
    return text.translate(_TO_ASCII)


def match_key(text: str) -> str:
    """Aggressive comparison key: normalised, ASCII digits, lower case, no spaces/punctuation."""
    text = to_ascii_digits(normalize(text)).lower()
    return re.sub(r"[\s\W_]+", "", text)


def numbers_in(text: str) -> list[float]:
    out = []
    for m in re.findall(r"\d[\d,]*(?:\.\d+)?", to_ascii_digits(text or "")):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def span_score(span: str, text: str) -> float:
    """0..100: how well `span` occurs inside `text` (exact substring = 100)."""
    a, b = match_key(span), match_key(text)
    if not a:
        return 0.0
    if a in b:
        return 100.0
    return float(fuzz.partial_ratio(a, b))


def similarity(a: str | None, b: str | None) -> float:
    """0..1 similarity of two short names (order-insensitive)."""
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(match_key_spaced(a), match_key_spaced(b)) / 100.0


def match_key_spaced(text: str) -> str:
    text = to_ascii_digits(normalize(text)).lower()
    return re.sub(r"[^\w\s]+|_", " ", text)
