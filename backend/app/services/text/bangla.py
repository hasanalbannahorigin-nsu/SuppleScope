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


def letters_key(text: str) -> str:
    """Normalised, ASCII digits, lower case; keeps every letter, vowel sign and digit (match_key drops Bangla
    vowel signs, so 'সেল' and 'সুল' look the same there); everything else becomes a space."""
    text = to_ascii_digits(normalize(text)).lower()
    return " ".join("".join(c if unicodedata.category(c)[0] in "LMN" else " " for c in text).split())


# Linking words and auxiliary verbs an atomic claim may add to a fragment ("স্থায়ী সমাধান" -> "... করবে").
FUNCTION_WORDS = frozenset(letters_key(w) for w in """করবে করে করা করুন হয় হবে হয়ে দেবে দেয় দিবে দিয়ে থাকে রাখে রাখবে
    ও এবং আর বা থেকে জন্য এর এ কে না নয় সাথে সঙ্গে the and or of to is are be will can""".split())


def _content_words(text: str) -> list[str]:
    return [w for w in letters_key(text).split() if len(w) >= 2 and w not in FUNCTION_WORDS]


# Verb endings, longest first: an atomic claim may restate an imperative or another tense of the ad's verb
# ("ওজন কমান" -> "ওজন কমায়", "বাড়াবে" -> "বাড়ায়"). Only words that both end in one of these are compared by stem.
VERB_ENDINGS = tuple(sorted({letters_key(e) for e in "াচ্ছে িয়ে ানো াবে াতে ায় ান াও েছে ছে বে তে ুন েন ে".split()},
                            key=len, reverse=True))


def _verb_stem(word: str) -> str | None:
    for e in VERB_ENDINGS:
        if word.endswith(e) and len(_bare(word[: -len(e)])) >= 2:
            return word[: -len(e)]
    return None


def claim_key(text: str) -> str:
    """Key for spotting a repeated claim: letters and digits only, the last word by its verb stem
    ("ওজন বাড়ান!" and "ওজন বাড়ায়" give the same key; "হাড় মজবুত করে" and "দাঁত মজবুত করে" do not)."""
    words = letters_key(text).split()
    if words and (st := _verb_stem(words[-1])):
        words[-1] = st
    return " ".join(words)


def _word_finder(source: str, threshold: float):
    words = letters_key(source).split()
    hay = "".join(words)
    bare_hay = _bare(hay)
    stems = {st for x in words if (st := _verb_stem(x))}

    def found(w: str) -> bool:
        if w in hay or fuzz.partial_ratio(w, hay) >= threshold:
            return True
        if (st := _verb_stem(w)) and st in stems:  # same verb, other ending
            return True
        # short words differ by one vowel sign after OCR ("দুর" / "দূর"): compare the letters without signs
        return len(w) <= 4 and len(_bare(w)) >= 2 and _bare(w) in bare_hay

    return found


def missing_words(text: str, source: str, threshold: float = 80.0, min_coverage: float = 0.75) -> list[str]:
    """Content words of `text` (2+ characters, not FUNCTION_WORDS) that do not occur in `source`, spaces
    ignored so OCR-joined words still count. Returns them only when fewer than `min_coverage` are found
    (min_coverage=1.0: any missing word)."""
    words = _content_words(text)
    if not words:
        return []
    found = _word_finder(source, threshold)
    miss = [w for w in words if not found(w)]
    return miss if (len(words) - len(miss)) / len(words) < min_coverage else []


def _bare(text: str) -> str:
    return "".join(c for c in text if unicodedata.category(c)[0] != "M")


def match_key_spaced(text: str) -> str:
    text = to_ascii_digits(normalize(text)).lower()
    return re.sub(r"[^\w\s]+|_", " ", text)
