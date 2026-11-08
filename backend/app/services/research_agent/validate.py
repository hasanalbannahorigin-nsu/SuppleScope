"""Plain-code checks on passages and values. The semantic checks live in the product phase."""
from __future__ import annotations

from app.services.text.bangla import match_key, numbers_in, span_score

QUOTE_THRESHOLD = 90
VALUE_THRESHOLD = 80


def quote_in_source(passage: str, source_text: str) -> bool:
    return bool(passage.strip()) and span_score(passage, source_text) >= QUOTE_THRESHOLD


def value_in_passage(value: str, passage: str, *, extra_values: tuple[str | None, ...] = ()) -> bool:
    """The value (and any amount/unit numbers) must appear in the passage."""
    if not value.strip():
        return False
    if span_score(value, passage) < VALUE_THRESHOLD:
        return False
    nums = numbers_in(passage)
    for extra in extra_values:
        if not extra:
            continue
        for n in numbers_in(extra):
            if n not in nums:
                return False
    return True


def same_text(a: str | None, b: str | None) -> bool:
    return bool(a and b) and match_key(a) == match_key(b)
