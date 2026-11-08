"""Advertisement extraction: structured fields read from the OCR text ONLY.

Every value has a span pointing at the ad text. Nothing is looked up, inferred
or completed here: a field the ad does not show is null / empty.
"""
from __future__ import annotations

from enum import Enum

from pydantic import Field

from app.schemas.research.common import SCHEMA_VERSION, Strict


class TextSpan(Strict):
    text: str = Field(description="Verbatim span of the ad's OCR/transcript text")
    segment_id: str
    line_ids: list[str] = Field(default_factory=list)
    start_ms: int | None = Field(None, description="For video frames / speech")


class AdField(Strict):
    value: str = Field(description="As written in the ad (original script)")
    span: TextSpan


class AdIngredient(Strict):
    mention_id: str = Field(description="e.g. 'ing_m01'")
    name_original: str = Field(description="As written in the ad")
    name_en: str | None = Field(None, description="English rendering of the name for searching; a translation, not new information")
    amount: str | None = Field(None, description="Only if shown in the ad, e.g. '৬০০ মি.গ্রা.'")
    span: TextSpan


class ClaimType(str, Enum):
    CURE = "CURE"
    TREATMENT = "TREATMENT"
    PREVENTION = "PREVENTION"
    IMPROVEMENT = "IMPROVEMENT"
    SAFETY = "SAFETY"  # "no side effects", "100% safe"
    REPLACEMENT = "REPLACEMENT"  # "no need for insulin/medicine"
    OTHER = "OTHER"


class AdClaim(Strict):
    claim_id: str = Field(description="e.g. 'c01'; stable across the pipeline")
    text_original: str = Field(description="One atomic claim (one effect on one target), ad wording")
    text_en: str | None = Field(None, description="English translation for search; not used as evidence")
    claim_type: ClaimType
    target: str | None = Field(None, description="Condition or body function, e.g. 'ডায়াবেটিস'")
    target_en: str | None = None
    qualifiers: list[str] = Field(default_factory=list, description="Time frames, magnitudes, absolutes: '৭ দিনে', 'চিরতরে', '১০০%'")
    negated: bool = False
    span: TextSpan


class AdUsage(Strict):
    text: str
    dose_amount: float | None = None
    dose_unit: str | None = None
    times_per_day: float | None = None
    timing: str | None = None
    duration: str | None = None
    population: str | None = None
    span: TextSpan


class AdManufacturer(Strict):
    name: AdField | None = None
    address: AdField | None = None
    phone: AdField | None = None
    website: AdField | None = None


class RejectedItem(Strict):
    path: str = Field(description="e.g. 'claims[3]'")
    problems: list[str]
    item: dict


class ExtractorInfo(Strict):
    name: str = Field(description="'grounded_llm', 'regex_baseline', later the custom model")
    version: str
    model: str | None = None


class AdExtraction(Strict):
    schema_version: str = SCHEMA_VERSION
    ad_id: str
    extractor: ExtractorInfo
    product_name: AdField | None = None
    brand: AdField | None = None
    manufacturer: AdManufacturer = Field(default_factory=AdManufacturer)
    ingredients: list[AdIngredient] = Field(default_factory=list)
    claims: list[AdClaim] = Field(default_factory=list)
    usage: list[AdUsage] = Field(default_factory=list)
    warnings: list[AdField] = Field(default_factory=list)
    other_text: list[AdField] = Field(default_factory=list, description="Prices, phone/order lines, slogans, certifications as printed")
    unreadable: list[str] = Field(default_factory=list, description="Fragments too garbled to interpret")
    rejected: list[RejectedItem] = Field(default_factory=list, description="Items dropped because their span was not found in the ad text")
