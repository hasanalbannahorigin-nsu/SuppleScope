"""Product research output: identity, brand/manufacturer, official info, composition,
ad-vs-official differences, and finished-product evidence."""
from __future__ import annotations

from enum import Enum

from pydantic import Field

from app.schemas.research.common import ResearchStatus, Strict


class IdentityStatus(str, Enum):
    VERIFIED = "VERIFIED"  # official source matches the ad on name + brand/manufacturer (+ pack/ingredients where visible)
    PROBABLE = "PROBABLE"  # strong match but no official source
    IDENTITY_UNCERTAIN = "IDENTITY_UNCERTAIN"  # several plausible candidates, or weak match
    NOT_FOUND = "NOT_FOUND"  # no candidate found within budget


class MatchScores(Strict):
    """0..1 per criterion; None when the ad gives nothing to compare."""

    name: float | None = Field(None, ge=0, le=1)
    brand: float | None = Field(None, ge=0, le=1)
    manufacturer: float | None = Field(None, ge=0, le=1)
    ingredient_overlap: float | None = Field(None, ge=0, le=1)
    form_and_pack: float | None = Field(None, ge=0, le=1)
    country: float | None = Field(None, ge=0, le=1)


class ProductCandidate(Strict):
    candidate_id: str = Field(description="e.g. 'prod_c1'")
    name: str
    brand: str | None = None
    manufacturer: str | None = None
    country: str | None = None
    form: str | None = Field(None, description="capsule, tablet, powder, syrup ...")
    source_ids: list[str]
    match_scores: MatchScores
    overall_score: float = Field(ge=0, le=1)
    rejected_reason: str | None = Field(None, description="Why this look-alike is not the advertised product")


class ProductIdentity(Strict):
    status: IdentityStatus
    confidence: float = Field(ge=0, le=1)
    chosen_candidate_id: str | None = None
    candidates: list[ProductCandidate] = Field(default_factory=list)
    reasoning: str = Field(description="Which evidence decided identity, in plain words")
    queries_tried: list[str] = Field(default_factory=list)


class BrandManufacturer(Strict):
    status: ResearchStatus
    brand_finding_ids: list[str] = Field(default_factory=list)
    manufacturer_finding_ids: list[str] = Field(default_factory=list)
    official_website: str | None = None
    official_website_source_id: str | None = None
    country: str | None = None


class ComposedIngredient(Strict):
    ingredient_id: str = Field(description="e.g. 'ing_01'; used by IngredientResearch and ClaimResearch")
    name: str = Field(description="As on the official/third-party source")
    standardized_name: str | None = Field(None, description="Searchable English name, e.g. 'Gymnema sylvestre leaf extract'")
    amount: str | None = None
    unit: str | None = None
    per: str | None = Field(None, description="'per capsule', 'per 10 ml' ...")
    information_class: str = Field(description="InformationClass of the best finding for this ingredient")
    finding_ids: list[str]
    ad_mention_id: str | None = Field(None, description="AdIngredient.mention_id if the ad also shows it")


class Composition(Strict):
    status: ResearchStatus
    ingredients: list[ComposedIngredient] = Field(default_factory=list)
    serving_size_finding_ids: list[str] = Field(default_factory=list)
    dosage_directions_finding_ids: list[str] = Field(default_factory=list)
    intended_purpose_finding_ids: list[str] = Field(default_factory=list)
    manufacturer_description_finding_ids: list[str] = Field(default_factory=list)
    label_warning_finding_ids: list[str] = Field(default_factory=list)
    label_source_ids: list[str] = Field(default_factory=list, description="Official label / insert sources used")


class RegulatoryRecord(Strict):
    authority: str = Field(description="e.g. 'DGDA (Bangladesh)', 'BSTI', 'FDA'")
    registration_number: str | None = None
    category: str | None = Field(None, description="e.g. 'Ayurvedic', 'Unani', 'food supplement'")
    status: str | None = None
    finding_ids: list[str]


class DiscrepancyKind(str, Enum):
    INGREDIENT_IN_AD_NOT_IN_OFFICIAL = "INGREDIENT_IN_AD_NOT_IN_OFFICIAL"
    INGREDIENT_IN_OFFICIAL_NOT_IN_AD = "INGREDIENT_IN_OFFICIAL_NOT_IN_AD"
    AMOUNT_MISMATCH = "AMOUNT_MISMATCH"
    DOSAGE_MISMATCH = "DOSAGE_MISMATCH"
    NAME_OR_BRAND_MISMATCH = "NAME_OR_BRAND_MISMATCH"
    CLAIM_NOT_IN_OFFICIAL_MATERIAL = "CLAIM_NOT_IN_OFFICIAL_MATERIAL"
    WARNING_IN_OFFICIAL_NOT_IN_AD = "WARNING_IN_OFFICIAL_NOT_IN_AD"


class Discrepancy(Strict):
    kind: DiscrepancyKind
    ad_ref: str | None = Field(None, description="AdExtraction path, e.g. 'ingredients[0]' or 'claims[c03]'")
    finding_ids: list[str] = Field(default_factory=list)
    description: str


class FinishedProductResearch(Strict):
    """Studies on the exact finished product. Never filled from ingredient studies."""

    status: ResearchStatus
    evidence_ids: list[str] = Field(default_factory=list, description="EvidenceRecords with scope FINISHED_PRODUCT")
    manufacturer_claimed_studies: list[str] = Field(default_factory=list, description="finding_ids where the manufacturer cites a study that was not independently found")
    queries_tried: list[str] = Field(default_factory=list)
    notes: str | None = None


class ProductResearch(Strict):
    identity: ProductIdentity
    brand_manufacturer: BrandManufacturer
    composition: Composition
    regulatory: list[RegulatoryRecord] = Field(default_factory=list)
    ad_vs_official: list[Discrepancy] = Field(default_factory=list)
    finished_product: FinishedProductResearch
