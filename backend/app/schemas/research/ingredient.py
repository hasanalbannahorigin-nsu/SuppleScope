"""Ingredient-by-ingredient research output (scope INGREDIENT only)."""
from __future__ import annotations

from pydantic import Field

from app.schemas.research.common import ResearchStatus, Strict


class EffectEvidence(Strict):
    effect: str = Field(description="Proposed effect studied, e.g. 'lowers fasting blood glucose'")
    evidence_ids: list[str] = Field(description="EvidenceRecords (scope INGREDIENT) about this effect")
    summary: str | None = Field(None, description="What the linked studies report, neutral; no claim judgement")


class SafetyItem(Strict):
    description: str
    severity_reported: str | None = Field(None, description="As stated by the source, not assigned by the agent")
    evidence_ids: list[str] = Field(default_factory=list)
    finding_ids: list[str] = Field(default_factory=list)


class DrugInteraction(Strict):
    drug_or_class: str = Field(description="e.g. 'insulin', 'sulfonylureas', 'warfarin'")
    description: str
    severity_reported: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    finding_ids: list[str] = Field(default_factory=list)


class DiseaseCaution(Strict):
    condition: str = Field(description="e.g. 'pregnancy', 'chronic kidney disease'")
    description: str
    evidence_ids: list[str] = Field(default_factory=list)
    finding_ids: list[str] = Field(default_factory=list)


class IngredientSafety(Strict):
    status: ResearchStatus
    side_effects: list[SafetyItem] = Field(default_factory=list)
    contraindications: list[SafetyItem] = Field(default_factory=list)
    drug_interactions: list[DrugInteraction] = Field(default_factory=list)
    disease_cautions: list[DiseaseCaution] = Field(default_factory=list)


class IngredientResearch(Strict):
    ingredient_id: str = Field(description="Matches ComposedIngredient.ingredient_id")
    name: str
    standardized_name: str | None = None
    scientific_name: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    origin: str = Field(description="'AD_ONLY', 'OFFICIAL_ONLY', 'AD_AND_OFFICIAL', 'THIRD_PARTY_ONLY'")
    status: ResearchStatus
    description: str | None = Field(None, description="What the ingredient is, from a cited reference source")
    description_finding_ids: list[str] = Field(default_factory=list)
    effects: list[EffectEvidence] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list, description="All studies found for this ingredient")
    tested_dose_range: str | None = Field(None, description="Range of doses used in the linked human studies, as reported")
    safety: IngredientSafety
    queries_tried: list[str] = Field(default_factory=list)
    notes: str | None = None
