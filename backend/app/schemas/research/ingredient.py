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


class KnowledgeLink(Strict):
    """Where the ingredient was matched in a dictionary or reference database (name normalisation)."""

    database: str = Field(description="'SuppleScope dictionary', 'iDISK2.0', 'NIH ODS', 'NCCIH'")
    entity_id: str | None = Field(None, description="Id in that database, e.g. an iDISK2.0 DSI id")
    matched_name: str
    match_score: float | None = Field(None, description="0-100; 100 = exact name match")
    method: str = Field("exact", description="'exact', 'fuzzy' or 'lookup'")
    url: str | None = None
    source_id: str | None = None


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
    knowledge_links: list[KnowledgeLink] = Field(default_factory=list,
                                                 description="Dictionary / iDISK2.0 / fact-sheet matches of this ingredient")
    reference_finding_ids: list[str] = Field(default_factory=list,
                                             description="Findings (class REFERENCE) from iDISK2.0, NIH ODS, NCCIH and drug labels")
    notes: str | None = None
