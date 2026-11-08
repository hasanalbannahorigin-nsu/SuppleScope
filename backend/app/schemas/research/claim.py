"""Claim-by-claim research output. Describes evidence; contains NO verdict."""
from __future__ import annotations

from enum import Enum

from pydantic import Field

from app.schemas.research.common import EvidenceScope, ResearchStatus, Strict


class EvidenceRelation(str, Enum):
    """Relation of ONE source to the claim's topic, as the source states it.
    This is a description of the source, not a decision about the claim."""

    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    CONTEXT = "CONTEXT"  # relevant background (mechanism, safety, related outcome)


class MatchLevel(str, Enum):
    SAME = "SAME"
    SIMILAR = "SIMILAR"
    DIFFERENT = "DIFFERENT"
    UNKNOWN = "UNKNOWN"


class ClaimEvidenceMatch(Strict):
    """How closely a study matches what the ad actually says."""

    outcome: MatchLevel = Field(description="Study outcome vs the claimed effect")
    population: MatchLevel = Field(description="Study population vs the ad's audience")
    dose: MatchLevel = Field(description="Tested dose vs the product's dose (if known)")
    intervention: MatchLevel = Field(description="Same finished product, same ingredient form, or different")
    qualifiers: MatchLevel = Field(description="Magnitude / time frame / absoluteness ('in 7 days', 'permanently')")
    human_evidence: bool | None = None
    notes: str | None = None


class ClaimEvidenceLink(Strict):
    evidence_id: str
    scope: EvidenceScope = Field(description="INGREDIENT or FINISHED_PRODUCT; copied from the EvidenceRecord")
    ingredient_id: str | None = Field(None, description="Set when scope is INGREDIENT")
    relation: EvidenceRelation
    match: ClaimEvidenceMatch
    relevant_passage: str = Field(description="The part of the evidence passage that bears on this claim")


class ClaimResearch(Strict):
    claim_id: str = Field(description="AdClaim.claim_id")
    claim_text_original: str
    claim_text_en: str | None = None
    claim_type: str
    target: str | None = None
    qualifiers: list[str] = Field(default_factory=list)
    linked_ingredient_ids: list[str] = Field(default_factory=list, description="Ingredients plausibly related to the claim's target")
    research_questions: list[str] = Field(default_factory=list, description="Questions the agent searched, in English")
    status: ResearchStatus
    finished_product_status: ResearchStatus = Field(description="Was any FINISHED_PRODUCT evidence found for this claim?")
    evidence_links: list[ClaimEvidenceLink] = Field(default_factory=list)
    official_material_mentions_claim: bool | None = Field(None, description="Does the official label/site make the same claim?")
    official_material_finding_ids: list[str] = Field(default_factory=list)
    queries_tried: list[str] = Field(default_factory=list)
    notes: str | None = None
