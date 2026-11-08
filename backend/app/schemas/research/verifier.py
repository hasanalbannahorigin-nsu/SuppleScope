"""Hand-off to the separate custom Verification Model, and what it returns.

The research agent never writes VerifierOutput. VerifierInput is built from a
finished ResearchDossierV2 by plain code (build_verifier_inputs), one per claim.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field

from app.schemas.research.ad_extraction import AdClaim
from app.schemas.research.claim import ClaimEvidenceLink
from app.schemas.research.common import (
    SCHEMA_VERSION,
    EvidenceRecord,
    Finding,
    MissingItem,
    ResearchStatus,
    Source,
    Strict,
)
from app.schemas.research.product import ComposedIngredient, Discrepancy, IdentityStatus


class ProductContext(Strict):
    identity_status: IdentityStatus
    identity_confidence: float
    product_name: str | None = None
    brand: str | None = None
    manufacturer: str | None = None
    composition_status: ResearchStatus
    finished_product_status: ResearchStatus


class VerifierInput(Strict):
    """One record per claim. Self-contained: the model needs nothing else."""

    schema_version: str = SCHEMA_VERSION
    dossier_id: str
    ad_id: str
    claim: AdClaim
    product: ProductContext
    related_ingredients: list[ComposedIngredient] = Field(default_factory=list)
    evidence_links: list[ClaimEvidenceLink] = Field(default_factory=list)
    evidence: list[EvidenceRecord] = Field(default_factory=list, description="Only records referenced by evidence_links")
    findings: list[Finding] = Field(default_factory=list, description="Findings referenced by related_ingredients and official_material")
    official_material_mentions_claim: bool | None = None
    discrepancies: list[Discrepancy] = Field(default_factory=list, description="Ad-vs-official differences touching this claim or its ingredients")
    sources: list[Source] = Field(default_factory=list, description="Only sources referenced above")
    claim_research_status: ResearchStatus
    missing_information: list[MissingItem] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


class VerdictLabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_ASSESSABLE = "NOT_ASSESSABLE"


class VerifierOutput(Strict):
    """Written only by the custom Verification Model (guide section 7)."""

    schema_version: str = SCHEMA_VERSION
    dossier_id: str
    ad_id: str
    claim_id: str
    label: VerdictLabel
    confidence: float | None = Field(None, ge=0, le=1)
    label_probabilities: dict[VerdictLabel, float] | None = None
    abstained: bool = False
    rationale: str | None = None
    evidence_ids: list[str] = Field(default_factory=list, description="Evidence the model relied on")
    model_name: str
    model_version: str
    created_at: datetime
