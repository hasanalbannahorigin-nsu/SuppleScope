"""The complete Research Dossier written by the research agent.

It is the agent's ONLY output. It has no verdict field by design.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from app.schemas.research.ad_extraction import AdExtraction
from app.schemas.research.claim import ClaimResearch
from app.schemas.research.common import (
    SCHEMA_VERSION,
    Conflict,
    EvidenceRecord,
    EvidenceScope,
    Finding,
    MissingItem,
    Source,
    Strict,
)
from app.schemas.research.ingredient import IngredientResearch
from app.schemas.research.product import ProductResearch


class ResearchBudget(Strict):
    max_search_calls: int
    max_fetches: int
    max_llm_calls: int
    max_retries_per_call: int
    max_seconds: int
    max_literature_calls: int = 0
    used_search_calls: int = 0
    used_fetches: int = 0
    used_llm_calls: int = 0
    used_literature_calls: int = 0
    elapsed_seconds: float = 0
    exhausted: bool = False
    exhausted_reason: str | None = Field(None, description="'search', 'fetch', 'llm', 'literature', 'time'")


class AgentInfo(Strict):
    agent_version: str
    llm_provider: str
    llm_model: str
    search_provider: str = Field(description="'duckduckgo', 'tavily', ... (configurable)")
    literature_providers: list[str] = Field(default_factory=list)


class LogEntry(Strict):
    step: int
    phase: str = Field(description="identity | brand_manufacturer | official_sources | composition | ingredient | finished_product | claim | safety | validation")
    action: str = Field(description="'search', 'fetch', 'read', 'validate', 'plan'")
    tool: str | None = None
    query: str | None = None
    target_url: str | None = None
    result_count: int | None = None
    produced_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    at: datetime


class SafetySummary(Strict):
    """Cross-ingredient view for the report and later patient-risk step. References only;
    the detail lives in IngredientResearch.safety."""

    ingredient_ids_with_interactions: list[str] = Field(default_factory=list)
    ingredient_ids_with_disease_cautions: list[str] = Field(default_factory=list)
    label_warning_finding_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class ResearchDossierV2(Strict):
    schema_version: str = SCHEMA_VERSION
    dossier_id: str
    ad_id: str
    created_at: datetime
    agent: AgentInfo
    budget: ResearchBudget
    advertisement: AdExtraction = Field(description="Exactly what the ad shows; copied, never edited by the agent")
    product: ProductResearch
    ingredients: list[IngredientResearch] = Field(default_factory=list)
    claims: list[ClaimResearch] = Field(default_factory=list)
    safety: SafetySummary = Field(default_factory=SafetySummary)
    sources: list[Source] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    missing_information: list[MissingItem] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    research_log: list[LogEntry] = Field(default_factory=list)
    disclaimer: str = (
        "This dossier collects and organises sources. It contains no verdict on any claim and is "
        "not medical advice. Ingredient-level evidence does not show that the finished product is effective."
    )

    @model_validator(mode="after")
    def _check_references(self) -> "ResearchDossierV2":
        source_ids = {s.source_id for s in self.sources}
        finding_ids = {f.finding_id for f in self.findings}
        evidence = {e.evidence_id: e for e in self.evidence}
        claim_ids = {c.claim_id for c in self.advertisement.claims}
        problems: list[str] = []
        for f in self.findings:
            if f.source_id not in source_ids:
                problems.append(f"finding {f.finding_id} -> unknown source {f.source_id}")
        for e in self.evidence:
            if e.source_id not in source_ids:
                problems.append(f"evidence {e.evidence_id} -> unknown source {e.source_id}")
        for ing in self.ingredients:
            for eid in ing.evidence_ids + [x for eff in ing.effects for x in eff.evidence_ids]:
                rec = evidence.get(eid)
                if rec is None:
                    problems.append(f"ingredient {ing.ingredient_id} -> unknown evidence {eid}")
                elif rec.scope != EvidenceScope.INGREDIENT:
                    problems.append(f"ingredient {ing.ingredient_id} cites FINISHED_PRODUCT evidence {eid}")
        for eid in self.product.finished_product.evidence_ids:
            rec = evidence.get(eid)
            if rec is None:
                problems.append(f"finished_product -> unknown evidence {eid}")
            elif rec.scope != EvidenceScope.FINISHED_PRODUCT:
                problems.append(f"finished_product cites INGREDIENT evidence {eid}")
        for c in self.claims:
            if c.claim_id not in claim_ids:
                problems.append(f"claim research {c.claim_id} has no matching ad claim")
            for link in c.evidence_links:
                rec = evidence.get(link.evidence_id)
                if rec is None:
                    problems.append(f"claim {c.claim_id} -> unknown evidence {link.evidence_id}")
                elif rec.scope != link.scope:
                    problems.append(f"claim {c.claim_id} link {link.evidence_id} scope mismatch")
        all_finding_refs = (
            self.product.brand_manufacturer.brand_finding_ids
            + self.product.brand_manufacturer.manufacturer_finding_ids
            + [fid for ing in self.product.composition.ingredients for fid in ing.finding_ids]
        )
        for fid in all_finding_refs:
            if fid not in finding_ids:
                problems.append(f"unknown finding {fid}")
        if problems:
            raise ValueError("Dossier references are inconsistent: " + "; ".join(problems[:20]))
        return self
