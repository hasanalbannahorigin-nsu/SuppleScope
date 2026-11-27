"""Patient-specific risk report (ingredients of one product x one patient's medicines and conditions)."""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field

from app.schemas.research.common import SCHEMA_VERSION, Strict


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class PatientProfile(Strict):
    age: int | None = Field(None, ge=0, le=120)
    medications: list[str] = Field(default_factory=list, description="As the user typed them (brand or generic)")
    conditions: list[str] = Field(default_factory=list, description="As the user typed them (English or Bangla)")
    pregnant: bool | None = None
    breastfeeding: bool | None = None
    free_text: str | None = None


class PatientFactor(Strict):
    kind: str = Field(description="'drug', 'condition' or 'population'")
    name: str = Field(description="Canonical name: generic medicine or condition")
    typed_as: str
    classes: list[str] = Field(default_factory=list, description="Drug classes, e.g. 'anticoagulant'")
    normalised_by: str = Field(description="'brand_table', 'generic_name', 'rxnorm', 'condition_terms' or 'none'")


class RiskItem(Strict):
    ingredient_id: str | None
    ingredient: str
    factor: str = Field(description="The medicine, class or condition it concerns")
    factor_kind: str
    level: RiskLevel
    mechanism: str
    advice: str
    basis: str = Field(description="'curated_rule' (pharmacist-reviewable table) or 'dossier_finding' (quoted source)")
    rule_id: str | None = None
    finding_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list, description="Source names or URLs")


class PatientRiskReport(Strict):
    schema_version: str = SCHEMA_VERSION
    dossier_id: str
    ad_id: str
    created_at: datetime
    engine_version: str
    overall: RiskLevel
    items: list[RiskItem] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list, description="What could not be checked, and why")
    factors: list[PatientFactor] = Field(default_factory=list)
    ingredients_considered: list[str] = Field(default_factory=list)
    disclaimer: str
