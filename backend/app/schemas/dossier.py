from typing import Any, Literal

from pydantic import BaseModel, Field


VerificationStatus = Literal[
    "OFFICIALLY_VERIFIED",
    "LABEL_REPORTED",
    "THIRD_PARTY_REPORTED",
    "UNVERIFIED",
    "NOT_FOUND",
]

IdentityStatus = Literal["verified", "uncertain", "not_found"]


class DossierIngredient(BaseModel):
    name: str
    alternative_names: str | None = None
    scientific_name: str | None = None
    amount: float | None = None
    unit: str | None = None
    source_url: str | None = None
    source_type: str | None = None
    verification_status: VerificationStatus = "UNVERIFIED"


class ProductIdentity(BaseModel):
    product_name: str | None = None
    brand: str | None = None
    manufacturer: str | None = None
    product_category: str | None = None
    official_website: str | None = None
    formulation: str | None = None
    description: str | None = None
    identity_status: IdentityStatus = "not_found"
    notes: str | None = None


class CompositionBlock(BaseModel):
    ingredients: list[DossierIngredient] = Field(default_factory=list)
    composition_status: str = "not_found"
    notes: str | None = None


class SourceRef(BaseModel):
    title: str | None = None
    url: str | None = None
    source_type: str
    publisher: str | None = None
    retrieved_for: str | None = None


class ScientificEvidenceItem(BaseModel):
    title: str
    authors: str | None = None
    publication_year: int | None = None
    doi: str | None = None
    url: str | None = None
    study_type: str | None = None
    population: str | None = None
    intervention: str | None = None
    dosage: str | None = None
    outcomes: str | None = None
    limitations: str | None = None
    source_api: str | None = None
    related_ingredient: str | None = None
    evidence_scope: Literal["product", "ingredient"] = "ingredient"
    evidence_passages: list[str] = Field(default_factory=list)


class IngredientResearchItem(BaseModel):
    ingredient_name: str
    scientific_name: str | None = None
    common_name: str | None = None
    biological_properties: str | None = None
    intended_uses: str | None = None
    known_health_effects: str | None = None
    side_effects: str | None = None
    safety_concerns: str | None = None
    human_studies_summary: str | None = None
    animal_studies_summary: str | None = None
    laboratory_studies_summary: str | None = None
    sources: list[SourceRef] = Field(default_factory=list)


class EffectivenessItem(BaseModel):
    scope: Literal["product", "ingredient"]
    subject: str
    summary: str | None = None
    evidence: list[ScientificEvidenceItem] = Field(default_factory=list)
    caveat: str | None = None


class InteractionItem(BaseModel):
    ingredient: str
    medicine: str
    interaction_information: str | None = None
    possible_concerns: str | None = None
    evidence: str | None = None
    uncertainty: str | None = None
    source_references: list[SourceRef] = Field(default_factory=list)


class DiseaseRiskItem(BaseModel):
    ingredient: str
    disease: str
    available_evidence: str | None = None
    potential_concern: str | None = None
    limitations: str | None = None
    source_references: list[SourceRef] = Field(default_factory=list)


class ResearchDossier(BaseModel):
    product_identity: ProductIdentity
    composition: CompositionBlock
    product_information: dict[str, Any] = Field(default_factory=dict)
    ingredient_research: list[IngredientResearchItem] = Field(default_factory=list)
    scientific_evidence: list[ScientificEvidenceItem] = Field(default_factory=list)
    effectiveness_research: list[EffectivenessItem] = Field(default_factory=list)
    side_effects: list[dict[str, Any]] = Field(default_factory=list)
    supplement_drug_interactions: list[InteractionItem] = Field(default_factory=list)
    supplement_disease_risks: list[DiseaseRiskItem] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    medical_disclaimer: str = (
        "SuppleScope collects and organizes publicly available research. "
        "It does not diagnose, treat, or prescribe. The custom verification "
        "model is not integrated. Findings are not medical advice."
    )
