"""Shared building blocks: sources, facts, evidence records, status codes."""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "2.0.0"


class Strict(BaseModel):
    """Reject unknown keys so a contract change is never silent."""

    model_config = ConfigDict(extra="forbid")


class ResearchStatus(str, Enum):
    """Outcome of a research question. Absence of evidence is never 'false'."""

    FOUND = "FOUND"
    PARTIAL = "PARTIAL"  # some but not all of the asked-for information was found
    NOT_FOUND = "NOT_FOUND"  # searched within budget, nothing reliable found
    IDENTITY_UNCERTAIN = "IDENTITY_UNCERTAIN"  # cannot research product-level facts safely
    NO_RELEVANT_STUDY_FOUND = "NO_RELEVANT_STUDY_FOUND"  # literature searched, nothing relevant
    CONFLICTING = "CONFLICTING"  # sources disagree; see conflicts
    NOT_SEARCHED = "NOT_SEARCHED"  # budget/time/retry limit reached before this was searched


class SourceType(str, Enum):
    OFFICIAL_MANUFACTURER = "OFFICIAL_MANUFACTURER"  # manufacturer's own site/pages
    OFFICIAL_LABEL = "OFFICIAL_LABEL"  # product label / package insert from the manufacturer
    REGULATORY = "REGULATORY"  # e.g. DGDA registry, FDA, other regulator records
    THIRD_PARTY_RETAIL = "THIRD_PARTY_RETAIL"  # online pharmacies, marketplaces
    THIRD_PARTY_OTHER = "THIRD_PARTY_OTHER"  # news, blogs, review sites, social media
    SCIENTIFIC_LITERATURE = "SCIENTIFIC_LITERATURE"  # PubMed / Europe PMC / journal pages
    CLINICAL_TRIAL_REGISTRY = "CLINICAL_TRIAL_REGISTRY"  # ClinicalTrials.gov etc.
    DRUG_LABEL_DATABASE = "DRUG_LABEL_DATABASE"  # OpenFDA drug labels etc.
    REFERENCE_DATABASE = "REFERENCE_DATABASE"  # NIH ODS fact sheets, LiverTox, etc.
    ADVERTISEMENT = "ADVERTISEMENT"  # the uploaded ad itself


class InformationClass(str, Enum):
    """How a finding should be read. Shown separately in the report and to the verifier."""

    AD_STATED = "AD_STATED"  # visible/audible in the uploaded ad only
    OFFICIAL_MANUFACTURER = "OFFICIAL_MANUFACTURER"
    OFFICIAL_LABEL = "OFFICIAL_LABEL"
    REGULATORY = "REGULATORY"
    THIRD_PARTY = "THIRD_PARTY"
    SCIENTIFIC_EVIDENCE = "SCIENTIFIC_EVIDENCE"
    UNVERIFIED = "UNVERIFIED"  # failed or skipped validation
    CONFLICTING = "CONFLICTING"  # contradicted by another source of similar or higher rank


class Validation(Strict):
    """Checks run on every finding before it enters the dossier.

    quote_in_source and value_in_passage are deterministic string checks.
    refers_to_product and passage_supports_finding are semantic checks (LLM
    reader with a fixed prompt, or rule) and record who made them.
    """

    quote_in_source: bool = Field(description="Passage found verbatim (after normalisation) in the fetched source text")
    value_in_passage: bool | None = Field(None, description="The finding's value (name, number, unit) appears in the passage; None when not applicable")
    refers_to_product: bool | None = Field(None, description="Source is about the identified product (not a look-alike); None for ingredient-level literature")
    passage_supports_finding: bool = Field(description="Passage actually states the finding, not just mentions the words")
    checked_by: str = Field(description="'rule', or the reader model id, e.g. 'gemini-flash-lite-latest'")
    notes: str | None = None

    @property
    def passed(self) -> bool:
        return (
            self.quote_in_source
            and self.value_in_passage is not False
            and self.refers_to_product is not False
            and self.passage_supports_finding
        )


class Source(Strict):
    source_id: str = Field(description="Stable id within the dossier, e.g. 'src_007'")
    source_type: SourceType
    url: str | None = None
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    nct_id: str | None = Field(None, description="ClinicalTrials.gov id")
    title: str | None = None
    site_or_publisher: str | None = None
    authors: list[str] = Field(default_factory=list)
    journal: str | None = None
    publication_year: int | None = None
    publication_date: str | None = Field(None, description="As given by the source, ISO when parseable")
    language: str | None = Field(None, description="'bn', 'en', ...")
    retrieved_at: datetime
    retrieved_by_tool: str = Field(description="e.g. 'web_search:duckduckgo', 'fetch_page', 'pubmed_efetch'")
    content_hash: str | None = Field(None, description="sha256 of the fetched text, so passages can be re-checked")
    reliability_rank: int = Field(ge=1, le=9, description="1 = most reliable; see SOURCE_RANK in the agent")
    notes: str | None = None


class Finding(Strict):
    """One atomic fact about the product or an ingredient, with provenance."""

    finding_id: str = Field(description="e.g. 'f_012'")
    field: str = Field(description="What the fact is about, e.g. 'ingredient', 'amount', 'manufacturer_name', 'registration_number', 'dosage', 'intended_purpose', 'warning'")
    value: str
    unit: str | None = None
    subject_id: str | None = Field(None, description="product candidate id or ingredient_id this fact belongs to")
    source_id: str
    passage: str = Field(description="Supporting text copied from the source")
    information_class: InformationClass
    validation: Validation


class StudyType(str, Enum):
    SYSTEMATIC_REVIEW = "SYSTEMATIC_REVIEW"
    META_ANALYSIS = "META_ANALYSIS"
    RCT = "RCT"
    NON_RANDOMIZED_TRIAL = "NON_RANDOMIZED_TRIAL"
    OBSERVATIONAL = "OBSERVATIONAL"
    CASE_REPORT = "CASE_REPORT"
    ANIMAL = "ANIMAL"
    IN_VITRO = "IN_VITRO"
    NARRATIVE_REVIEW = "NARRATIVE_REVIEW"
    REGISTERED_TRIAL_NO_RESULTS = "REGISTERED_TRIAL_NO_RESULTS"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class EvidenceScope(str, Enum):
    """Kept apart everywhere: ingredient evidence is never product evidence."""

    INGREDIENT = "INGREDIENT"
    FINISHED_PRODUCT = "FINISHED_PRODUCT"


class EvidenceRecord(Strict):
    """One study (or registry entry) as read from its abstract or record."""

    evidence_id: str = Field(description="e.g. 'ev_031'")
    scope: EvidenceScope
    subject_id: str = Field(description="ingredient_id for INGREDIENT scope, product candidate id for FINISHED_PRODUCT")
    subject_name: str
    source_id: str
    study_type: StudyType
    human_study: bool | None = None
    population: str | None = None
    sample_size: int | None = None
    intervention: str | None = None
    dose: str | None = Field(None, description="Dose tested, as reported")
    duration: str | None = None
    comparator: str | None = None
    outcomes: list[str] = Field(default_factory=list)
    result_summary: str | None = Field(None, description="What the study reports, in neutral words; no judgement of the ad")
    limitations: list[str] = Field(default_factory=list)
    passage: str = Field(description="Abstract/record excerpt supporting result_summary")
    abstract_available: bool
    validation: Validation


class Conflict(Strict):
    topic: str = Field(description="e.g. 'ingredient list', 'amount of vitamin D3'")
    finding_ids: list[str]
    description: str


class MissingItem(Strict):
    item: str = Field(description="e.g. 'official label', 'amount of chromium picolinate', 'studies on finished product'")
    phase: str
    status: ResearchStatus
    queries_tried: list[str] = Field(default_factory=list)
    reason: str | None = None
