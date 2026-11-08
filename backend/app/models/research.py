from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.database import Base
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin

JSONType = JSON().with_variant(SQLITE_JSON(), "sqlite")


class ResearchJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "research_jobs"

    advertisement_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("advertisements.id"), index=True
    )
    job_type: Mapped[str] = mapped_column(String(32), index=True)  # analyze | research
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    current_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    stages: Mapped[list | None] = mapped_column(JSONType, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    advertisement = relationship("Advertisement", back_populates="jobs")


class PatientContext(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "patient_contexts"

    advertisement_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("advertisements.id"), unique=True
    )
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    diseases: Mapped[str | None] = mapped_column(Text, nullable=True)
    medications: Mapped[str | None] = mapped_column(Text, nullable=True)
    health_context: Mapped[str | None] = mapped_column(Text, nullable=True)

    advertisement = relationship("Advertisement", back_populates="patient_context")


class ProductIdentification(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "product_identification"

    advertisement_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("advertisements.id"), unique=True
    )
    product_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    brand: Mapped[str | None] = mapped_column(String(255), nullable=True)
    manufacturer: Mapped[str | None] = mapped_column(String(255), nullable=True)
    product_category: Mapped[str | None] = mapped_column(String(255), nullable=True)
    official_website: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    formulation: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    identity_status: Mapped[str] = mapped_column(String(32), default="not_found")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    extra: Mapped[dict | None] = mapped_column(JSONType, nullable=True)

    advertisement = relationship("Advertisement", back_populates="identification")


class ProductComposition(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "product_compositions"

    advertisement_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("advertisements.id"), unique=True
    )
    composition_status: Mapped[str] = mapped_column(String(64), default="not_found")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    advertisement = relationship("Advertisement", back_populates="composition")
    ingredients = relationship(
        "ProductCompositionIngredient",
        back_populates="composition",
        cascade="all, delete-orphan",
    )


class ProductCompositionIngredient(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "product_composition_ingredients"

    composition_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("product_compositions.id"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    alternative_names: Mapped[str | None] = mapped_column(Text, nullable=True)
    scientific_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    unit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    source_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    verification_status: Mapped[str] = mapped_column(String(32), default="UNVERIFIED")

    composition = relationship("ProductComposition", back_populates="ingredients")


class ResearchSource(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "research_sources"

    advertisement_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("advertisements.id"), index=True
    )
    title: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    source_type: Mapped[str] = mapped_column(String(64))
    publisher: Mapped[str | None] = mapped_column(String(255), nullable=True)
    retrieved_for: Mapped[str | None] = mapped_column(String(128), nullable=True)
    extra: Mapped[dict | None] = mapped_column(JSONType, nullable=True)


class ScientificPaper(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "scientific_papers"

    advertisement_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("advertisements.id"), index=True
    )
    title: Mapped[str] = mapped_column(String(1024))
    authors: Mapped[str | None] = mapped_column(Text, nullable=True)
    publication_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    doi: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    study_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    population: Mapped[str | None] = mapped_column(Text, nullable=True)
    intervention: Mapped[str | None] = mapped_column(Text, nullable=True)
    dosage: Mapped[str | None] = mapped_column(Text, nullable=True)
    outcomes: Mapped[str | None] = mapped_column(Text, nullable=True)
    limitations: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_api: Mapped[str | None] = mapped_column(String(64), nullable=True)
    related_ingredient: Mapped[str | None] = mapped_column(String(255), nullable=True)
    evidence_scope: Mapped[str] = mapped_column(String(32), default="ingredient")  # product | ingredient


class EvidencePassage(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "evidence_passages"

    advertisement_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("advertisements.id"), index=True
    )
    paper_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("scientific_papers.id"), nullable=True
    )
    text: Mapped[str] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)


class IngredientResearch(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "ingredient_research"

    advertisement_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("advertisements.id"), index=True
    )
    ingredient_name: Mapped[str] = mapped_column(String(255), index=True)
    scientific_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    common_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    biological_properties: Mapped[str | None] = mapped_column(Text, nullable=True)
    intended_uses: Mapped[str | None] = mapped_column(Text, nullable=True)
    known_health_effects: Mapped[str | None] = mapped_column(Text, nullable=True)
    side_effects: Mapped[str | None] = mapped_column(Text, nullable=True)
    safety_concerns: Mapped[str | None] = mapped_column(Text, nullable=True)
    human_studies_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    animal_studies_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    laboratory_studies_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    sources: Mapped[list | None] = mapped_column(JSONType, nullable=True)


class InteractionResearch(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "interaction_research"

    advertisement_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("advertisements.id"), index=True
    )
    ingredient: Mapped[str] = mapped_column(String(255))
    medicine: Mapped[str] = mapped_column(String(255))
    interaction_information: Mapped[str | None] = mapped_column(Text, nullable=True)
    possible_concerns: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    uncertainty: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_references: Mapped[list | None] = mapped_column(JSONType, nullable=True)


class DiseaseRiskResearch(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "disease_risk_research"

    advertisement_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("advertisements.id"), index=True
    )
    ingredient: Mapped[str] = mapped_column(String(255))
    disease: Mapped[str] = mapped_column(String(255))
    available_evidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    potential_concern: Mapped[str | None] = mapped_column(Text, nullable=True)
    limitations: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_references: Mapped[list | None] = mapped_column(JSONType, nullable=True)


class ResearchReport(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "research_reports"

    advertisement_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("advertisements.id"), unique=True
    )
    dossier: Mapped[dict] = mapped_column(JSONType)
    missing_information: Mapped[list | None] = mapped_column(JSONType, nullable=True)
    uncertainties: Mapped[list | None] = mapped_column(JSONType, nullable=True)

    advertisement = relationship("Advertisement", back_populates="report")
