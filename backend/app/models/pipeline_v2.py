"""Stored v2 pipeline artifacts (JSON validated by app.schemas.research)."""
from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from app.database import Base
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin

JSONType = JSON().with_variant(SQLITE_JSON(), "sqlite")


class AdExtractionRecord(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """AdExtraction (schemas v2) for one advertisement; latest row wins."""

    __tablename__ = "ad_extractions_v2"

    advertisement_id: Mapped[str] = mapped_column(String(36), ForeignKey("advertisements.id"), index=True)
    extractor: Mapped[str] = mapped_column(String(64))
    schema_version: Mapped[str] = mapped_column(String(16))
    payload: Mapped[dict] = mapped_column(JSONType)


class ResearchDossierRecord(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """ResearchDossierV2 written by the research agent; latest row wins. No verdicts here."""

    __tablename__ = "research_dossiers_v2"

    advertisement_id: Mapped[str] = mapped_column(String(36), ForeignKey("advertisements.id"), index=True)
    dossier_id: Mapped[str] = mapped_column(String(64), index=True)
    schema_version: Mapped[str] = mapped_column(String(16))
    payload: Mapped[dict] = mapped_column(JSONType)
