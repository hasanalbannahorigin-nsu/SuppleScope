from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class Advertisement(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "advertisements"

    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    original_filename: Mapped[str] = mapped_column(String(512))
    media_type: Mapped[str] = mapped_column(String(32))  # image | video
    content_type: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(64), default="uploaded", index=True)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    owner = relationship("User", back_populates="advertisements")
    files = relationship("UploadedFile", back_populates="advertisement", cascade="all, delete-orphan")
    frames = relationship("ExtractedFrame", back_populates="advertisement", cascade="all, delete-orphan")
    ocr_results = relationship("OcrResult", back_populates="advertisement", cascade="all, delete-orphan")
    transcripts = relationship(
        "SpeechTranscript", back_populates="advertisement", cascade="all, delete-orphan"
    )
    products = relationship(
        "ExtractedProduct", back_populates="advertisement", cascade="all, delete-orphan"
    )
    claims = relationship("ExtractedClaim", back_populates="advertisement", cascade="all, delete-orphan")
    mentioned_ingredients = relationship(
        "MentionedIngredient", back_populates="advertisement", cascade="all, delete-orphan"
    )
    jobs = relationship("ResearchJob", back_populates="advertisement", cascade="all, delete-orphan")
    patient_context = relationship(
        "PatientContext", back_populates="advertisement", uselist=False, cascade="all, delete-orphan"
    )
    identification = relationship(
        "ProductIdentification",
        back_populates="advertisement",
        uselist=False,
        cascade="all, delete-orphan",
    )
    composition = relationship(
        "ProductComposition",
        back_populates="advertisement",
        uselist=False,
        cascade="all, delete-orphan",
    )
    report = relationship(
        "ResearchReport", back_populates="advertisement", uselist=False, cascade="all, delete-orphan"
    )
