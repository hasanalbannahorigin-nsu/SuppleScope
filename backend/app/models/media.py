from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.database import Base
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin

JSONType = JSON().with_variant(SQLITE_JSON(), "sqlite")


class UploadedFile(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "uploaded_files"

    advertisement_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("advertisements.id"), index=True
    )
    storage_key: Mapped[str] = mapped_column(String(1024))
    original_filename: Mapped[str] = mapped_column(String(512))
    content_type: Mapped[str] = mapped_column(String(128))
    size_bytes: Mapped[int] = mapped_column(Integer)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    role: Mapped[str] = mapped_column(String(32), default="original")  # original | derived

    advertisement = relationship("Advertisement", back_populates="files")


class ExtractedFrame(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "extracted_frames"

    advertisement_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("advertisements.id"), index=True
    )
    frame_index: Mapped[int] = mapped_column(Integer)
    timestamp_ms: Mapped[int] = mapped_column(Integer, index=True)
    storage_key: Mapped[str] = mapped_column(String(1024))
    scene_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_duplicate: Mapped[bool] = mapped_column(default=False)
    selection_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)

    advertisement = relationship("Advertisement", back_populates="frames")
    ocr_results = relationship("OcrResult", back_populates="frame")


class OcrResult(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "ocr_results"

    advertisement_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("advertisements.id"), index=True
    )
    frame_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("extracted_frames.id"), nullable=True, index=True
    )
    provider: Mapped[str] = mapped_column(String(64), index=True)
    raw_text: Mapped[str] = mapped_column(Text, default="")
    cleaned_text: Mapped[str] = mapped_column(Text, default="")
    average_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    bounding_boxes: Mapped[dict | list | None] = mapped_column(JSONType, nullable=True)
    language_hints: Mapped[list | None] = mapped_column(JSONType, nullable=True)
    extra: Mapped[dict | None] = mapped_column(JSONType, nullable=True)

    advertisement = relationship("Advertisement", back_populates="ocr_results")
    frame = relationship("ExtractedFrame", back_populates="ocr_results")


class SpeechTranscript(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "speech_transcripts"

    advertisement_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("advertisements.id"), index=True
    )
    provider: Mapped[str] = mapped_column(String(64))
    language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    full_text: Mapped[str] = mapped_column(Text, default="")
    segments: Mapped[list | None] = mapped_column(JSONType, nullable=True)

    advertisement = relationship("Advertisement", back_populates="transcripts")
