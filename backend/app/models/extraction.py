from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.database import Base
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin

JSONType = JSON().with_variant(SQLITE_JSON(), "sqlite")


class ExtractedProduct(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "extracted_products"

    advertisement_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("advertisements.id"), index=True
    )
    product_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    brand: Mapped[str | None] = mapped_column(String(255), nullable=True)
    manufacturer: Mapped[str | None] = mapped_column(String(255), nullable=True)
    product_category: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="combined")  # ocr | speech | combined
    original_span: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(nullable=True)
    extra: Mapped[dict | None] = mapped_column(JSONType, nullable=True)

    advertisement = relationship("Advertisement", back_populates="products")


class ExtractedClaim(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "extracted_claims"

    advertisement_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("advertisements.id"), index=True
    )
    claim_text_original: Mapped[str] = mapped_column(Text)
    claim_text_en: Mapped[str | None] = mapped_column(Text, nullable=True)
    claim_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="ocr")
    extra: Mapped[dict | None] = mapped_column(JSONType, nullable=True)

    advertisement = relationship("Advertisement", back_populates="claims")


class MentionedIngredient(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "ingredients"

    advertisement_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("advertisements.id"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    original_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="ocr")

    advertisement = relationship("Advertisement", back_populates="mentioned_ingredients")
