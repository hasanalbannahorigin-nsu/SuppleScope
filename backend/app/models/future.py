from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class FutureModelConfiguration(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "future_model_configurations"

    name: Mapped[str] = mapped_column(String(128), default="custom_verification_model")
    status: Mapped[str] = mapped_column(String(32), default="NOT_INTEGRATED")
    description: Mapped[str] = mapped_column(
        Text,
        default=(
            "Placeholder for the future custom verification model. "
            "This model is not trained or integrated in the current development phase."
        ),
    )
