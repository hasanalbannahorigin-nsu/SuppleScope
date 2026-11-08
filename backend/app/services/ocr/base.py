from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OcrBox:
    text: str
    confidence: float | None
    x: int | None = None
    y: int | None = None
    width: int | None = None
    height: int | None = None


@dataclass
class OcrOutput:
    provider: str
    raw_text: str
    cleaned_text: str
    average_confidence: float | None
    bounding_boxes: list[dict] = field(default_factory=list)
    language_hints: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


class OcrProvider(ABC):
    name: str

    def unavailable_reason(self) -> str | None:
        if self.available():
            return None
        return f"{self.name} is not available"

    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def recognize(self, image_path: Path) -> OcrOutput:
        raise NotImplementedError
