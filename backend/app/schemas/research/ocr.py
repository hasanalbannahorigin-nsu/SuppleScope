"""OCR output: all visible (and, for video, audible) text, nothing else."""
from __future__ import annotations

from enum import Enum

from pydantic import Field

from app.schemas.research.common import SCHEMA_VERSION, Strict


class BBox(Strict):
    x: int
    y: int
    width: int
    height: int


class OcrLine(Strict):
    line_id: str = Field(description="e.g. 'seg01_l004'")
    text: str
    confidence: float | None = Field(None, ge=0, le=1)
    bbox: BBox | None = None


class SegmentKind(str, Enum):
    IMAGE = "IMAGE"  # the uploaded image
    VIDEO_FRAME = "VIDEO_FRAME"  # one selected frame of a video
    SPEECH = "SPEECH"  # one transcript segment of a video's audio


class OcrSegment(Strict):
    segment_id: str = Field(description="e.g. 'seg01'")
    kind: SegmentKind
    frame_id: str | None = None
    start_ms: int | None = Field(None, description="Frame timestamp, or speech segment start")
    end_ms: int | None = None
    provider: str = Field(description="'tesseract', 'google_vision', 'faster_whisper', ...")
    raw_text: str
    cleaned_text: str = Field(description="Unicode-normalised only; numbers, units, negations and qualifiers untouched")
    average_confidence: float | None = Field(None, ge=0, le=1)
    lines: list[OcrLine] = Field(default_factory=list)


class OcrDocument(Strict):
    schema_version: str = SCHEMA_VERSION
    ad_id: str
    media_type: str = Field(description="'image' or 'video'")
    segments: list[OcrSegment]
    full_text: str = Field(description="cleaned_text of all segments in reading/time order, separated by blank lines")
    providers_unavailable: list[str] = Field(default_factory=list, description="'google_vision: not configured', ...")
