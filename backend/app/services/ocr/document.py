"""Build the v2 OcrDocument from stored OcrResult rows / OcrOutput objects."""
from __future__ import annotations

from collections.abc import Iterable

from app.schemas.research.ocr import BBox, OcrDocument, OcrLine, OcrSegment, SegmentKind


def _lines(segment_id: str, text: str, boxes: list[dict] | None) -> list[OcrLine]:
    out = []
    for i, line in enumerate([ln for ln in (text or "").splitlines() if ln.strip()], start=1):
        out.append(OcrLine(line_id=f"{segment_id}_l{i:03d}", text=line.strip()))
    return out


def build_ocr_document(
    *,
    ad_id: str,
    media_type: str,
    ocr_rows: Iterable,
    frames: dict | None = None,
    transcripts: Iterable = (),
    preferred_provider: str | None = None,
    unavailable: list[str] | None = None,
) -> OcrDocument:
    """ocr_rows: objects with provider, raw_text, cleaned_text, average_confidence, frame_id, bounding_boxes.
    frames: frame_id -> timestamp_ms. transcripts: objects with provider, segments/full_text."""
    rows = list(ocr_rows)
    if preferred_provider and any(r.provider == preferred_provider for r in rows):
        rows = [r for r in rows if r.provider == preferred_provider]
    frames = frames or {}
    rows.sort(key=lambda r: (frames.get(getattr(r, "frame_id", None)) or 0))
    segments: list[OcrSegment] = []
    for r in rows:
        sid = f"seg{len(segments) + 1:02d}"
        frame_id = getattr(r, "frame_id", None)
        conf = r.average_confidence
        segments.append(
            OcrSegment(
                segment_id=sid,
                kind=SegmentKind.VIDEO_FRAME if frame_id else SegmentKind.IMAGE,
                frame_id=frame_id,
                start_ms=frames.get(frame_id) if frame_id else None,
                provider=r.provider,
                raw_text=r.raw_text or "",
                cleaned_text=r.cleaned_text or "",
                average_confidence=min(max(conf, 0.0), 1.0) if conf is not None else None,
                lines=_lines(sid, r.cleaned_text or "", getattr(r, "bounding_boxes", None)),
            )
        )
    for t in transcripts:
        for seg in (getattr(t, "segments", None) or [{"start": None, "end": None, "text": t.full_text}]):
            sid = f"seg{len(segments) + 1:02d}"
            text = (seg.get("text") or "").strip()
            start, end = seg.get("start"), seg.get("end")  # seconds, from the speech adapters
            segments.append(
                OcrSegment(
                    segment_id=sid, kind=SegmentKind.SPEECH,
                    start_ms=int(start * 1000) if start is not None else None,
                    end_ms=int(end * 1000) if end is not None else None,
                    provider=t.provider, raw_text=text, cleaned_text=text, lines=_lines(sid, text, None),
                )
            )
    full = "\n\n".join(s.cleaned_text for s in segments if s.cleaned_text.strip())
    return OcrDocument(ad_id=ad_id, media_type=media_type, segments=segments, full_text=full,
                       providers_unavailable=unavailable or [])


def ocr_document_from_text(ad_id: str, text: str, provider: str = "text") -> OcrDocument:
    """For evaluation and tests: one IMAGE segment from plain text."""
    class _Row:
        pass

    row = _Row()
    row.provider, row.raw_text, row.cleaned_text, row.average_confidence, row.frame_id, row.bounding_boxes = (
        provider, text, text, None, None, None)
    return build_ocr_document(ad_id=ad_id, media_type="image", ocr_rows=[row])
