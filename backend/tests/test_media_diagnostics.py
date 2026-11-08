from types import SimpleNamespace

from app.services import media_diagnostics


def test_media_diagnostics_reports_provider_and_tool_availability(monkeypatch):
    settings = SimpleNamespace(
        ocr_primary_provider="tesseract",
        ocr_secondary_provider="google_vision",
        ffmpeg_path="ffmpeg",
        ffprobe_path="ffprobe",
        stt_provider="none",
    )
    monkeypatch.setattr(media_diagnostics, "get_settings", lambda: settings)
    monkeypatch.setattr(
        media_diagnostics.TesseractOcrProvider,
        "unavailable_reason",
        lambda self: "Install ben language data.",
    )
    monkeypatch.setattr(
        media_diagnostics.GoogleVisionOcrProvider,
        "unavailable_reason",
        lambda self: "Cloud credentials are not configured.",
    )
    monkeypatch.setattr(
        media_diagnostics,
        "_executable_status",
        lambda name, path: {
            "available": name == "ffmpeg",
            "path": None,
            "version": None,
            "reason": None if name == "ffmpeg" else "Install ffprobe.",
        },
    )
    monkeypatch.setattr(
        media_diagnostics,
        "transcription_status",
        lambda: ("none", None),
    )

    result = media_diagnostics.get_media_diagnostics()

    assert result["ocr"]["available"] is False
    assert result["ocr"]["languages"] == "ben+eng"
    assert "ben" in result["ocr"]["providers"]["tesseract"]["reason"]
    assert result["video"]["audio_extraction_available"] is True
    assert result["video"]["frame_extraction_available"] is False
    assert result["speech"]["available"] is False
    assert result["speech"]["reason"] == "Speech transcription is disabled (STT_PROVIDER=none)."
