import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services import speech


def test_openai_whisper_adapter_preserves_segments(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        stt_provider="openai_whisper_api",
        openai_api_key="test-key",
        openai_base_url="https://example.test/v1",
        openai_stt_model="whisper-test",
    )
    monkeypatch.setattr(speech, "get_settings", lambda: settings)

    class FakeClient:
        def __init__(self, **kwargs):
            self.audio = SimpleNamespace(
                transcriptions=SimpleNamespace(
                    create=lambda **options: SimpleNamespace(
                        text="  বাংলা বিজ্ঞাপন  ",
                        language="bn",
                        segments=[
                            SimpleNamespace(start=0.25, end=1.5, text="বাংলা বিজ্ঞাপন")
                        ],
                    )
                )
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeClient))
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"mock audio")

    result = speech.transcribe_audio(Path(audio))

    assert result.full_text == "বাংলা বিজ্ঞাপন"
    assert result.language == "bn"
    assert result.segments == [{"start": 0.25, "end": 1.5, "text": "বাংলা বিজ্ঞাপন"}]


def test_no_provider_is_reported_as_optional(monkeypatch):
    monkeypatch.setattr(
        speech,
        "get_settings",
        lambda: SimpleNamespace(stt_provider="none"),
    )

    assert speech.transcription_status() == ("none", None)
    with pytest.raises(RuntimeError, match="disabled"):
        speech.transcribe_audio(Path("unused.wav"))


def test_openai_provider_reports_missing_api_key(monkeypatch):
    monkeypatch.setattr(
        speech,
        "get_settings",
        lambda: SimpleNamespace(openai_api_key=""),
    )

    assert "OPENAI_API_KEY" in speech.OpenAIWhisperProvider().unavailable_reason()
