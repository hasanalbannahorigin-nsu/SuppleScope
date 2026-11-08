from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings


@dataclass
class TranscriptResult:
    provider: str
    language: str | None
    full_text: str
    segments: list[dict]


class SpeechToTextProvider(ABC):
    name: str

    @abstractmethod
    def unavailable_reason(self) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def transcribe(self, audio_path: Path) -> TranscriptResult:
        raise NotImplementedError


class OpenAIWhisperProvider(SpeechToTextProvider):
    name = "openai_whisper_api"

    def unavailable_reason(self) -> str | None:
        settings = get_settings()
        if not settings.openai_api_key:
            return "OPENAI_API_KEY is required for OpenAI Whisper transcription."
        try:
            import openai  # noqa: F401
        except ImportError:
            return "Install the backend dependency openai to use OpenAI Whisper."
        return None

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        from openai import OpenAI

        settings = get_settings()
        client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        with audio_path.open("rb") as handle:
            response = client.audio.transcriptions.create(
                model=settings.openai_stt_model,
                file=handle,
                response_format="verbose_json",
            )
        segments = []
        for item in getattr(response, "segments", None) or []:
            if hasattr(item, "model_dump"):
                payload = item.model_dump()
            elif isinstance(item, dict):
                payload = item
            else:
                payload = {
                    "start": getattr(item, "start", None),
                    "end": getattr(item, "end", None),
                    "text": getattr(item, "text", ""),
                }
            segments.append(
                {
                    "start": payload.get("start"),
                    "end": payload.get("end"),
                    "text": payload.get("text"),
                }
            )
        return TranscriptResult(
            provider=self.name,
            language=getattr(response, "language", None),
            full_text=(response.text or "").strip(),
            segments=segments,
        )


class FasterWhisperProvider(SpeechToTextProvider):
    name = "faster_whisper"

    def unavailable_reason(self) -> str | None:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return (
                "faster-whisper is not installed. Install it to enable local "
                "transcription, or configure another STT_PROVIDER."
            )
        return None

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        from faster_whisper import WhisperModel

        settings = get_settings()
        model = WhisperModel(settings.faster_whisper_model, device="cpu", compute_type="int8")
        segments_iter, info = model.transcribe(str(audio_path), vad_filter=True)
        segments = []
        texts = []
        for segment in segments_iter:
            text = segment.text
            segments.append({"start": segment.start, "end": segment.end, "text": text})
            texts.append(text)
        return TranscriptResult(
            provider=self.name,
            language=getattr(info, "language", None),
            full_text=" ".join(texts).strip(),
            segments=segments,
        )


PROVIDERS: dict[str, type[SpeechToTextProvider]] = {
    OpenAIWhisperProvider.name: OpenAIWhisperProvider,
    FasterWhisperProvider.name: FasterWhisperProvider,
}


def get_provider(name: str) -> SpeechToTextProvider:
    provider_class = PROVIDERS.get(name)
    if provider_class is None:
        raise RuntimeError(f"Unknown STT_PROVIDER: {name}")
    return provider_class()


def transcription_status() -> tuple[str, str | None]:
    name = (get_settings().stt_provider or "none").strip().lower()
    if name in {"none", "off", ""}:
        return "none", None
    provider = get_provider(name)
    return name, provider.unavailable_reason()


def transcribe_audio(audio_path: Path) -> TranscriptResult:
    name, reason = transcription_status()
    if name == "none":
        raise RuntimeError("Speech-to-text is disabled (STT_PROVIDER=none)")
    if reason:
        raise RuntimeError(reason)
    return get_provider(name).transcribe(audio_path)
