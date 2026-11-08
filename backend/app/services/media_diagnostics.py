from __future__ import annotations

import logging
import shutil
import subprocess

import cv2

from app.config import get_settings
from app.services.ocr.google_vision_provider import GoogleVisionOcrProvider
from app.services.ocr.tesseract_provider import TesseractOcrProvider
from app.services.speech import transcription_status

logger = logging.getLogger(__name__)


def _executable_status(name: str, configured_path: str) -> dict[str, str | bool | None]:
    path = shutil.which(configured_path or name)
    if path is None:
        return {
            "available": False,
            "path": None,
            "version": None,
            "reason": (
                f"{name} was not found. Install FFmpeg and add its bin folder to "
                f"PATH, or set {name.upper()}_PATH in .env."
            ),
        }
    try:
        result = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "available": False,
            "path": path,
            "version": None,
            "reason": f"Could not run {path}: {exc}",
        }
    if result.returncode:
        return {
            "available": False,
            "path": path,
            "version": None,
            "reason": result.stderr.strip() or f"{path} exited with code {result.returncode}.",
        }
    first_line = result.stdout.splitlines()[0] if result.stdout else "version unknown"
    return {"available": True, "path": path, "version": first_line, "reason": None}


def get_media_diagnostics() -> dict:
    settings = get_settings()
    providers = {}
    for name, provider in (
        ("tesseract", TesseractOcrProvider()),
        ("google_vision", GoogleVisionOcrProvider()),
    ):
        reason = provider.unavailable_reason()
        providers[name] = {
            "available": reason is None,
            "reason": reason,
        }

    ffmpeg = _executable_status("ffmpeg", settings.ffmpeg_path)
    ffprobe = _executable_status("ffprobe", settings.ffprobe_path)
    opencv = {
        "available": bool(cv2.__version__),
        "version": cv2.__version__,
        "reason": None if cv2.__version__ else "OpenCV could not be loaded.",
    }

    try:
        stt_provider, stt_reason = transcription_status()
    except RuntimeError as exc:
        stt_provider, stt_reason = settings.stt_provider, str(exc)
    speech_available = stt_provider != "none" and stt_reason is None
    result = {
        "ocr": {
            "primary": settings.ocr_primary_provider,
            "secondary": settings.ocr_secondary_provider,
            "languages": "ben+eng",
            "providers": providers,
            "available": any(item["available"] for item in providers.values()),
        },
        "video": {
            "ffmpeg": ffmpeg,
            "ffprobe": ffprobe,
            "opencv": opencv,
            "frame_extraction_available": bool(ffprobe["available"] and opencv["available"]),
            "audio_extraction_available": bool(ffmpeg["available"]),
        },
        "speech": {
            "provider": stt_provider,
            "available": speech_available,
            "reason": stt_reason
            or ("Speech transcription is disabled (STT_PROVIDER=none)." if not speech_available else None),
        },
    }

    for name, provider in providers.items():
        if provider["available"]:
            logger.info("Media diagnostic: OCR provider %s available", name)
        else:
            logger.warning("Media diagnostic: OCR provider %s unavailable: %s", name, provider["reason"])
    for name, info in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)):
        if info["available"]:
            logger.info("Media diagnostic: %s available at %s (%s)", name, info["path"], info["version"])
        else:
            logger.warning("Media diagnostic: %s unavailable: %s", name, info["reason"])
    logger.info("Media diagnostic: OpenCV %s", cv2.__version__)
    if speech_available:
        logger.info("Media diagnostic: speech provider %s available", stt_provider)
    else:
        logger.info("Media diagnostic: speech provider unavailable: %s", result["speech"]["reason"])
    return result
