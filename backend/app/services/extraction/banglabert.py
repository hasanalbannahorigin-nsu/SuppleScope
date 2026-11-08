from __future__ import annotations

import logging

from app.config import get_settings
from app.services.extraction.baseline import ExtractionResult, extract_from_text

logger = logging.getLogger(__name__)

_pipeline = None


def _load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    settings = get_settings()
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch  # noqa: F401
    except Exception as exc:
        logger.info("BanglaBERT dependencies missing: %s", exc)
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(settings.banglabert_model_name)
        model = AutoModel.from_pretrained(settings.banglabert_model_name)
        _pipeline = {"tokenizer": tokenizer, "model": model, "name": settings.banglabert_model_name}
        return _pipeline
    except Exception:
        logger.exception("Failed to load BanglaBERT encoder")
        return None


def extract_with_optional_banglabert(*, ocr_text: str, speech_text: str = "") -> ExtractionResult:
    """
    Working baseline plus a modular BanglaBERT encoder adapter.

    The encoder is not a custom-trained NER/verification model. When the
    Hugging Face model cannot be loaded, extraction still succeeds via baseline.
    """
    result = extract_from_text(ocr_text=ocr_text, speech_text=speech_text)
    settings = get_settings()
    if settings.extraction_backend != "banglabert":
        return result
    loaded = _load_pipeline()
    if not loaded:
        result.extra["banglabert"] = "not_loaded"
        return result
    # Encode combined text as a future fine-tuning hook; do not invent entities.
    text = (ocr_text or "")[:800]
    try:
        import torch

        tokens = loaded["tokenizer"](text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = loaded["model"](**tokens)
        pooled = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        result.extra["banglabert"] = {
            "model": loaded["name"],
            "embedding_dim": len(pooled) if isinstance(pooled, list) else None,
            "status": "encoder_only_not_finetuned",
        }
    except Exception:
        logger.exception("BanglaBERT encode failed")
        result.extra["banglabert"] = "encode_failed"
    return result
