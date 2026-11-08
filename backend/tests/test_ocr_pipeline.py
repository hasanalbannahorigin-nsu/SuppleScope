from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.ocr import pipeline


class UnavailableProvider:
    def __init__(self, name, reason):
        self.name = name
        self.reason = reason

    def unavailable_reason(self):
        return self.reason


def test_pipeline_reports_missing_local_language_and_cloud_credentials(monkeypatch):
    settings = SimpleNamespace(
        ocr_primary_provider="tesseract",
        ocr_secondary_provider="google_vision",
    )
    providers = {
        "tesseract": UnavailableProvider("tesseract", "Tesseract Bengali data ben is missing."),
        "google_vision": UnavailableProvider("google_vision", "GOOGLE_APPLICATION_CREDENTIALS is missing."),
    }
    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)
    monkeypatch.setattr(pipeline, "get_provider", lambda name: providers[name])

    with pytest.raises(RuntimeError) as error:
        pipeline.run_ocr_on_image(Path("sample.png"))

    assert "Tesseract Bengali data ben is missing" in str(error.value)
    assert "GOOGLE_APPLICATION_CREDENTIALS is missing" in str(error.value)
