"""End-to-end job pipeline with v2 extraction + research agent, using fakes (no network)."""
from functools import partial

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.database import Base
from app.jobs import pipeline
from app.jobs.stages import ANALYZE_STAGES, RESEARCH_STAGES, empty_stages
from app.models import (
    AdExtractionRecord,
    Advertisement,
    ExtractedClaim,
    ExtractedProduct,
    ResearchDossierRecord,
    ResearchJob,
    ResearchReport,
    UploadedFile,
    User,
)
from app.schemas.research import ResearchDossierV2
from app.services.ocr.base import OcrOutput
from app.services.research_agent import runner
from app.services.verification.handoff import build_verifier_inputs
from tests.test_research_agent_v2 import AD_TEXT, FAKE_LITERATURE, FakeLlm, FakeSearch, _budget, fake_fetch


@pytest.fixture
def db_session(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr(pipeline, "SessionLocal", Session)
    return Session


def _seed(Session):
    db = Session()
    user = User(email="t@example.com", hashed_password="x", full_name="T")
    db.add(user)
    db.flush()
    ad = Advertisement(user_id=user.id, original_filename="ad.png", media_type="image", content_type="image/png")
    db.add(ad)
    db.flush()
    db.add(UploadedFile(advertisement_id=ad.id, storage_key="uploads/x/ad.png", original_filename="ad.png",
                        content_type="image/png", size_bytes=1, role="original"))
    job = ResearchJob(advertisement_id=ad.id, job_type="analyze", status="queued", stages=empty_stages(ANALYZE_STAGES))
    db.add(job)
    db.commit()
    return ad.id, job.id


def _settings(**kw):
    return Settings(_env_file=None, **kw)


def test_v2_pipeline_end_to_end(db_session, monkeypatch):
    settings = _settings(extraction_backend="grounded_llm", research_agent="v2", llm_api_key="fake")
    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)
    monkeypatch.setattr("app.services.pipeline_v2.get_settings", lambda: settings)
    monkeypatch.setattr("app.services.pipeline_v2.llm_from_settings", lambda: FakeLlm())
    monkeypatch.setattr(pipeline, "run_ocr_on_image", lambda path, compare=True: [
        OcrOutput(provider="tesseract", raw_text=AD_TEXT, cleaned_text=AD_TEXT, average_confidence=0.8)])
    monkeypatch.setattr(runner, "run_research", partial(runner.run_research, llm=FakeLlm(), search=FakeSearch(),
                                                        fetcher=fake_fetch, literature=FAKE_LITERATURE,
                                                        budget=_budget(), use_cache=False))
    ad_id, job_id = _seed(db_session)

    pipeline.run_analyze_job(job_id)
    db = db_session()
    job = db.get(ResearchJob, job_id)
    assert job.status == "completed", job.error_message
    assert db.query(AdExtractionRecord).filter_by(advertisement_id=ad_id).one().extractor == "grounded_llm"
    assert db.query(ExtractedProduct).filter_by(advertisement_id=ad_id).one().product_name == "ডায়বিডেক্স"
    assert db.query(ExtractedClaim).filter_by(advertisement_id=ad_id).count() == 2

    rjob = ResearchJob(advertisement_id=ad_id, job_type="research", status="queued", stages=empty_stages(RESEARCH_STAGES))
    db.add(rjob)
    db.commit()
    pipeline.run_research_job(rjob.id)
    db.expire_all()
    assert db.get(ResearchJob, rjob.id).status == "completed", db.get(ResearchJob, rjob.id).error_message
    record = db.query(ResearchDossierRecord).filter_by(advertisement_id=ad_id).one()
    dossier = ResearchDossierV2.model_validate(record.payload)
    assert dossier.product.identity.status.value == "VERIFIED"
    assert len(build_verifier_inputs(dossier)) == 2  # one verifier input per ad claim
    # the existing report still works, fed from the v2 dossier
    report = db.query(ResearchReport).filter_by(advertisement_id=ad_id).one()
    assert report.dossier["product_identity"]["identity_status"] == "verified"
    assert [i["name"] for i in report.dossier["composition"]["ingredients"]] == [
        "Syzygium cumini seed extract", "Gymnema sylvestre leaf extract"]


def test_legacy_flow_unchanged_by_default(db_session, monkeypatch):
    settings = _settings()
    assert settings.extraction_backend == "baseline" and settings.research_agent == "legacy"
    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)
    monkeypatch.setattr("app.services.pipeline_v2.get_settings", lambda: settings)
    monkeypatch.setattr(pipeline, "run_ocr_on_image", lambda path, compare=True: [
        OcrOutput(provider="tesseract", raw_text=AD_TEXT, cleaned_text=AD_TEXT, average_confidence=0.8)])
    ad_id, job_id = _seed(db_session)
    pipeline.run_analyze_job(job_id)
    db = db_session()
    assert db.get(ResearchJob, job_id).status == "completed"
    assert db.query(AdExtractionRecord).filter_by(advertisement_id=ad_id).one().extractor == "regex_baseline"
    assert db.query(ExtractedProduct).filter_by(advertisement_id=ad_id).one().extra["backend"] == "baseline"


def test_legacy_report_shows_ingredient_and_claim_research():
    from app.services.pipeline_v2 import to_legacy_dossier
    from tests.test_research_agent_v2 import _full_run
    legacy = to_legacy_dossier(_full_run())
    syz = next(i for i in legacy.ingredient_research if "Syzygium" in i.ingredient_name)
    assert syz.human_studies_summary and syz.side_effects == "mild GI discomfort"
    assert all(e.evidence_scope == "ingredient" for e in legacy.scientific_evidence)
    assert legacy.effectiveness_research and "No verdict" in legacy.effectiveness_research[0].summary
    assert legacy.effectiveness_research[0].caveat.startswith("Ingredient-level")
