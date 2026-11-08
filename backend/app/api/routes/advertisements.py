from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import get_owned_advertisement
from app.database import get_db
from app.jobs.pipeline import run_analyze_job, run_research_job
from app.jobs.stages import ALL_STAGES, ANALYZE_STAGES, RESEARCH_STAGES, empty_stages
from app.models.advertisement import Advertisement
from app.models.extraction import ExtractedClaim, ExtractedProduct, MentionedIngredient
from app.models.media import ExtractedFrame, OcrResult, SpeechTranscript, UploadedFile
from app.models.pipeline_v2 import AdExtractionRecord, ResearchDossierRecord
from app.models.research import (
    DiseaseRiskResearch,
    InteractionResearch,
    PatientContext,
    ResearchJob,
    ResearchReport,
)
from app.models.user import User
from app.security import get_current_user
from app.services.files import media_type_for, suffix_of, validate_upload
from app.services.pdf_report import dossier_to_pdf
from app.services.storage import storage
from app.utils.background import spawn

router = APIRouter(prefix="/api/advertisements", tags=["advertisements"])


class PatientContextIn(BaseModel):
    age: int | None = Field(default=None, ge=0, le=120)
    diseases: str | None = None
    medications: str | None = None
    health_context: str | None = None


def _serialize_ad(ad: Advertisement) -> dict:
    return {
        "id": ad.id,
        "original_filename": ad.original_filename,
        "media_type": ad.media_type,
        "status": ad.status,
        "created_at": ad.created_at.isoformat() if ad.created_at else None,
    }


@router.post("/upload")
def upload_advertisement(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    validate_upload(file)
    filename = file.filename or "upload"
    media_type = media_type_for(filename)
    key, size, checksum = storage.save_upload(
        file, folder=f"uploads/{user.id}", suffix=suffix_of(filename)
    )
    ad = Advertisement(
        user_id=user.id,
        original_filename=filename,
        media_type=media_type,
        content_type=file.content_type or "application/octet-stream",
        status="uploaded",
        title=filename,
    )
    db.add(ad)
    db.flush()
    db.add(
        UploadedFile(
            advertisement_id=ad.id,
            storage_key=key,
            original_filename=filename,
            content_type=ad.content_type,
            size_bytes=size,
            checksum_sha256=checksum,
            role="original",
        )
    )
    db.commit()
    db.refresh(ad)
    return _serialize_ad(ad)


@router.get("")
def list_advertisements(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> list[dict]:
    rows = (
        db.query(Advertisement)
        .filter(Advertisement.user_id == user.id)
        .order_by(Advertisement.created_at.desc())
        .all()
    )
    return [_serialize_ad(row) for row in rows]


@router.post("/{advertisement_id}/analyze")
def analyze(
    advertisement_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    ad = get_owned_advertisement(db, user, advertisement_id)
    existing = (
        db.query(ResearchJob)
        .filter_by(advertisement_id=ad.id, job_type="analyze")
        .order_by(ResearchJob.created_at.desc())
        .first()
    )
    if existing and existing.status in {"queued", "running"}:
        return {"job_id": existing.id, "status": existing.status}
    job = ResearchJob(
        advertisement_id=ad.id,
        job_type="analyze",
        status="queued",
        current_stage="advertisement_uploaded",
        stages=empty_stages(ANALYZE_STAGES),
    )
    db.add(job)
    ad.status = "analyzing"
    db.add(ad)
    db.commit()
    db.refresh(job)
    spawn(run_analyze_job, job.id)
    return {"job_id": job.id, "status": job.status}


@router.get("/{advertisement_id}/status")
def status(
    advertisement_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    ad = get_owned_advertisement(db, user, advertisement_id)
    jobs = db.query(ResearchJob).filter_by(advertisement_id=ad.id).all()
    merged: dict[str, dict] = {key: {"key": key, "status": "not_started", "detail": None} for key in ALL_STAGES}
    merged["advertisement_uploaded"] = {
        "key": "advertisement_uploaded",
        "status": "completed",
        "detail": "Stored in object storage",
    }
    for job in jobs:
        for stage in job.stages or []:
            merged[stage["key"]] = stage
    analyze_job = next((j for j in jobs if j.job_type == "analyze"), None)
    research_job = next((j for j in jobs if j.job_type == "research"), None)
    return {
        "advertisement_id": ad.id,
        "advertisement_status": ad.status,
        "analyze_job": {
            "id": analyze_job.id if analyze_job else None,
            "status": analyze_job.status if analyze_job else "not_started",
            "error": analyze_job.error_message if analyze_job else None,
        },
        "research_job": {
            "id": research_job.id if research_job else None,
            "status": research_job.status if research_job else "not_started",
            "error": research_job.error_message if research_job else None,
        },
        "stages": [merged[key] for key in ALL_STAGES],
    }


@router.get("/{advertisement_id}/ocr")
def ocr(
    advertisement_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict]:
    ad = get_owned_advertisement(db, user, advertisement_id)
    rows = db.query(OcrResult).filter_by(advertisement_id=ad.id).all()
    frames_by_id = {
        frame.id: frame
        for frame in db.query(ExtractedFrame).filter_by(advertisement_id=ad.id).all()
    }
    return [
        {
            "id": row.id,
            "provider": row.provider,
            "raw_text": row.raw_text,
            "cleaned_text": row.cleaned_text,
            "average_confidence": row.average_confidence,
            "bounding_boxes": row.bounding_boxes,
            "frame_id": row.frame_id,
            "frame_index": frames_by_id[row.frame_id].frame_index
            if row.frame_id in frames_by_id
            else None,
            "timestamp_ms": frames_by_id[row.frame_id].timestamp_ms
            if row.frame_id in frames_by_id
            else None,
            "source_url": (
                f"/api/advertisements/{ad.id}/frames/{row.frame_id}/image"
                if row.frame_id
                else f"/api/advertisements/{ad.id}/original"
            ),
        }
        for row in rows
    ]


@router.get("/{advertisement_id}/frames")
def frames(
    advertisement_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict]:
    ad = get_owned_advertisement(db, user, advertisement_id)
    rows = (
        db.query(ExtractedFrame)
        .filter_by(advertisement_id=ad.id)
        .order_by(ExtractedFrame.timestamp_ms)
        .all()
    )
    return [
        {
            "id": row.id,
            "frame_index": row.frame_index,
            "timestamp_ms": row.timestamp_ms,
            "scene_score": row.scene_score,
            "selection_reason": row.selection_reason,
            "url": f"/api/advertisements/{ad.id}/frames/{row.id}/image",
        }
        for row in rows
    ]


@router.get("/{advertisement_id}/frames/{frame_id}/image")
def frame_image(
    advertisement_id: str,
    frame_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FileResponse:
    ad = get_owned_advertisement(db, user, advertisement_id)
    frame = db.get(ExtractedFrame, frame_id)
    if frame is None or frame.advertisement_id != ad.id:
        raise HTTPException(status_code=404, detail="Frame not found")
    path = storage.absolute_path(frame.storage_key)
    return FileResponse(path, media_type="image/jpeg")


@router.get("/{advertisement_id}/original")
def original_media(
    advertisement_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ad = get_owned_advertisement(db, user, advertisement_id)
    original = next((f for f in ad.files if f.role == "original"), None)
    if original is None:
        raise HTTPException(status_code=404, detail="File not found")
    path = storage.absolute_path(original.storage_key)
    return FileResponse(path, media_type=original.content_type, filename=original.original_filename)


@router.get("/{advertisement_id}/extraction")
def extraction(
    advertisement_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    ad = get_owned_advertisement(db, user, advertisement_id)
    products = db.query(ExtractedProduct).filter_by(advertisement_id=ad.id).all()
    claims = db.query(ExtractedClaim).filter_by(advertisement_id=ad.id).all()
    ingredients = db.query(MentionedIngredient).filter_by(advertisement_id=ad.id).all()
    transcripts = db.query(SpeechTranscript).filter_by(advertisement_id=ad.id).all()
    return {
        "products": [
            {
                "product_name": p.product_name,
                "brand": p.brand,
                "manufacturer": p.manufacturer,
                "product_category": p.product_category,
                "extra": p.extra,
            }
            for p in products
        ],
        "claims": [
            {
                "claim_text_original": c.claim_text_original,
                "claim_text_en": c.claim_text_en,
                "claim_type": c.claim_type,
                "source": c.source,
            }
            for c in claims
        ],
        "mentioned_ingredients": [{"name": i.name, "original_text": i.original_text, "source": i.source} for i in ingredients],
        "transcripts": [
            {"provider": t.provider, "language": t.language, "full_text": t.full_text, "segments": t.segments}
            for t in transcripts
        ],
    }


@router.get("/{advertisement_id}/ad-extraction")
def ad_extraction_v2(
    advertisement_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """AdExtraction (schemas v2): only what the ad shows, with spans."""
    ad = get_owned_advertisement(db, user, advertisement_id)
    row = (db.query(AdExtractionRecord).filter_by(advertisement_id=ad.id)
           .order_by(AdExtractionRecord.created_at.desc()).first())
    if row is None:
        raise HTTPException(status_code=404, detail="No v2 ad extraction yet; run analysis first")
    return row.payload


def _latest_dossier_v2(db: Session, ad_id: str) -> ResearchDossierRecord:
    row = (db.query(ResearchDossierRecord).filter_by(advertisement_id=ad_id)
           .order_by(ResearchDossierRecord.created_at.desc()).first())
    if row is None:
        raise HTTPException(status_code=404, detail="No v2 research dossier yet; set RESEARCH_AGENT=v2 and run research")
    return row


@router.get("/{advertisement_id}/dossier-v2")
def research_dossier_v2(
    advertisement_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """The complete ResearchDossierV2 written by the research agent (no verdicts)."""
    ad = get_owned_advertisement(db, user, advertisement_id)
    return _latest_dossier_v2(db, ad.id).payload


@router.get("/{advertisement_id}/verifier-input.jsonl")
def verifier_input_jsonl(
    advertisement_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    """One VerifierInput per claim, as JSON Lines, for the separate custom Verification Model."""
    import json

    from app.schemas.research.dossier import ResearchDossierV2
    from app.services.verification.handoff import build_verifier_inputs

    ad = get_owned_advertisement(db, user, advertisement_id)
    dossier = ResearchDossierV2.model_validate(_latest_dossier_v2(db, ad.id).payload)
    body = "".join(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n" for r in build_verifier_inputs(dossier))
    return Response(content=body, media_type="application/x-ndjson",
                    headers={"Content-Disposition": f'attachment; filename="verifier_input_{ad.id}.jsonl"'})


@router.post("/{advertisement_id}/research")
def start_research(
    advertisement_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    ad = get_owned_advertisement(db, user, advertisement_id)
    existing = (
        db.query(ResearchJob)
        .filter_by(advertisement_id=ad.id, job_type="research")
        .order_by(ResearchJob.created_at.desc())
        .first()
    )
    if existing and existing.status in {"queued", "running"}:
        return {"job_id": existing.id, "status": existing.status}
    job = ResearchJob(
        advertisement_id=ad.id,
        job_type="research",
        status="queued",
        current_stage="product_identification_started",
        stages=empty_stages(RESEARCH_STAGES),
    )
    db.add(job)
    ad.status = "researching"
    db.add(ad)
    db.commit()
    db.refresh(job)
    spawn(run_research_job, job.id)
    return {"job_id": job.id, "status": job.status}


@router.get("/{advertisement_id}/research-status")
def research_status(
    advertisement_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    payload = status(advertisement_id, user, db)
    return payload["research_job"] | {"stages": [s for s in payload["stages"] if s["key"] in RESEARCH_STAGES]}


@router.get("/{advertisement_id}/research-report")
def research_report(
    advertisement_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    ad = get_owned_advertisement(db, user, advertisement_id)
    report = db.query(ResearchReport).filter_by(advertisement_id=ad.id).one_or_none()
    if report is None:
        raise HTTPException(status_code=404, detail="Research report not generated yet")
    return {
        "advertisement_id": ad.id,
        "dossier": report.dossier,
        "missing_information": report.missing_information,
        "uncertainties": report.uncertainties,
        "updated_at": report.updated_at.isoformat() if report.updated_at else None,
    }


@router.get("/{advertisement_id}/research-report.pdf")
def research_report_pdf(
    advertisement_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    ad = get_owned_advertisement(db, user, advertisement_id)
    report = db.query(ResearchReport).filter_by(advertisement_id=ad.id).one_or_none()
    if report is None:
        raise HTTPException(status_code=404, detail="Research report not generated yet")
    data = dossier_to_pdf(ad.original_filename, report.dossier or {})
    return Response(content=data, media_type="application/pdf")


@router.post("/{advertisement_id}/patient-context")
def save_patient_context(
    advertisement_id: str,
    payload: PatientContextIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    ad = get_owned_advertisement(db, user, advertisement_id)
    row = db.query(PatientContext).filter_by(advertisement_id=ad.id).one_or_none()
    if row is None:
        row = PatientContext(advertisement_id=ad.id)
    row.age = payload.age
    row.diseases = payload.diseases
    row.medications = payload.medications
    row.health_context = payload.health_context
    db.add(row)
    db.commit()
    return payload.model_dump()


@router.get("/{advertisement_id}/patient-context")
def get_patient_context(
    advertisement_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    ad = get_owned_advertisement(db, user, advertisement_id)
    row = db.query(PatientContext).filter_by(advertisement_id=ad.id).one_or_none()
    if row is None:
        return {"age": None, "diseases": None, "medications": None, "health_context": None}
    return {
        "age": row.age,
        "diseases": row.diseases,
        "medications": row.medications,
        "health_context": row.health_context,
    }


@router.get("/{advertisement_id}/risk-research")
def risk_research(
    advertisement_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    ad = get_owned_advertisement(db, user, advertisement_id)
    interactions = db.query(InteractionResearch).filter_by(advertisement_id=ad.id).all()
    diseases = db.query(DiseaseRiskResearch).filter_by(advertisement_id=ad.id).all()
    return {
        "disclaimer": "Not a diagnosis or prescription. Custom verification model is not integrated.",
        "supplement_drug_interactions": [
            {
                "ingredient": i.ingredient,
                "medicine": i.medicine,
                "interaction_information": i.interaction_information,
                "possible_concerns": i.possible_concerns,
                "evidence": i.evidence,
                "uncertainty": i.uncertainty,
                "source_references": i.source_references,
            }
            for i in interactions
        ],
        "supplement_disease_risks": [
            {
                "ingredient": d.ingredient,
                "disease": d.disease,
                "available_evidence": d.available_evidence,
                "potential_concern": d.potential_concern,
                "limitations": d.limitations,
                "source_references": d.source_references,
            }
            for d in diseases
        ],
    }
