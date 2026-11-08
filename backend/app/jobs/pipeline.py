from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from app.agents.graph import llm_configured, run_research_agent
from app.database import SessionLocal
from app.jobs.stages import ANALYZE_STAGES, RESEARCH_STAGES, empty_stages, set_stage
from app.models.advertisement import Advertisement
from app.models.extraction import ExtractedClaim, ExtractedProduct, MentionedIngredient
from app.models.media import ExtractedFrame, OcrResult, SpeechTranscript, UploadedFile
from app.models.research import (
    DiseaseRiskResearch,
    EvidencePassage,
    IngredientResearch,
    InteractionResearch,
    ProductComposition,
    ProductCompositionIngredient,
    ProductIdentification,
    ResearchJob,
    ResearchReport,
    ResearchSource,
    ScientificPaper,
)
from app.schemas.dossier import ResearchDossier
from app.services.extraction.banglabert import extract_with_optional_banglabert
from app.services.ocr.pipeline import run_ocr_on_image
from app.services.research.fallback import run_scripted_research
from app.services.speech import transcribe_audio, transcription_status
from app.services.storage import storage
from app.services.video import extract_audio, extract_informative_frames, write_frame_jpeg

logger = logging.getLogger(__name__)


def _save_job(db: Session, job: ResearchJob, **kwargs) -> None:
    for key, value in kwargs.items():
        setattr(job, key, value)
    db.add(job)
    db.commit()
    db.refresh(job)


def run_analyze_job(job_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.get(ResearchJob, job_id)
        if job is None:
            return
        ad = db.get(Advertisement, job.advertisement_id)
        if ad is None:
            _save_job(db, job, status="failed", error_message="Advertisement missing")
            return
        stages = job.stages or empty_stages(ANALYZE_STAGES)
        _save_job(db, job, status="running", current_stage="advertisement_uploaded", stages=set_stage(stages, "advertisement_uploaded", "completed", "File stored"))
        original = next((f for f in ad.files if f.role == "original"), None)
        if original is None:
            ad.status = "failed"
            _save_job(db, job, status="failed", error_message="Original file missing")
            return
        source_path = storage.absolute_path(original.storage_key)

        if ad.media_type == "video":
            stages = set_stage(job.stages, "video_preprocessing", "in_progress")
            _save_job(db, job, current_stage="video_preprocessing", stages=stages)
            try:
                meta, frames = extract_informative_frames(source_path)
                stages = set_stage(
                    job.stages,
                    "video_preprocessing",
                    "completed",
                    f"duration_ms={meta.duration_ms} fps={meta.fps:.2f}",
                )
                stages = set_stage(stages, "frames_extracted", "in_progress")
                _save_job(db, job, stages=stages, current_stage="frames_extracted")
                for frame in frames:
                    dest = storage.root / "frames" / ad.id / f"{frame.frame_index:03d}.jpg"
                    write_frame_jpeg(frame.image, dest)
                    rel = dest.relative_to(storage.root).as_posix()
                    db.add(
                        ExtractedFrame(
                            advertisement_id=ad.id,
                            frame_index=frame.frame_index,
                            timestamp_ms=frame.timestamp_ms,
                            storage_key=rel,
                            scene_score=frame.scene_score,
                            selection_reason=frame.selection_reason,
                        )
                    )
                db.commit()
                stages = set_stage(
                    job.stages, "frames_extracted", "completed", f"{len(frames)} informative frames"
                )
                _save_job(db, job, stages=stages)
            except Exception as exc:
                logger.exception("video processing failed")
                ad.status = "failed"
                stages = set_stage(job.stages, "video_preprocessing", "failed", str(exc))
                _save_job(db, job, status="failed", error_message=str(exc), stages=stages)
                return
        else:
            stages = set_stage(job.stages, "video_preprocessing", "skipped", "Image upload")
            stages = set_stage(stages, "frames_extracted", "skipped", "Image upload")
            _save_job(db, job, stages=stages)

        stages = set_stage(job.stages, "bangla_ocr_completed", "in_progress")
        _save_job(db, job, current_stage="bangla_ocr_completed", stages=stages)
        try:
            targets: list[tuple[Path, str | None]] = []
            if ad.media_type == "image":
                targets.append((source_path, None))
            else:
                for frame in db.query(ExtractedFrame).filter_by(advertisement_id=ad.id).all():
                    targets.append((storage.absolute_path(frame.storage_key), frame.id))
            for path, frame_id in targets:
                outputs = run_ocr_on_image(path, compare=True)
                for output in outputs:
                    db.add(
                        OcrResult(
                            advertisement_id=ad.id,
                            frame_id=frame_id,
                            provider=output.provider,
                            raw_text=output.raw_text,
                            cleaned_text=output.cleaned_text,
                            average_confidence=output.average_confidence,
                            bounding_boxes=output.bounding_boxes,
                            language_hints=output.language_hints,
                            extra=output.extra,
                        )
                    )
            db.commit()
            stages = set_stage(job.stages, "bangla_ocr_completed", "completed", f"{len(targets)} image(s)")
            _save_job(db, job, stages=stages)
        except Exception as exc:
            logger.exception("OCR failed")
            ad.status = "failed"
            stages = set_stage(job.stages, "bangla_ocr_completed", "failed", str(exc))
            _save_job(db, job, status="failed", error_message=str(exc), stages=stages)
            return

        if ad.media_type == "video":
            provider_name, unavailable_reason = transcription_status()
            if provider_name == "none":
                stages = set_stage(
                    job.stages,
                    "speech_transcription_completed",
                    "skipped",
                    "Speech transcription is not configured (STT_PROVIDER=none).",
                )
                _save_job(db, job, stages=stages)
            elif unavailable_reason:
                stages = set_stage(
                    job.stages,
                    "speech_transcription_completed",
                    "skipped",
                    unavailable_reason,
                )
                _save_job(db, job, stages=stages)
            else:
                stages = set_stage(job.stages, "speech_transcription_completed", "in_progress")
                _save_job(db, job, current_stage="speech_transcription_completed", stages=stages)
                try:
                    audio_path = storage.root / "audio" / f"{ad.id}.wav"
                    extract_audio(source_path, audio_path)
                    transcript = transcribe_audio(audio_path)
                    db.add(
                        SpeechTranscript(
                            advertisement_id=ad.id,
                            provider=transcript.provider,
                            language=transcript.language,
                            full_text=transcript.full_text,
                            segments=transcript.segments,
                        )
                    )
                    db.commit()
                    stages = set_stage(job.stages, "speech_transcription_completed", "completed")
                    _save_job(db, job, stages=stages)
                except Exception as exc:
                    logger.exception("speech failed")
                    stages = set_stage(
                        job.stages,
                        "speech_transcription_completed",
                        "failed",
                        str(exc),
                    )
                    _save_job(db, job, stages=stages)
        else:
            stages = set_stage(job.stages, "speech_transcription_completed", "skipped", "Image upload")
            _save_job(db, job, stages=stages)

        stages = set_stage(job.stages, "information_extraction_completed", "in_progress")
        _save_job(db, job, current_stage="information_extraction_completed", stages=stages)
        ocr_bits = [
            row.cleaned_text
            for row in db.query(OcrResult).filter_by(advertisement_id=ad.id).all()
            if row.provider == _preferred_ocr_provider(db, ad.id)
        ]
        if not ocr_bits:
            ocr_bits = [row.cleaned_text for row in db.query(OcrResult).filter_by(advertisement_id=ad.id).all()]
        speech_bits = [
            row.full_text for row in db.query(SpeechTranscript).filter_by(advertisement_id=ad.id).all()
        ]
        extracted = extract_with_optional_banglabert(
            ocr_text="\n".join(ocr_bits), speech_text="\n".join(speech_bits)
        )
        db.add(
            ExtractedProduct(
                advertisement_id=ad.id,
                product_name=extracted.product_name,
                brand=extracted.brand,
                manufacturer=extracted.manufacturer,
                product_category=extracted.product_category,
                source="combined",
                original_span=extracted.product_name,
                extra=extracted.extra,
            )
        )
        for claim in extracted.claims:
            db.add(
                ExtractedClaim(
                    advertisement_id=ad.id,
                    claim_text_original=claim["claim_text_original"],
                    claim_text_en=claim.get("claim_text_en"),
                    claim_type=claim.get("claim_type"),
                    source=claim.get("source") or "ocr",
                )
            )
        for ingredient in extracted.ingredients:
            db.add(
                MentionedIngredient(
                    advertisement_id=ad.id,
                    name=ingredient["name"],
                    original_text=ingredient.get("original_text"),
                    source=ingredient.get("source") or "ocr",
                )
            )
        db.commit()
        stages = set_stage(job.stages, "information_extraction_completed", "completed")
        ad.status = "analyzed"
        db.add(ad)
        _save_job(db, job, status="completed", current_stage="information_extraction_completed", stages=stages)
    except Exception as exc:
        logger.exception("analyze job crashed")
        job = db.get(ResearchJob, job_id)
        if job:
            ad = db.get(Advertisement, job.advertisement_id)
            if ad:
                ad.status = "failed"
            _save_job(db, job, status="failed", error_message=str(exc))
    finally:
        db.close()


def _preferred_ocr_provider(db: Session, advertisement_id: str) -> str:
    providers = [
        row[0]
        for row in db.query(OcrResult.provider).filter_by(advertisement_id=advertisement_id).distinct()
    ]
    if "google_vision" in providers:
        return "google_vision"
    if providers:
        return providers[0]
    return "tesseract"


def _persist_dossier(db: Session, ad: Advertisement, dossier: ResearchDossier) -> None:
    ident = db.query(ProductIdentification).filter_by(advertisement_id=ad.id).one_or_none()
    if ident is None:
        ident = ProductIdentification(advertisement_id=ad.id)
    ident.product_name = dossier.product_identity.product_name
    ident.brand = dossier.product_identity.brand
    ident.manufacturer = dossier.product_identity.manufacturer
    ident.product_category = dossier.product_identity.product_category
    ident.official_website = dossier.product_identity.official_website
    ident.formulation = dossier.product_identity.formulation
    ident.description = dossier.product_identity.description
    ident.identity_status = dossier.product_identity.identity_status
    ident.notes = dossier.product_identity.notes
    db.add(ident)

    composition = db.query(ProductComposition).filter_by(advertisement_id=ad.id).one_or_none()
    if composition is None:
        composition = ProductComposition(advertisement_id=ad.id)
    composition.composition_status = dossier.composition.composition_status
    composition.notes = dossier.composition.notes
    db.add(composition)
    db.flush()
    db.query(ProductCompositionIngredient).filter_by(composition_id=composition.id).delete()
    for item in dossier.composition.ingredients:
        db.add(
            ProductCompositionIngredient(
                composition_id=composition.id,
                name=item.name,
                alternative_names=item.alternative_names,
                scientific_name=item.scientific_name,
                amount=item.amount,
                unit=item.unit,
                source_url=item.source_url,
                source_type=item.source_type,
                verification_status=item.verification_status,
            )
        )

    db.query(ScientificPaper).filter_by(advertisement_id=ad.id).delete()
    db.query(EvidencePassage).filter_by(advertisement_id=ad.id).delete()
    db.query(ResearchSource).filter_by(advertisement_id=ad.id).delete()
    db.query(IngredientResearch).filter_by(advertisement_id=ad.id).delete()
    db.query(InteractionResearch).filter_by(advertisement_id=ad.id).delete()
    db.query(DiseaseRiskResearch).filter_by(advertisement_id=ad.id).delete()

    for paper in dossier.scientific_evidence:
        row = ScientificPaper(
            advertisement_id=ad.id,
            title=paper.title,
            authors=paper.authors,
            publication_year=paper.publication_year,
            doi=paper.doi,
            url=paper.url,
            study_type=paper.study_type,
            population=paper.population,
            intervention=paper.intervention,
            dosage=paper.dosage,
            outcomes=paper.outcomes,
            limitations=paper.limitations,
            source_api=paper.source_api,
            related_ingredient=paper.related_ingredient,
            evidence_scope=paper.evidence_scope,
        )
        db.add(row)
        db.flush()
        for passage in paper.evidence_passages:
            db.add(EvidencePassage(advertisement_id=ad.id, paper_id=row.id, text=passage, source_url=paper.url))

    for source in dossier.sources:
        db.add(
            ResearchSource(
                advertisement_id=ad.id,
                title=source.title,
                url=source.url,
                source_type=source.source_type,
                publisher=source.publisher,
                retrieved_for=source.retrieved_for,
            )
        )
    for item in dossier.ingredient_research:
        db.add(
            IngredientResearch(
                advertisement_id=ad.id,
                ingredient_name=item.ingredient_name,
                scientific_name=item.scientific_name,
                common_name=item.common_name,
                biological_properties=item.biological_properties,
                intended_uses=item.intended_uses,
                known_health_effects=item.known_health_effects,
                side_effects=item.side_effects,
                safety_concerns=item.safety_concerns,
                human_studies_summary=item.human_studies_summary,
                animal_studies_summary=item.animal_studies_summary,
                laboratory_studies_summary=item.laboratory_studies_summary,
                sources=[s.model_dump() for s in item.sources],
            )
        )
    for item in dossier.supplement_drug_interactions:
        db.add(
            InteractionResearch(
                advertisement_id=ad.id,
                ingredient=item.ingredient,
                medicine=item.medicine,
                interaction_information=item.interaction_information,
                possible_concerns=item.possible_concerns,
                evidence=item.evidence,
                uncertainty=item.uncertainty,
                source_references=[s.model_dump() for s in item.source_references],
            )
        )
    for item in dossier.supplement_disease_risks:
        db.add(
            DiseaseRiskResearch(
                advertisement_id=ad.id,
                ingredient=item.ingredient,
                disease=item.disease,
                available_evidence=item.available_evidence,
                potential_concern=item.potential_concern,
                limitations=item.limitations,
                source_references=[s.model_dump() for s in item.source_references],
            )
        )

    report = db.query(ResearchReport).filter_by(advertisement_id=ad.id).one_or_none()
    if report is None:
        report = ResearchReport(advertisement_id=ad.id, dossier={})
    report.dossier = dossier.model_dump()
    report.missing_information = dossier.missing_information
    report.uncertainties = dossier.uncertainties
    db.add(report)
    ad.status = "researched"
    db.add(ad)
    db.commit()


def run_research_job(job_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.get(ResearchJob, job_id)
        if job is None:
            return
        ad = db.get(Advertisement, job.advertisement_id)
        if ad is None:
            _save_job(db, job, status="failed", error_message="Advertisement missing")
            return
        stages = job.stages or empty_stages(RESEARCH_STAGES)
        _save_job(db, job, status="running", stages=set_stage(stages, "product_identification_started", "in_progress"), current_stage="product_identification_started")

        product = db.query(ExtractedProduct).filter_by(advertisement_id=ad.id).first()
        claims = db.query(ExtractedClaim).filter_by(advertisement_id=ad.id).all()
        ingredients = db.query(MentionedIngredient).filter_by(advertisement_id=ad.id).all()
        ocr = db.query(OcrResult).filter_by(advertisement_id=ad.id).all()
        transcripts = db.query(SpeechTranscript).filter_by(advertisement_id=ad.id).all()
        patient = ad.patient_context
        brief = {
            "product_name": product.product_name if product else None,
            "extracted_product": {
                "product_name": product.product_name if product else None,
                "brand": product.brand if product else None,
                "manufacturer": product.manufacturer if product else None,
                "category": product.product_category if product else None,
            },
            "claims": [c.claim_text_original for c in claims],
            "mentioned_ingredients": [{"name": i.name, "original_text": i.original_text} for i in ingredients],
            "ocr_text": "\n".join(r.cleaned_text for r in ocr if r.cleaned_text)[:8000],
            "speech_text": "\n".join(t.full_text for t in transcripts)[:4000],
            "patient_context": {
                "age": patient.age if patient else None,
                "diseases": patient.diseases if patient else None,
                "medications": patient.medications if patient else None,
                "health_context": patient.health_context if patient else None,
            },
            "instructions": "Do not invent ingredients or papers. Agent does not issue a verification verdict.",
        }

        try:
            if llm_configured():
                dossier = run_research_agent(brief)
            else:
                dossier = run_scripted_research(brief)
                stages = set_stage(
                    job.stages,
                    "product_identification_started",
                    "completed",
                    "LLM key missing; used live API scripted research (no fabricated citations)",
                )
                _save_job(db, job, stages=stages)
            for key in RESEARCH_STAGES[:-1]:
                stages = set_stage(job.stages, key, "completed")
            stages = set_stage(stages, "research_dossier_generated", "in_progress")
            _save_job(db, job, stages=stages, current_stage="research_dossier_generated")
            _persist_dossier(db, ad, dossier)
            stages = set_stage(job.stages, "research_dossier_generated", "completed")
            _save_job(db, job, status="completed", stages=stages, current_stage="research_dossier_generated")
        except Exception as exc:
            logger.exception("research failed")
            stages = set_stage(job.stages, job.current_stage or "product_identification_started", "failed", str(exc))
            _save_job(db, job, status="failed", error_message=str(exc), stages=stages)
    finally:
        db.close()
