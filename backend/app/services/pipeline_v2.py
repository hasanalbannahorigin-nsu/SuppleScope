"""Glue between the existing job pipeline and the v2 extraction / research agent.

Keeps the old flow working: with EXTRACTION_BACKEND=baseline and RESEARCH_AGENT=legacy
nothing changes. The v2 research dossier is also converted into the legacy dossier so
the existing report page keeps showing product identity, composition and sources.
"""
from __future__ import annotations

import logging

from app.config import get_settings
from app.schemas.dossier import (
    CompositionBlock,
    DossierIngredient,
    ProductIdentity as LegacyIdentity,
    ResearchDossier,
    SourceRef,
)
from app.schemas.research.ad_extraction import AdExtraction
from app.schemas.research.common import InformationClass
from app.schemas.research.dossier import ResearchDossierV2
from app.schemas.research.ocr import OcrDocument
from app.schemas.research.product import IdentityStatus
from app.services.extraction.grounded import GroundedExtractor, baseline_to_ad_extraction
from app.services.llm.client import llm_from_settings

logger = logging.getLogger(__name__)


def extract_ad(doc: OcrDocument, baseline_result) -> tuple[AdExtraction, str | None]:
    """Returns (extraction, note). Uses the grounded LLM extractor when configured,
    otherwise (or if it fails) wraps the regex baseline."""
    if get_settings().extraction_backend == "grounded_llm":
        llm = llm_from_settings()
        if llm is None:
            return baseline_to_ad_extraction(doc, baseline_result), "grounded_llm needs LLM_API_KEY; used regex baseline"
        try:
            return GroundedExtractor(llm).run(doc).extraction, None
        except Exception as exc:  # network / quota / invalid JSON
            logger.exception("grounded extraction failed")
            return baseline_to_ad_extraction(doc, baseline_result), f"grounded_llm failed ({exc}); used regex baseline"
    return baseline_to_ad_extraction(doc, baseline_result), None


_LEGACY_STATUS = {
    InformationClass.OFFICIAL_LABEL: "LABEL_REPORTED",
    InformationClass.OFFICIAL_MANUFACTURER: "OFFICIALLY_VERIFIED",
    InformationClass.REGULATORY: "OFFICIALLY_VERIFIED",
    InformationClass.THIRD_PARTY: "THIRD_PARTY_REPORTED",
}
_LEGACY_IDENTITY = {
    IdentityStatus.VERIFIED: "verified",
    IdentityStatus.PROBABLE: "uncertain",
    IdentityStatus.IDENTITY_UNCERTAIN: "uncertain",
    IdentityStatus.NOT_FOUND: "not_found",
}


def to_legacy_dossier(d: ResearchDossierV2) -> ResearchDossier:
    ident = d.product.identity
    chosen = next((c for c in ident.candidates if c.candidate_id == ident.chosen_candidate_id), None)
    findings = {f.finding_id: f for f in d.findings}
    sources = {s.source_id: s for s in d.sources}

    def first(ids: list[str]) -> str | None:
        return findings[ids[0]].value if ids else None

    ingredients = []
    for ci in d.product.composition.ingredients:
        f = findings[ci.finding_ids[0]]
        src = sources.get(f.source_id)
        ingredients.append(DossierIngredient(
            name=ci.name, scientific_name=ci.standardized_name,
            amount=float(ci.amount) if ci.amount and ci.amount.replace(".", "", 1).isdigit() else None,
            unit=ci.unit, source_url=src.url if src else None, source_type=src.source_type.value if src else None,
            verification_status=_LEGACY_STATUS.get(f.information_class, "UNVERIFIED"),
        ))
    bm = d.product.brand_manufacturer
    return ResearchDossier(
        product_identity=LegacyIdentity(
            product_name=chosen.name if chosen else (d.advertisement.product_name.value if d.advertisement.product_name else None),
            brand=chosen.brand if chosen else None,
            manufacturer=first(bm.manufacturer_finding_ids) or (chosen.manufacturer if chosen else None),
            official_website=bm.official_website,
            formulation=chosen.form if chosen else None,
            description=first(d.product.composition.manufacturer_description_finding_ids),
            identity_status=_LEGACY_IDENTITY[ident.status],
            notes=f"[{ident.status.value}] {ident.reasoning}",
        ),
        composition=CompositionBlock(ingredients=ingredients, composition_status=d.product.composition.status.value.lower(),
                                     notes="From research agent v2; see the v2 dossier for passages and validation."),
        product_information={
            "dosage": [findings[f].value for f in d.product.composition.dosage_directions_finding_ids],
            "intended_purpose": [findings[f].value for f in d.product.composition.intended_purpose_finding_ids],
            "label_warnings": [findings[f].value for f in d.product.composition.label_warning_finding_ids],
            "ad_vs_official": [x.description for x in d.product.ad_vs_official],
            "research_dossier_v2_id": d.dossier_id,
        },
        missing_information=[f"{m.item}: {m.status.value}" for m in d.missing_information],
        uncertainties=d.uncertainties,
        sources=[SourceRef(title=s.title, url=s.url, source_type=s.source_type.value, publisher=s.site_or_publisher,
                           retrieved_for="product research v2") for s in d.sources],
    )
