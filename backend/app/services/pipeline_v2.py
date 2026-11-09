"""Glue between the existing job pipeline and the v2 extraction / research agent.

Keeps the old flow working: with EXTRACTION_BACKEND=baseline and RESEARCH_AGENT=legacy
nothing changes. The v2 research dossier is also converted into the legacy dossier so
the existing report page keeps showing product identity, composition and sources.
"""
from __future__ import annotations

import logging
from collections import Counter

from app.config import get_settings
from app.schemas.dossier import (
    CompositionBlock,
    DiseaseRiskItem,
    DossierIngredient,
    EffectivenessItem,
    IngredientResearchItem,
    InteractionItem,
    ProductIdentity as LegacyIdentity,
    ResearchDossier,
    ScientificEvidenceItem,
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
    lit = _legacy_literature(d)
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
        **lit,
        missing_information=[f"{m.item}: {m.status.value}" for m in d.missing_information],
        uncertainties=d.uncertainties,
        sources=[SourceRef(title=s.title, url=s.url, source_type=s.source_type.value, publisher=s.site_or_publisher,
                           retrieved_for="product research v2") for s in d.sources],
    )


def _legacy_literature(d: ResearchDossierV2) -> dict:
    """Ingredient studies, safety and per-claim evidence for the existing report page (milestone 2)."""
    sources = {s.source_id: s for s in d.sources}
    kept = {e.evidence_id: e for e in d.evidence if e.validation.passed}
    names = {i.ingredient_id: i.name for i in d.ingredients}

    def ref(source_id: str) -> SourceRef:
        s = sources[source_id]
        return SourceRef(title=s.title, url=s.url, source_type=s.source_type.value, publisher=s.journal or s.site_or_publisher,
                         retrieved_for="ingredient literature")

    def item(e) -> ScientificEvidenceItem:
        s = sources[e.source_id]
        return ScientificEvidenceItem(
            title=s.title or "(untitled)", authors=", ".join(s.authors[:4]) or None, publication_year=s.publication_year,
            doi=s.doi, url=s.url, study_type=e.study_type.value, population=e.population, intervention=e.intervention,
            dosage=e.dose, outcomes=", ".join(e.outcomes) or None, limitations="; ".join(e.limitations) or None,
            source_api=s.site_or_publisher, related_ingredient=names.get(e.subject_id),
            evidence_scope="ingredient" if e.scope.value == "INGREDIENT" else "product", evidence_passages=[e.passage])

    def summary(recs) -> str | None:
        parts = [r.result_summary for r in recs if r.result_summary][:3]
        return " ".join(parts) or None

    ingredient_research, side_effects, interactions, risks = [], [], [], []
    for ing in d.ingredients:
        recs = [kept[e] for e in ing.evidence_ids if e in kept]
        human = [r for r in recs if r.human_study]
        animal = [r for r in recs if r.study_type.value == "ANIMAL"]
        lab = [r for r in recs if r.study_type.value == "IN_VITRO"]
        sf = ing.safety
        ingredient_research.append(IngredientResearchItem(
            ingredient_name=ing.name, scientific_name=ing.scientific_name or ing.standardized_name,
            common_name=", ".join(ing.synonyms[:3]) or None,
            known_health_effects=", ".join(e.effect for e in ing.effects) or None,
            side_effects="; ".join(x.description for x in sf.side_effects) or None,
            safety_concerns="; ".join([x.description for x in sf.contraindications]
                                      + [f"{x.condition}: {x.description}" for x in sf.disease_cautions]) or None,
            human_studies_summary=summary(human), animal_studies_summary=summary(animal),
            laboratory_studies_summary=summary(lab),
            sources=[ref(r.source_id) for r in recs][:8],
        ))
        for x in sf.side_effects + sf.contraindications:
            side_effects.append({"ingredient": ing.name, "description": x.description,
                                 "sources": [sources[kept[e].source_id].url for e in x.evidence_ids if e in kept]})
        for x in sf.drug_interactions:
            interactions.append(InteractionItem(
                ingredient=ing.name, medicine=x.drug_or_class, interaction_information=x.description,
                evidence=", ".join(kept[e].study_type.value for e in x.evidence_ids if e in kept) or None,
                source_references=[ref(kept[e].source_id) for e in x.evidence_ids if e in kept]))
        for x in sf.disease_cautions:
            risks.append(DiseaseRiskItem(
                ingredient=ing.name, disease=x.condition, available_evidence=x.description,
                source_references=[ref(kept[e].source_id) for e in x.evidence_ids if e in kept]))

    effectiveness = []
    for c in d.claims:
        recs = [kept[link.evidence_id] for link in c.evidence_links if link.evidence_id in kept]
        rel = Counter(link.relation.value for link in c.evidence_links)
        product_level = any(link.scope.value == "FINISHED_PRODUCT" for link in c.evidence_links)
        effectiveness.append(EffectivenessItem(
            scope="product" if product_level else "ingredient",
            subject=c.claim_text_en or c.claim_text_original,
            summary=(f"Research status {c.status.value}; {len(c.evidence_links)} linked studies "
                     f"({rel.get('SUPPORTS', 0)} report the effect, {rel.get('CONTRADICTS', 0)} do not, "
                     f"{rel.get('CONTEXT', 0)} background). No verdict: see the verification model."),
            evidence=[item(r) for r in recs][:8],
            caveat=None if product_level else "Ingredient-level studies; they do not show that this product works.",
        ))
    return {
        "ingredient_research": ingredient_research,
        "scientific_evidence": [item(e) for e in kept.values()],
        "effectiveness_research": effectiveness,
        "side_effects": side_effects,
        "supplement_drug_interactions": interactions,
        "supplement_disease_risks": risks,
    }
