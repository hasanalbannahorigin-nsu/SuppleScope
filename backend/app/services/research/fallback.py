from __future__ import annotations

import logging

from app.schemas.dossier import (
    CompositionBlock,
    EffectivenessItem,
    ProductIdentity,
    ResearchDossier,
    ScientificEvidenceItem,
    SourceRef,
)
from app.services.research.clinicaltrials import search_clinical_trials
from app.services.research.crossref import search_crossref
from app.services.research.europepmc import search_europe_pmc
from app.services.research.openfda import search_openfda_label
from app.services.research.pubmed import search_pubmed
from app.services.research.websearch import search_web

logger = logging.getLogger(__name__)


def _paper_items(raw: list[dict], *, ingredient: str | None, scope: str) -> list[ScientificEvidenceItem]:
    items = []
    for row in raw:
        title = row.get("title")
        if not title:
            continue
        items.append(
            ScientificEvidenceItem(
                title=title,
                authors=row.get("authors"),
                publication_year=row.get("publication_year"),
                doi=row.get("doi"),
                url=row.get("url"),
                study_type=row.get("study_type"),
                outcomes=row.get("outcomes"),
                source_api=row.get("source_api"),
                related_ingredient=ingredient,
                evidence_scope=scope,  # type: ignore[arg-type]
                evidence_passages=row.get("evidence_passages") or [],
            )
        )
    return items


def run_scripted_research(brief: dict) -> ResearchDossier:
    """
    Deterministic tool pipeline used when no LLM key is configured.
    Still calls live APIs. Never fabricates papers or ingredients.
    """
    product_name = (
        (brief.get("extracted_product") or {}).get("product_name")
        or brief.get("product_name")
        or ""
    )
    ingredients = [i.get("name") for i in brief.get("mentioned_ingredients") or [] if i.get("name")]
    identity_status = "uncertain" if product_name else "not_found"
    missing = []
    uncertainties = [
        "Exact product identity was not confirmed by an official label in this non-LLM research pass.",
        "Ingredient lists are not inferred from similarly named products.",
    ]
    sources: list[SourceRef] = []
    evidence: list[ScientificEvidenceItem] = []

    queries = []
    if product_name:
        queries.append(f"{product_name} supplement ingredients label manufacturer")
    for name in ingredients[:6]:
        queries.append(name)

    web_hits = []
    for query in queries[:4]:
        try:
            web_hits.extend(search_web(query, max_results=4))
        except Exception:
            logger.exception("web search failed for %s", query)
    for hit in web_hits:
        if hit.get("url"):
            sources.append(
                SourceRef(
                    title=hit.get("title"),
                    url=hit.get("url"),
                    source_type="web_search",
                    publisher=hit.get("provider"),
                    retrieved_for=product_name or "product_search",
                )
            )

    product_papers: list[ScientificEvidenceItem] = []
    if product_name:
        try:
            product_papers = _paper_items(
                search_pubmed(f"{product_name} supplement", retmax=5),
                ingredient=None,
                scope="product",
            )
        except Exception:
            logger.exception("product pubmed failed")
        evidence.extend(product_papers)

    ingredient_research = []
    effectiveness = []
    if product_papers:
        effectiveness.append(
            EffectivenessItem(
                scope="product",
                subject=product_name,
                summary="Product-name literature hits only. These do not confirm that this exact SKU was studied.",
                evidence=product_papers,
                caveat="Product-name matches can refer to different formulations.",
            )
        )
    else:
        missing.append("No product-specific PubMed records were retrieved for the extracted name.")

    for name in ingredients[:5]:
        papers = []
        for searcher in (
            lambda q: search_pubmed(q, retmax=5),
            lambda q: search_europe_pmc(q, page_size=5),
            lambda q: search_crossref(q, rows=4),
        ):
            try:
                papers.extend(_paper_items(searcher(name), ingredient=name, scope="ingredient"))
            except Exception:
                logger.exception("literature search failed for %s", name)
        # de-dupe by doi/title
        seen = set()
        unique = []
        for paper in papers:
            key = (paper.doi or paper.title).lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(paper)
        evidence.extend(unique)
        ingredient_research.append(
            {
                "ingredient_name": name,
                "common_name": name,
                "known_health_effects": None,
                "human_studies_summary": f"{len(unique)} literature records retrieved; not a verification verdict.",
                "sources": [
                    {"title": p.title, "url": p.url, "source_type": "scientific_literature"}
                    for p in unique[:8]
                ],
            }
        )
        effectiveness.append(
            EffectivenessItem(
                scope="ingredient",
                subject=name,
                summary="Ingredient-level literature only. This does not prove the advertised product is effective.",
                evidence=unique[:8],
                caveat="Ingredient evidence is not product evidence.",
            )
        )
        try:
            trials = search_clinical_trials(name)
            for trial in trials:
                if trial.get("title"):
                    evidence.append(
                        ScientificEvidenceItem(
                            title=trial["title"],
                            url=trial.get("url"),
                            study_type=trial.get("study_type") or "clinical_trial",
                            outcomes=trial.get("outcomes"),
                            source_api="clinicaltrials",
                            related_ingredient=name,
                            evidence_scope="ingredient",
                        )
                    )
        except Exception:
            logger.exception("clinicaltrials failed for %s")

    medications = (brief.get("patient_context") or {}).get("medications") or ""
    diseases = (brief.get("patient_context") or {}).get("diseases") or ""
    interactions = []
    risks = []
    if medications:
        for med in [m.strip() for m in medications.replace("\n", ",").split(",") if m.strip()][:5]:
            labels = []
            try:
                labels = search_openfda_label(med)
            except Exception:
                logger.exception("openfda failed")
            for name in ingredients[:4] or ["supplement"]:
                interactions.append(
                    {
                        "ingredient": name,
                        "medicine": med,
                        "interaction_information": (labels[0].get("drug_interactions") if labels else None),
                        "possible_concerns": (labels[0].get("warnings") if labels else None),
                        "evidence": "OpenFDA label text if present; otherwise not found.",
                        "uncertainty": "Absence of a label hit is not proof of safety.",
                        "source_references": [
                            {
                                "title": "OpenFDA drug label",
                                "url": "https://open.fda.gov/apis/drug/label/",
                                "source_type": "openfda",
                            }
                        ]
                        if labels
                        else [],
                    }
                )
    if diseases:
        for disease in [d.strip() for d in diseases.replace("\n", ",").split(",") if d.strip()][:5]:
            for name in ingredients[:4] or [product_name or "supplement"]:
                papers = []
                try:
                    papers = _paper_items(
                        search_pubmed(f"{name} {disease} risk", retmax=4),
                        ingredient=name,
                        scope="ingredient",
                    )
                except Exception:
                    logger.exception("disease pubmed failed")
                risks.append(
                    {
                        "ingredient": name,
                        "disease": disease,
                        "available_evidence": "; ".join(p.title for p in papers[:3]) or None,
                        "potential_concern": None,
                        "limitations": "Automated retrieval only. Not a clinical assessment.",
                        "source_references": [
                            {"title": p.title, "url": p.url, "source_type": "scientific_literature"}
                            for p in papers[:4]
                        ],
                    }
                )
                evidence.extend(papers)

    if not ingredients:
        missing.append("No ingredients were mentioned in the advertisement, and none were confirmed from official sources.")

    for paper in evidence:
        if paper.url:
            sources.append(
                SourceRef(
                    title=paper.title,
                    url=paper.url,
                    source_type="scientific_literature",
                    publisher=paper.source_api,
                    retrieved_for=paper.related_ingredient or product_name,
                )
            )

    from app.schemas.dossier import IngredientResearchItem, InteractionItem, DiseaseRiskItem

    return ResearchDossier(
        product_identity=ProductIdentity(
            product_name=product_name or None,
            identity_status=identity_status,  # type: ignore[arg-type]
            notes="Scripted research pass without LLM tool-calling. Identity is not treated as verified.",
        ),
        composition=CompositionBlock(
            ingredients=[],
            composition_status="not_found",
            notes="No official composition was confirmed. Ingredients were not inferred.",
        ),
        product_information={"web_hits": web_hits[:10]},
        ingredient_research=[IngredientResearchItem.model_validate(x) for x in ingredient_research],
        scientific_evidence=evidence,
        effectiveness_research=effectiveness,
        supplement_drug_interactions=[InteractionItem.model_validate(x) for x in interactions],
        supplement_disease_risks=[DiseaseRiskItem.model_validate(x) for x in risks],
        missing_information=missing,
        uncertainties=uncertainties,
        sources=sources,
    )
