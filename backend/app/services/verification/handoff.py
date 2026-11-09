"""Build per-claim VerifierInput records from a finished ResearchDossierV2.

Plain code, no LLM: the verifier sees exactly what the dossier holds for that claim.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from app.schemas.research.dossier import ResearchDossierV2
from app.schemas.research.verifier import ProductContext, VerifierInput


def _product_context(d: ResearchDossierV2) -> ProductContext:
    ident = d.product.identity
    chosen = next((c for c in ident.candidates if c.candidate_id == ident.chosen_candidate_id), None)
    return ProductContext(
        identity_status=ident.status,
        identity_confidence=ident.confidence,
        product_name=chosen.name if chosen else (d.advertisement.product_name.value if d.advertisement.product_name else None),
        brand=chosen.brand if chosen else None,
        manufacturer=chosen.manufacturer if chosen else None,
        composition_status=d.product.composition.status,
        finished_product_status=d.product.finished_product.status,
    )


def build_verifier_inputs(d: ResearchDossierV2) -> list[VerifierInput]:
    product = _product_context(d)
    ad_claims = {c.claim_id: c for c in d.advertisement.claims}
    evidence = {e.evidence_id: e for e in d.evidence}
    findings = {f.finding_id: f for f in d.findings}
    sources = {s.source_id: s for s in d.sources}
    composed = {i.ingredient_id: i for i in d.product.composition.ingredients}
    out: list[VerifierInput] = []
    for cr in d.claims:
        claim = ad_claims[cr.claim_id]
        ingredients = [composed[i] for i in cr.linked_ingredient_ids if i in composed]
        ev = [evidence[e] for e in dict.fromkeys(link.evidence_id for link in cr.evidence_links)]
        finding_ids = [fid for ing in ingredients for fid in ing.finding_ids] + cr.official_material_finding_ids
        fs = [findings[f] for f in dict.fromkeys(finding_ids) if f in findings]
        ing_mentions = {i.ad_mention_id for i in ingredients if i.ad_mention_id}
        discrepancies = [
            x for x in d.product.ad_vs_official
            if x.ad_ref in {f"claims[{cr.claim_id}]"} | {f"ingredients[{m}]" for m in ing_mentions}
            or set(x.finding_ids) & set(finding_ids)
        ]
        used_sources = dict.fromkeys([e.source_id for e in ev] + [f.source_id for f in fs])
        out.append(
            VerifierInput(
                dossier_id=d.dossier_id,
                ad_id=d.ad_id,
                claim=claim,
                product=product,
                related_ingredients=ingredients,
                evidence_links=cr.evidence_links,
                evidence=ev,
                findings=fs,
                official_material_mentions_claim=cr.official_material_mentions_claim,
                discrepancies=discrepancies,
                sources=[sources[s] for s in used_sources if s in sources],
                claim_research_status=cr.status,
                missing_information=d.missing_information,
                uncertainties=d.uncertainties,
            )
        )
    return out


def write_jsonl(records: Iterable[VerifierInput], path: Path) -> int:
    """One JSON object per line: the format for training data and batch inference."""
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n")
            n += 1
    return n
