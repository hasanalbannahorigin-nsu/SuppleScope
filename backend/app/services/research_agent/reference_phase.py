"""Reference knowledge per ingredient (after the literature search of phase 3).

Sources, each optional (RESEARCH_REFERENCE_SOURCES and RESEARCH_OPENFDA):
- SuppleScope Bangla-Latin dictionary (always): English / scientific name, recorded as a KnowledgeLink.
- iDISK2.0 (local SQLite): interactions, adverse effects, cautions and indications as one-hop relations.
- NCCIH fact sheets (and NIH ODS when enabled; ODS refused scripts with 403 in Nov 2026): safety sentences,
  copied word for word.
- openFDA drug labels whose interaction / warning sections name the ingredient.

Everything becomes a Finding of class REFERENCE whose passage is the exact database text, plus a
safety item (drug interaction, disease caution or side effect) pointing at that finding. None of it is
evidence that the ingredient works; it feeds the patient-risk step and gives the verifier context.
"""
from __future__ import annotations

import logging
import re

from app.config import get_settings
from app.schemas.research.common import Finding, InformationClass, SourceType, Validation
from app.schemas.research.ingredient import DiseaseCaution, DrugInteraction, IngredientResearch, KnowledgeLink, SafetyItem
from app.services.knowledge import factsheets as F
from app.services.knowledge.idisk import IdiskStore
from app.services.knowledge.normalise import IngredientNormaliser, Normalised, default_normaliser
from app.services.research_agent.budget import BudgetExhausted
from app.services.research_agent.state import ResearchState
from app.services.research_agent.tools import openfda

logger = logging.getLogger(__name__)

CONDITION_WORDS = [
    ("pregnancy", r"pregnan"), ("breastfeeding", r"breast.?feed|lactat|nursing"), ("liver disease", r"liver|hepat"),
    ("kidney disease", r"kidney|renal"), ("surgery", r"surgery|surgical"), ("bleeding disorder", r"bleed"),
    ("diabetes", r"diabet|blood sugar|glucose|hypoglyc"), ("thyroid disease", r"thyroid"),
    ("low or high blood pressure", r"blood pressure|hypotens|hypertens"), ("allergy", r"allerg"),
    ("autoimmune disease", r"autoimmun"), ("children", r"\bchild|infant"),
]
DRUG_WORDS = re.compile(
    r"\b(warfarin|anticoagulant\w*|antiplatelet\w*|blood thinner\w*|aspirin|insulin|metformin|sulfonylurea\w*|"
    r"antidiabet\w*|diabetes (?:medicine|medication|drug)s?|levothyroxine|thyroid (?:medicine|medication|hormone)s?|"
    r"antihypertensive\w*|blood pressure (?:medicine|medication|drug)s?|immunosuppress\w*|sedative\w*|benzodiazepine\w*|"
    r"antidepressant\w*|chemotherap\w*|antibiotic\w*|tetracycline\w*|quinolone\w*|bisphosphonate\w*|digoxin|lithium|"
    r"diuretic\w*|statin\w*|cyclosporine|tacrolimus|oral contraceptive\w*|birth control pills?|proton pump inhibitor\w*)\b",
    re.I)


def _condition(sentence: str) -> str | None:
    for name, pat in CONDITION_WORDS:
        if re.search(pat, sentence, re.I):
            return name
    return None


def _drug(sentence: str) -> str | None:
    m = DRUG_WORDS.search(sentence)
    if m:
        return m.group(1).lower()
    if re.search(r"interact|medicat|medicine", sentence, re.I):
        return "medicines (see passage)"
    return None


def enabled() -> set[str]:
    return {x.strip().lower() for x in (get_settings().research_reference_sources or "").split(",") if x.strip()}


class ReferencePhase:
    def __init__(self, state: ResearchState, *, normaliser: IngredientNormaliser | None = None,
                 idisk: IdiskStore | None | bool = True, sources: set[str] | None = None, use_openfda: bool | None = None,
                 sheet_fetchers: dict | None = None, label_search=None) -> None:
        self.s = state
        self.norm = normaliser or default_normaliser()
        self.sources = enabled() if sources is None else sources
        if idisk is True:
            idisk = IdiskStore.open_default() if "idisk" in self.sources else None
        self.idisk: IdiskStore | None = idisk or None
        self.use_openfda = get_settings().research_openfda if use_openfda is None else use_openfda
        self.sheet_fetchers = sheet_fetchers or {"ods": F.ods_fact_sheet, "nccih": F.nccih_fact_sheet}
        self.label_search = label_search or openfda.drug_label_mentions

    def normalise(self, *names: str | None) -> Normalised | None:
        for n in names:
            r = self.norm.normalise(n)
            if r.entry_id:
                return r
        return None

    def enrich(self, ing: IngredientResearch, english_names: list[str]) -> IngredientResearch:
        links: list[KnowledgeLink] = list(ing.knowledge_links)
        fids: list[str] = []
        safety = ing.safety.model_copy(deep=True)
        nz = self.normalise(ing.name, ing.standardized_name, ing.scientific_name)
        if nz and not any(link.database == "SuppleScope dictionary" for link in links):
            links.append(KnowledgeLink(database="SuppleScope dictionary", entity_id=nz.entry_id, matched_name=nz.matched or nz.input,
                                       match_score=nz.score, method=nz.method))
        names = [n for n in dict.fromkeys([*(english_names or []), *(nz.search_names if nz else []),
                                           ing.scientific_name or "", ing.standardized_name or ""]) if n]
        english = (nz.english if nz else None) or next((n for n in names if re.fullmatch(r"[A-Za-z0-9 .,'\-]+", n)), None)
        try:
            if self.idisk is not None and names:
                self._idisk(ing, names, links, fids, safety)
            for db in ("ods", "nccih"):
                if db in self.sources and english:
                    self._sheet(db, ing, english, links, fids, safety)
            if self.use_openfda and names:
                self._labels(ing, names, fids, safety)
        except BudgetExhausted as exc:
            self.s.uncertainties.append(f"Reference lookups stopped early ({exc.kind} budget).")
        return ing.model_copy(update={"knowledge_links": links, "reference_finding_ids": ing.reference_finding_ids + fids,
                                      "safety": safety})

    # ---- helpers -----------------------------------------------------------------------------------------

    def _finding(self, ing: IngredientResearch, field: str, value: str, source_id: str, passage: str) -> str:
        f = Finding(finding_id=self.s.new_id("f"), field=field, value=value[:300], subject_id=ing.ingredient_id,
                    source_id=source_id, passage=passage[:1500], information_class=InformationClass.REFERENCE,
                    validation=Validation(quote_in_source=True, value_in_passage=True, refers_to_product=None,
                                          passage_supports_finding=True, checked_by="rule",
                                          notes="database text copied verbatim"))
        self.s.findings[f.finding_id] = f
        return f.finding_id

    def _idisk(self, ing, names, links, fids, safety) -> None:
        m = self.idisk.match(names)
        if not m:
            return
        facts = self.idisk.facts(m.entity_id)
        text = "\n".join(f.text for f in facts) or f"iDISK2.0 ingredient {m.name}"
        src = self.s.add_reference_source(source_type=SourceType.REFERENCE_DATABASE, url="https://github.com/houyurain/iDISK2.0",
                                          title=f"iDISK2.0 record {m.entity_id}: {m.name}", publisher="iDISK2.0",
                                          text=text, tool="idisk2_sqlite")
        links.append(KnowledgeLink(database="iDISK2.0", entity_id=m.entity_id, matched_name=m.matched_name,
                                   match_score=m.score, method="exact" if m.score >= 100 else "fuzzy", source_id=src.source_id))
        self.s.add_log("ingredient", "search", tool="idisk2_sqlite", query=" | ".join(names[:3]), result_count=len(facts))
        for fact in facts[:40]:
            if fact.kind not in ("interaction", "adverse_effect", "caution", "effective_for", "attribute"):
                continue
            fid = self._finding(ing, f"idisk_{fact.kind}", fact.value, src.source_id, fact.text)
            fids.append(fid)
            if fact.kind == "interaction":
                safety.drug_interactions.append(DrugInteraction(drug_or_class=fact.value.lower()[:120],
                                                                description=fact.text[:500], finding_ids=[fid]))
            elif fact.kind == "adverse_effect":
                safety.side_effects.append(SafetyItem(description=fact.text[:500], finding_ids=[fid]))
            elif fact.kind == "caution":
                safety.disease_cautions.append(DiseaseCaution(condition=fact.value.lower()[:120], description=fact.text[:500],
                                                              finding_ids=[fid]))
            elif fact.kind == "attribute" and "safety" in fact.relation.lower():
                # The DSI "Safety" text (from MSKCC / LNHPD) holds the cautions; split it like a fact sheet.
                for sent in F.safety_sentences(fact.value, limit=10):
                    drug, cond = _drug(sent), _condition(sent)
                    sid = self._finding(ing, "idisk_safety", (drug or cond or "safety")[:120], src.source_id, sent)
                    fids.append(sid)
                    if drug:
                        safety.drug_interactions.append(DrugInteraction(drug_or_class=drug, description=sent[:500],
                                                                        finding_ids=[sid]))
                    elif cond:
                        safety.disease_cautions.append(DiseaseCaution(condition=cond, description=sent[:500],
                                                                      finding_ids=[sid]))
                    else:
                        safety.side_effects.append(SafetyItem(description=sent[:500], finding_ids=[sid]))

    def _sheet(self, db, ing, english, links, fids, safety) -> None:
        fetch = self.sheet_fetchers[db]
        sheet = self.s.lookup(f"factsheet:{db}", english.lower(), lambda: fetch(english), "safety",
                              encode=F.sheet_to_json, decode=F.sheet_from_json, query=english)
        if not sheet:
            return
        src = self.s.add_reference_source(source_type=SourceType.REFERENCE_DATABASE, url=sheet.url, title=sheet.title,
                                          publisher=sheet.database, text=sheet.text, tool=f"{db}_fact_sheet")
        links.append(KnowledgeLink(database=sheet.database, matched_name=english, method="lookup", url=sheet.url,
                                   source_id=src.source_id))
        for sent in sheet.safety_sentences[:12]:
            drug, cond = _drug(sent), _condition(sent)
            fid = self._finding(ing, f"{db}_safety", (drug or cond or "safety")[:120], src.source_id, sent)
            fids.append(fid)
            if drug:
                safety.drug_interactions.append(DrugInteraction(drug_or_class=drug, description=sent[:500], finding_ids=[fid]))
            elif cond:
                safety.disease_cautions.append(DiseaseCaution(condition=cond, description=sent[:500], finding_ids=[fid]))
            else:
                safety.side_effects.append(SafetyItem(description=sent[:500], finding_ids=[fid]))

    def _labels(self, ing, names, fids, safety) -> None:
        latin = [n for n in names if re.fullmatch(r"[A-Za-z0-9 .,'\-]+", n)][:3]
        if not latin:
            return
        mentions = self.s.lookup("openfda:drug_label", "|".join(x.lower() for x in latin), lambda: self.label_search(latin),
                                 "safety", encode=openfda.mentions_to_json, decode=openfda.mentions_from_json,
                                 query=" OR ".join(latin))
        for m in (mentions or [])[:6]:
            src = self.s.add_reference_source(source_type=SourceType.DRUG_LABEL_DATABASE, url=m.url,
                                              title=f"FDA drug label: {m.drug}", publisher="openFDA drug label",
                                              text=m.sentence, tool="openfda_drug_label")
            fid = self._finding(ing, "drug_label_mention", m.drug, src.source_id, m.sentence)
            fids.append(fid)
            safety.drug_interactions.append(DrugInteraction(
                drug_or_class=m.drug[:120], description=f"The FDA label of {m.drug} names this ingredient in its "
                f"{m.section.replace('_', ' ')} section: {m.sentence[:380]}", finding_ids=[fid]))
