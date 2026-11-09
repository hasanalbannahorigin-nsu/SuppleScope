"""Research phases 3-5: ingredient literature, finished-product studies, claim-by-claim evidence.

Rules from the approved plan, enforced here:
- Ingredient evidence (scope INGREDIENT) and finished-product evidence (scope FINISHED_PRODUCT) are kept
  apart; ingredient studies are never presented as studies of the product.
- Every evidence record quotes its abstract, and the quote is checked against the abstract text that was
  actually retrieved. Records that fail the check are kept for audit but never linked to a claim.
- The agent describes how each study relates to each claim (SUPPORTS / CONTRADICTS / CONTEXT plus match
  levels). It never decides whether a claim is true; that is the separate verification model's job.
- Search, LLM and time limits come from the shared Budget. Anything not reached is NOT_SEARCHED.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from app.schemas.research.ad_extraction import AdClaim, AdExtraction
from app.schemas.research.claim import (
    ClaimEvidenceLink,
    ClaimEvidenceMatch,
    ClaimResearch,
    EvidenceRelation,
    MatchLevel,
)
from app.schemas.research.common import (
    EvidenceRecord,
    EvidenceScope,
    InformationClass,
    MissingItem,
    ResearchStatus,
    StudyType,
    Validation,
)
from app.schemas.research.ingredient import (
    DiseaseCaution,
    DrugInteraction,
    EffectEvidence,
    IngredientResearch,
    IngredientSafety,
    SafetyItem,
)
from app.schemas.research.product import FinishedProductResearch, IdentityStatus, ProductResearch
from app.services.research_agent import prompts as P
from app.services.research_agent.budget import BudgetExhausted
from app.services.research_agent.product_phase import core_product_name, script_of
from app.services.research_agent.state import ResearchState
from app.services.research_agent.tools.literature import Paper
from app.services.research_agent.validate import quote_in_source
from app.services.text.bangla import match_key, similarity

logger = logging.getLogger(__name__)

PAPERS_PER_READ = 5
ABSTRACT_CHARS = 3000
MAX_RECORDS_PER_LINK_CALL = 12
OFFICIAL_CLASSES = {InformationClass.OFFICIAL_MANUFACTURER, InformationClass.OFFICIAL_LABEL, InformationClass.REGULATORY}
HUMAN_TYPES = {StudyType.SYSTEMATIC_REVIEW, StudyType.META_ANALYSIS, StudyType.RCT, StudyType.NON_RANDOMIZED_TRIAL,
               StudyType.OBSERVATIONAL, StudyType.CASE_REPORT}
# PubMed publication types that settle the study design better than a reader model can.
PUBTYPE_TO_STUDY = [
    ("meta-analysis", StudyType.META_ANALYSIS),
    ("systematic review", StudyType.SYSTEMATIC_REVIEW),
    ("randomized controlled trial", StudyType.RCT),
    ("case reports", StudyType.CASE_REPORT),
]
SAFETY_TERMS = '"adverse effects" OR toxicity OR safety OR "drug interactions" OR "herb-drug interaction" OR hepatotoxicity'
TRIAL_TERMS = '"clinical trial" OR randomized OR "systematic review" OR "meta-analysis"'


@dataclass
class _Ing:
    ingredient_id: str
    name: str
    standardized_name: str | None
    origin: str
    amount: str | None = None
    names: P.IngredientNames | None = None
    claim_ids: list[str] = field(default_factory=list)
    outcome_terms: list[str] = field(default_factory=list)


def _study_type(reading_type: str, pub_types: list[str]) -> StudyType:
    lowered = [p.lower() for p in pub_types]
    for key, st in PUBTYPE_TO_STUDY:
        if any(key == p for p in lowered):
            return st
    try:
        return StudyType(reading_type.upper())
    except ValueError:
        return StudyType.OTHER


def _latin(names: list[str]) -> list[str]:
    out = []
    for n in names:
        n = " ".join((n or "").replace('"', " ").split())
        if n and script_of(n) == "latin" and n.lower() not in (x.lower() for x in out):
            out.append(n)
    return out


def _fallback_kind(claim: AdClaim) -> str:
    t = claim.claim_type.value
    if t == "SAFETY":
        return "safety"
    if t in ("CURE", "TREATMENT", "PREVENTION", "IMPROVEMENT", "REPLACEMENT"):
        return "health_effect"
    return "other"


def build_query(provider: str, names: list[str], terms: list[str] | str | None) -> str:
    """PubMed: names searched in title/abstract; Europe PMC: plain phrases."""
    tag = "[tiab]" if provider == "pubmed" else ""
    left = " OR ".join(f'"{n}"{tag}' for n in names)
    if not terms:
        return f"({left})"
    right = terms if isinstance(terms, str) else " OR ".join(f'"{t}"' for t in terms)
    return f"({left}) AND ({right})"


class EvidencePhases:
    def __init__(self, state: ResearchState, ad: AdExtraction, product: ProductResearch, *, max_ingredients: int = 6,
                 papers_per_query: int = 5, providers: list[str] | None = None) -> None:
        self.s = state
        self.ad = ad
        self.product = product
        self.max_ingredients = max_ingredients
        self.papers_per_query = papers_per_query
        self.providers = providers or ["pubmed"]
        self.evidence: dict[str, EvidenceRecord] = {}
        self.ing_queries: dict[str, list[str]] = {}
        self.exhausted: str | None = None
        ident = product.identity
        self.identity_ok = ident.status in (IdentityStatus.VERIFIED, IdentityStatus.PROBABLE)
        self.chosen = next((c for c in ident.candidates if c.candidate_id == ident.chosen_candidate_id), None)

    # ---- entry point --------------------------------------------------------------------------

    def run(self) -> tuple[list[IngredientResearch], FinishedProductResearch, list[ClaimResearch]]:
        ings = self._ingredient_list()
        plan = self._plan(ings)
        by_id = {i.ingredient_id: i for i in ings}
        for n in plan.ingredients:
            if n.ingredient_id in by_id:
                by_id[n.ingredient_id].names = n
        claim_plans = {c.claim_id: c for c in plan.claims}
        for cp in claim_plans.values():
            if cp.kind != "health_effect":
                continue
            for iid in cp.ingredient_ids:
                if iid in by_id:
                    by_id[iid].claim_ids.append(cp.claim_id)
                    by_id[iid].outcome_terms += [t for t in cp.outcome_terms if t not in by_id[iid].outcome_terms]

        # Ingredients tied to a health claim first; the rest after, up to the configured limit.
        order = sorted(ings, key=lambda i: (not i.claim_ids, ings.index(i)))
        ingredient_results: dict[str, IngredientResearch] = {}
        for rank, ing in enumerate(order):
            if rank >= self.max_ingredients:
                ingredient_results[ing.ingredient_id] = self._not_searched(ing, f"over the limit of {self.max_ingredients} "
                                                                          "ingredients per ad (RESEARCH_MAX_INGREDIENTS)")
                continue
            if self.exhausted:
                ingredient_results[ing.ingredient_id] = self._not_searched(ing, f"research budget exhausted ({self.exhausted})")
                continue
            try:
                ingredient_results[ing.ingredient_id] = self._research_ingredient(ing)
            except BudgetExhausted as exc:
                self.exhausted = exc.kind
                ingredient_results[ing.ingredient_id] = self._not_searched(ing, f"research budget exhausted ({exc.kind})")

        finished = self._finished_product()
        claims = [self._research_claim(c, claim_plans.get(c.claim_id), by_id, ingredient_results, finished)
                  for c in self.ad.claims]
        results = [ingredient_results[i.ingredient_id] for i in ings]
        self._record_missing(results, finished, claims)
        return results, finished, claims

    @property
    def records(self) -> list[EvidenceRecord]:
        return list(self.evidence.values())

    # ---- which ingredients -------------------------------------------------------------------

    def _ingredient_list(self) -> list[_Ing]:
        """Composition from the confirmed product (if any) plus ingredients only the ad shows (AD_ONLY)."""
        out: list[_Ing] = []
        used_mentions = set()
        if self.identity_ok:
            official_values = {c.value for c in OFFICIAL_CLASSES}
            for ci in self.product.composition.ingredients:
                official = ci.information_class in official_values
                origin = ("AD_AND_OFFICIAL" if ci.ad_mention_id and official else
                          "OFFICIAL_ONLY" if official else "THIRD_PARTY_ONLY")
                if ci.ad_mention_id:
                    used_mentions.add(ci.ad_mention_id)
                amount = " ".join(x for x in (ci.amount, ci.unit, ci.per) if x) or None
                out.append(_Ing(ci.ingredient_id, ci.name, ci.standardized_name, origin, amount))
        taken = {i.ingredient_id for i in out}
        n = 0
        for m in self.ad.ingredients:
            if m.mention_id in used_mentions:
                continue
            n += 1
            while f"ing_ad_{n:02d}" in taken:
                n += 1
            out.append(_Ing(f"ing_ad_{n:02d}", m.name_original, m.name_en, "AD_ONLY", m.amount))
        return out

    # ---- planning ------------------------------------------------------------------------------

    def _plan(self, ings: list[_Ing]) -> P.EvidencePlan:
        if not ings and not self.ad.claims:
            return P.EvidencePlan()
        user = json.dumps({
            "product": self.chosen.name if self.chosen else (self.ad.product_name.value if self.ad.product_name else None),
            "ingredients": [{"ingredient_id": i.ingredient_id, "name": i.name, "standardized_name": i.standardized_name}
                            for i in ings],
            "claims": [{"claim_id": c.claim_id, "text": c.text_original, "text_en": c.text_en, "claim_type": c.claim_type.value,
                        "target_en": c.target_en, "qualifiers": c.qualifiers} for c in self.ad.claims],
        }, ensure_ascii=False, indent=1)
        try:
            plan = self.s.llm_parse(P.EVIDENCE_PLAN_SYSTEM, user, P.EvidencePlan, "ingredient", "plan_evidence")
        except BudgetExhausted as exc:
            self.exhausted = exc.kind
            plan = None
        if plan is not None:
            return plan
        # Fallback without the planner: search by the given names, every claim against every ingredient.
        return P.EvidencePlan(
            ingredients=[P.IngredientNames(ingredient_id=i.ingredient_id,
                                           search_names=_latin([i.standardized_name or "", i.name])) for i in ings],
            claims=[P.ClaimPlan(claim_id=c.claim_id, kind=_fallback_kind(c),
                                outcome_terms=[c.target_en] if c.target_en else [],
                                ingredient_ids=[i.ingredient_id for i in ings]) for c in self.ad.claims],
        )

    # ---- phase 3: one ingredient -----------------------------------------------------------------

    def _search(self, names: list[str], terms, phase: str) -> tuple[list[Paper], list[str]]:
        tried = []
        for provider in self.providers:
            q = build_query(provider, names, terms)
            tried.append(f"{provider}: {q}")
            papers = self.s.literature(provider, q, phase, retmax=self.papers_per_query)
            papers = [p for p in papers if p.abstract.strip()]
            if papers:
                return papers, tried
        return [], tried

    def _research_ingredient(self, ing: _Ing) -> IngredientResearch:
        nm = ing.names or P.IngredientNames(ingredient_id=ing.ingredient_id)
        names = _latin(nm.search_names + [nm.scientific_name or "", ing.standardized_name or ""] +
                       ([ing.name] if script_of(ing.name) == "latin" else []))[:3]
        base = dict(ingredient_id=ing.ingredient_id, name=ing.name, standardized_name=ing.standardized_name,
                    scientific_name=nm.scientific_name, synonyms=nm.synonyms[:6], origin=ing.origin)
        if not names:
            return IngredientResearch(**base, status=ResearchStatus.NOT_FOUND,
                                      safety=IngredientSafety(status=ResearchStatus.NOT_SEARCHED),
                                      notes="No English or Latin name to search the literature with.")
        tried: list[str] = []
        eff_papers, t1 = self._search(names, ing.outcome_terms[:5] or TRIAL_TERMS, "ingredient")
        tried += t1
        if not eff_papers and ing.outcome_terms:
            eff_papers, t2 = self._search(names, TRIAL_TERMS, "ingredient")  # broaden: any trial or review
            tried += t2
        safe_papers, t3 = self._search(names, SAFETY_TERMS, "safety")
        tried += t3
        self.ing_queries[ing.ingredient_id] = tried

        papers, seen = [], set()
        for p in eff_papers + safe_papers:
            key = p.pmid or p.doi or p.title.lower()
            if key not in seen:
                seen.add(key)
                papers.append(p)
        subject = nm.scientific_name or names[0]
        records, safety = self._read_papers(papers, EvidenceScope.INGREDIENT, ing.ingredient_id,
                                            f"{ing.name} ({subject})", "ingredient")
        passed = [r for r in records if r.validation.passed]
        effects: dict[str, list[str]] = {}
        for r in passed:
            for e in r.outcomes:
                effects.setdefault(e.strip().lower(), []).append(r.evidence_id)
        human = [r for r in passed if r.human_study]
        doses = list(dict.fromkeys(r.dose for r in human if r.dose))
        if human:
            status = ResearchStatus.FOUND
        elif passed:
            status = ResearchStatus.PARTIAL
        else:
            status = ResearchStatus.NO_RELEVANT_STUDY_FOUND
        safety_status = ResearchStatus.FOUND if any(safety.values()) else ResearchStatus.NO_RELEVANT_STUDY_FOUND
        return IngredientResearch(
            **base, status=status,
            effects=[EffectEvidence(effect=k, evidence_ids=v) for k, v in effects.items()],
            evidence_ids=[r.evidence_id for r in records],
            tested_dose_range="; ".join(doses)[:300] or None,
            safety=IngredientSafety(status=safety_status, **safety),
            queries_tried=tried,
            notes=None if human else ("Only non-human or review evidence was found." if passed else None),
        )

    def _read_papers(self, papers: list[Paper], scope: EvidenceScope, subject_id: str, subject_name: str,
                     phase: str) -> tuple[list[EvidenceRecord], dict]:
        records: list[EvidenceRecord] = []
        safety = {"side_effects": [], "contraindications": [], "drug_interactions": [], "disease_cautions": []}
        for start in range(0, len(papers), PAPERS_PER_READ):
            chunk = papers[start:start + PAPERS_PER_READ]
            user = json.dumps({
                "subject": subject_name, "scope": "ingredient" if scope == EvidenceScope.INGREDIENT else "finished product",
                "papers": [{"index": i, "title": p.title, "year": p.year, "publication_types": p.publication_types[:4],
                            "abstract": p.abstract[:ABSTRACT_CHARS]} for i, p in enumerate(chunk)],
            }, ensure_ascii=False, indent=1)
            reading = self.s.llm_parse(P.READ_PAPERS_SYSTEM, user, P.PapersReading, phase, "read_papers")
            if reading is None:
                continue
            for pr in reading.papers:
                if not (0 <= pr.index < len(chunk)) or not pr.relevant:
                    continue
                paper = chunk[pr.index]
                src = self.s.add_paper_source(paper)
                text = self.s.paper_texts[src.source_id]
                passage = (pr.passage or "").strip()
                q_ok = bool(passage) and quote_in_source(passage, text)
                st = _study_type(pr.study_type, paper.publication_types)
                human = pr.human_study if pr.human_study is not None else (True if st in HUMAN_TYPES else None)
                if st in (StudyType.ANIMAL, StudyType.IN_VITRO):
                    human = False
                rec = EvidenceRecord(
                    evidence_id=self.s.new_id("ev"), scope=scope, subject_id=subject_id, subject_name=subject_name,
                    source_id=src.source_id, study_type=st, human_study=human, population=pr.population,
                    sample_size=pr.sample_size, intervention=pr.intervention, dose=pr.dose, duration=pr.duration,
                    comparator=pr.comparator, outcomes=[e for e in pr.effects if e][:6], result_summary=pr.result_summary,
                    limitations=pr.limitations[:5], passage=passage or "(no passage given)",
                    abstract_available=bool(paper.abstract.strip()),
                    validation=Validation(quote_in_source=q_ok, value_in_passage=None, refers_to_product=None,
                                          passage_supports_finding=q_ok, checked_by="rule",
                                          notes=None if q_ok else "passage not found in the abstract"),
                )
                self.evidence[rec.evidence_id] = rec
                records.append(rec)
                if not q_ok:
                    continue
                for item in pr.safety:
                    if not quote_in_source(item.passage, text):
                        continue
                    if item.kind == "drug_interaction":
                        safety["drug_interactions"].append(DrugInteraction(
                            drug_or_class=item.drug_or_condition or "unspecified", description=item.description,
                            evidence_ids=[rec.evidence_id]))
                    elif item.kind == "disease_caution":
                        safety["disease_cautions"].append(DiseaseCaution(
                            condition=item.drug_or_condition or "unspecified", description=item.description,
                            evidence_ids=[rec.evidence_id]))
                    else:
                        key = "side_effects" if item.kind == "side_effect" else "contraindications"
                        safety[key].append(SafetyItem(description=item.description, evidence_ids=[rec.evidence_id]))
        return records, safety

    def _not_searched(self, ing: _Ing, reason: str) -> IngredientResearch:
        nm = ing.names
        return IngredientResearch(
            ingredient_id=ing.ingredient_id, name=ing.name, standardized_name=ing.standardized_name,
            scientific_name=nm.scientific_name if nm else None, synonyms=(nm.synonyms[:6] if nm else []),
            origin=ing.origin, status=ResearchStatus.NOT_SEARCHED,
            safety=IngredientSafety(status=ResearchStatus.NOT_SEARCHED), notes=reason)

    # ---- phase 4: the finished product -------------------------------------------------------------

    def _finished_product(self) -> FinishedProductResearch:
        if not self.identity_ok or not self.chosen:
            return FinishedProductResearch(status=ResearchStatus.IDENTITY_UNCERTAIN,
                                           notes="Product identity not confirmed, so product-level studies were not searched.")
        if self.exhausted:
            return FinishedProductResearch(status=ResearchStatus.NOT_SEARCHED, notes=f"research budget exhausted ({self.exhausted})")
        core = core_product_name(self.chosen.name)
        words = [w for w in core.split() if not re.fullmatch(r"[\d.]+", w)]
        name = " ".join(words) or self.chosen.name
        try:
            papers, tried = self._search([name], None, "finished_product")
            # Keep only papers that really name the product; a common word must not pull in unrelated studies.
            key = match_key(name)
            papers = [p for p in papers if key and key in match_key(f"{p.title} {p.abstract}")]
            records, _ = self._read_papers(papers, EvidenceScope.FINISHED_PRODUCT, self.chosen.candidate_id,
                                           self.chosen.name, "finished_product")
        except BudgetExhausted as exc:
            self.exhausted = exc.kind
            return FinishedProductResearch(status=ResearchStatus.NOT_SEARCHED, notes=f"research budget exhausted ({exc.kind})")
        passed = [r for r in records if r.validation.passed]
        return FinishedProductResearch(
            status=ResearchStatus.FOUND if passed else ResearchStatus.NO_RELEVANT_STUDY_FOUND,
            evidence_ids=[r.evidence_id for r in records], queries_tried=tried,
            notes=None if passed else "No study of this exact product was found in the literature searched.")

    # ---- phase 5: one claim ---------------------------------------------------------------------------

    def _research_claim(self, claim: AdClaim, plan: P.ClaimPlan | None, ings: dict[str, _Ing],
                        results: dict[str, IngredientResearch], finished: FinishedProductResearch) -> ClaimResearch:
        kind = plan.kind if plan else "health_effect"
        official = self._official_mentions(claim)
        base = dict(claim_id=claim.claim_id, claim_text_original=claim.text_original, claim_text_en=claim.text_en,
                    claim_type=claim.claim_type.value, target=claim.target_en or claim.target, qualifiers=claim.qualifiers,
                    research_questions=[plan.question] if plan and plan.question else [],
                    official_material_mentions_claim=official[0], official_material_finding_ids=official[1])
        fp_status = finished.status
        if kind in ("regulatory_or_quality", "other"):
            return ClaimResearch(**base, status=ResearchStatus.NOT_SEARCHED, finished_product_status=fp_status,
                                 notes="Not a health-effect claim; see the product's official and regulatory findings.")
        researched = [iid for iid, r in results.items() if r.status not in (ResearchStatus.NOT_SEARCHED,)]
        linked = [i for i in (plan.ingredient_ids if plan else []) if i in ings] if kind == "health_effect" else researched
        if not ings:
            status = (ResearchStatus.IDENTITY_UNCERTAIN if not self.identity_ok else ResearchStatus.NOT_FOUND)
            return ClaimResearch(**base, linked_ingredient_ids=[], status=status, finished_product_status=fp_status,
                                 notes="No ingredient is known (none in the ad and no confirmed product label), "
                                       "so there is nothing to search studies for.")
        pool_ids = linked or researched
        candidates = [self.evidence[e] for iid in pool_ids for e in results[iid].evidence_ids
                      if e in self.evidence and self.evidence[e].validation.passed]
        candidates += [self.evidence[e] for e in finished.evidence_ids
                       if e in self.evidence and self.evidence[e].validation.passed]
        if kind == "safety":
            safety_ids = {e for iid in pool_ids for grp in (results[iid].safety.side_effects, results[iid].safety.contraindications,
                                                            results[iid].safety.drug_interactions, results[iid].safety.disease_cautions)
                          for item in grp for e in item.evidence_ids}
            candidates = [r for r in candidates if r.evidence_id in safety_ids] or candidates
        queries = [q for iid in pool_ids for q in self.ing_queries.get(iid, [])]
        if not candidates:
            searched_any = any(results[i].status != ResearchStatus.NOT_SEARCHED for i in pool_ids)
            return ClaimResearch(**base, linked_ingredient_ids=linked, status=(ResearchStatus.NO_RELEVANT_STUDY_FOUND
                                 if searched_any else ResearchStatus.NOT_SEARCHED), finished_product_status=fp_status,
                                 queries_tried=queries)
        # Human studies and reviews first, then the rest.
        rank = {StudyType.META_ANALYSIS: 0, StudyType.SYSTEMATIC_REVIEW: 0, StudyType.RCT: 1}
        candidates.sort(key=lambda r: (rank.get(r.study_type, 2), not r.human_study))
        candidates = candidates[:MAX_RECORDS_PER_LINK_CALL]
        links, dropped = [], 0
        if self.exhausted:
            reading = None
        else:
            try:
                reading = self._link(claim, kind, candidates)
            except BudgetExhausted as exc:
                self.exhausted, reading = exc.kind, None
        if reading is None:
            return ClaimResearch(**base, linked_ingredient_ids=linked, status=ResearchStatus.NOT_SEARCHED,
                                 finished_product_status=fp_status, queries_tried=queries,
                                 notes=f"Studies were found but not related to the claim ({self.exhausted or 'reader failed'}).")
        for li in reading.links:
            if not (0 <= li.index < len(candidates)):
                continue
            rec = candidates[li.index]
            passage = li.relevant_passage.strip()
            if not passage or not (quote_in_source(passage, rec.passage)
                                   or quote_in_source(passage, self.s.paper_texts.get(rec.source_id, ""))):
                dropped += 1
                continue
            links.append(ClaimEvidenceLink(
                evidence_id=rec.evidence_id, scope=rec.scope,
                ingredient_id=rec.subject_id if rec.scope == EvidenceScope.INGREDIENT else None,
                relation=EvidenceRelation(li.relation),
                match=ClaimEvidenceMatch(outcome=MatchLevel(li.outcome), population=MatchLevel(li.population),
                                         dose=MatchLevel(li.dose), intervention=MatchLevel(li.intervention),
                                         qualifiers=MatchLevel(li.qualifiers), human_evidence=rec.human_study,
                                         notes=li.notes),
                relevant_passage=passage))
        direct = [x for x in links if x.relation != EvidenceRelation.CONTEXT
                  and x.match.outcome in (MatchLevel.SAME, MatchLevel.SIMILAR)]
        status = (ResearchStatus.FOUND if direct else ResearchStatus.PARTIAL if links
                  else ResearchStatus.NO_RELEVANT_STUDY_FOUND)
        if any(x.scope == EvidenceScope.FINISHED_PRODUCT for x in links):
            fp_status = ResearchStatus.FOUND
        elif fp_status == ResearchStatus.FOUND:
            fp_status = ResearchStatus.NO_RELEVANT_STUDY_FOUND  # product studies exist, none on this claim
        notes = f"{dropped} link(s) dropped: quoted passage not found in the study record." if dropped else None
        return ClaimResearch(**base, linked_ingredient_ids=linked, status=status, finished_product_status=fp_status,
                             evidence_links=links, queries_tried=queries, notes=notes)

    def _link(self, claim: AdClaim, kind: str, records: list[EvidenceRecord]) -> P.ClaimLinks | None:
        usage = [u.text for u in self.ad.usage][:3]
        user = json.dumps({
            "claim": {"text": claim.text_original, "text_en": claim.text_en, "kind": kind, "target_en": claim.target_en,
                      "qualifiers": claim.qualifiers},
            "product_dose_from_ad": usage or None,
            "records": [{"index": i, "subject": r.subject_name, "scope": r.scope.value, "study_type": r.study_type.value,
                         "human_study": r.human_study, "population": r.population, "dose": r.dose, "duration": r.duration,
                         "outcomes": r.outcomes, "result_summary": r.result_summary, "passage": r.passage}
                        for i, r in enumerate(records)],
        }, ensure_ascii=False, indent=1)
        return self.s.llm_parse(P.LINK_SYSTEM, user, P.ClaimLinks, "claim", "link_claim")

    def _official_mentions(self, claim: AdClaim) -> tuple[bool | None, list[str]]:
        """Rule-based: does an official page state a similar purpose/claim? None when there is no official page."""
        if not self.identity_ok:
            return None, []
        official = [f for f in self.s.findings.values() if f.information_class in OFFICIAL_CLASSES
                    and f.validation.passed and f.field in ("claim", "intended_purpose", "description")]
        if not official:
            return None, []
        text = claim.text_en or claim.text_original
        hits = [f.finding_id for f in official if similarity(text, f.value) >= 0.6]
        return (bool(hits), hits)

    # ---- phase 6: what is still missing --------------------------------------------------------------

    def _record_missing(self, ings: list[IngredientResearch], finished: FinishedProductResearch,
                        claims: list[ClaimResearch]) -> None:
        for r in ings:
            if r.status in (ResearchStatus.NO_RELEVANT_STUDY_FOUND, ResearchStatus.NOT_FOUND, ResearchStatus.NOT_SEARCHED):
                self.s.missing.append(MissingItem(item=f"studies on ingredient '{r.name}'", phase="ingredient", status=r.status,
                                                  queries_tried=r.queries_tried[:6], reason=r.notes))
        if finished.status != ResearchStatus.FOUND:
            self.s.missing.append(MissingItem(item="studies on the finished product", phase="finished_product",
                                              status=finished.status, queries_tried=finished.queries_tried[:4],
                                              reason=finished.notes))
        for c in claims:
            if c.status not in (ResearchStatus.FOUND, ResearchStatus.PARTIAL):
                self.s.missing.append(MissingItem(item=f"evidence for claim {c.claim_id}", phase="claim", status=c.status,
                                                  queries_tried=c.queries_tried[:4], reason=c.notes))
