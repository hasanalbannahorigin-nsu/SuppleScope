"""Phases 1-2: product identification, brand/manufacturer, official website and label,
full ingredient/composition discovery, ad-vs-official comparison.

Loop:  plan queries -> search -> triage -> fetch -> read page -> validate facts
       -> group pages into candidate products -> decide identity -> list gaps
       -> follow-up queries (until gaps closed, rounds used up, or budget exhausted)
"""
from __future__ import annotations

import json
import logging
import re

from rapidfuzz import fuzz
from dataclasses import dataclass

from app.schemas.research.ad_extraction import AdExtraction
from app.schemas.research.common import (
    Conflict,
    Finding,
    InformationClass,
    MissingItem,
    ResearchStatus,
    SourceType,
    Validation,
)
from app.schemas.research.product import (
    BrandManufacturer,
    ComposedIngredient,
    Composition,
    Discrepancy,
    DiscrepancyKind,
    FinishedProductResearch,
    IdentityStatus,
    MatchScores,
    ProductCandidate,
    ProductIdentity,
    ProductResearch,
    RegulatoryRecord,
)
from app.services.research_agent import prompts as P
from app.services.research_agent.budget import BudgetExhausted
from app.services.research_agent.state import NEVER_OFFICIAL, SOURCE_RANK, ReadPage, ResearchState, domain
from app.services.research_agent.validate import quote_in_source, value_in_passage
from app.services.text.bangla import similarity, to_ascii_digits

logger = logging.getLogger(__name__)

OFFICIAL = {SourceType.OFFICIAL_LABEL, SourceType.OFFICIAL_MANUFACTURER, SourceType.REGULATORY}
PAGE_CHARS_FOR_LLM = 14000
FETCHES_PER_ROUND = 5
NAME_MATCH = 0.85

CLASS_FOR_TYPE = {
    SourceType.OFFICIAL_LABEL: InformationClass.OFFICIAL_LABEL,
    SourceType.OFFICIAL_MANUFACTURER: InformationClass.OFFICIAL_MANUFACTURER,
    SourceType.REGULATORY: InformationClass.REGULATORY,
    SourceType.THIRD_PARTY_RETAIL: InformationClass.THIRD_PARTY,
    SourceType.THIRD_PARTY_OTHER: InformationClass.THIRD_PARTY,
}


def _ad_summary(ad: AdExtraction, name_variants: list[str]) -> dict:
    m = ad.manufacturer
    return {
        "product_name": ad.product_name.value if ad.product_name else None,
        "name_variants": name_variants,
        "brand": ad.brand.value if ad.brand else None,
        "manufacturer": m.name.value if m.name else None,
        "manufacturer_website": m.website.value if m.website else None,
        "ingredients": [i.name_en or i.name_original for i in ad.ingredients],
        "claims_en": [c.text_en or c.text_original for c in ad.claims][:8],
        "other_text": [o.value for o in ad.other_text][:8],
    }


_COUNT = re.compile(r"\b\d+\s*(?:x\s*\d+\s*)?(?:pcs|pc|pieces|capsules|caps|tablets|tabs|softgels|sachets|strips)\b")
_PACK_AMOUNT = re.compile(r"\b\d+(?:\.\d+)?\s*(?:kg|gm|g|grams?|ml|l|ltr|litre|liter)\b")  # 120g, 225 ml = pack size
_FORM_WORDS = re.compile(r"\b(?:capsules?|caps|tablets?|tabs|softgels?|syrup|powder|sachets?|pot|jar|bottle|box|pack)\b")
_UNIT = re.compile(r"(\d)\s*(?:mg|mcg|µg|iu)\b")  # 500 mg = strength, kept as the number
_LEGAL = re.compile(r"\b(?:ltd|limited|pvt|private|plc|inc|llc|co|company|corp|corporation|bd|bangladesh)\b")
_BENGALI = re.compile(r"[ঀ-৿]")
_LATIN = re.compile(r"[A-Za-z]")
# Words shops add around a product name that do not make it a different product.
_GENERIC = {"natural", "herbal", "organic", "original", "ayurvedic", "unani", "supplement", "food", "product",
            "research", "weight", "gain", "loss", "for", "and", "the", "of", "with", "bd", "bangladesh", "imported"}


def script_of(text: str | None) -> str | None:
    """'bn' or 'latin' (whichever letters dominate), None for empty/other."""
    if not text:
        return None
    bn, la = len(_BENGALI.findall(text)), len(_LATIN.findall(text))
    if not bn and not la:
        return None
    return "bn" if bn > la else "latin"


def core_product_name(name: str) -> str:
    """Product name without pack size, form words or a marketing tail:
    'Dibedex 500 mg (30 Pcs)' -> 'dibedex 500'; 'Healthy Plus Milk Supplement 120g' -> 'healthy plus milk supplement'.

    The strength number (500 mg) is kept, because a different strength is a different product.
    """
    t = (name or "").lower()
    # a dash/bar with a space next to it, or a dash where Latin text switches to Bangla, starts a tail
    parts = re.split(r"\s[-–—|:]\s?|[-–—|]\s|\s[-–—|]|(?<=[a-z0-9])[-–—](?=[\u0980-\u09FF])", t)
    if len(parts) > 1 and len(parts[0].split()) >= 2:
        t = parts[0]  # 'Healthy Plus Milk Supplement-১মাসে ৮-১০কেজি ...' -> before the tail
    t = to_ascii_digits(t)
    t = re.sub(r"\([^)]*\)?", " ", t)  # (30 Pcs), (Ayurvedic), unclosed '( Weight Gain'
    t = _COUNT.sub(" ", t)
    t = _PACK_AMOUNT.sub(" ", t)
    t = _UNIT.sub(r"\1", t)
    t = _FORM_WORDS.sub(" ", t)
    t = re.sub(r"[^\w\s.\u0980-\u09FF]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _core_company(name: str) -> str:
    t = re.sub(r"\([^)]*\)", " ", (name or "").lower())
    t = _LEGAL.sub(" ", re.sub(r"[^\w\s]+", " ", t))
    return re.sub(r"\s+", " ", t).strip()


def _comparable(a: str | None, b: str | None) -> bool:
    """Both present and in the same script. 'Hamdard' vs 'হামদর্দ' cannot be compared by spelling."""
    return bool(a and b) and script_of(a) == script_of(b)


def same_product(name_a: str, name_b: str, *, brand_a: str | None = None, brand_b: str | None = None,
                 maker_a: str | None = None, maker_b: str | None = None) -> bool:
    """Do two page descriptions name the same product?

    Same: pack-size variants (30 vs 60 pcs, 120 g), form words, a company-name prefix ('Index Dibedex 500'),
    generic shop words ('Weight gain ... milk supplement'). Different: another strength, a distinguishing
    extra word ('Plus', 'Silver'), or a clearly different brand or manufacturer.
    """
    company_words = set()
    for m in (maker_a, maker_b):
        if m and script_of(m) == "latin":
            company_words |= set(_core_company(m).split()) - {"laboratories", "laboratory", "pharma", "pharmaceuticals"}
    ca, cb = core_product_name(name_a) or name_a.lower(), core_product_name(name_b) or name_b.lower()
    na, nb = set(re.findall(r"\d+(?:\.\d+)?", ca)), set(re.findall(r"\d+(?:\.\d+)?", cb))
    if na and nb and na != nb:
        return False
    wa = [w for w in re.sub(r"[\d.]+", " ", ca).split() if w not in company_words]
    wb = [w for w in re.sub(r"[\d.]+", " ", cb).split() if w not in company_words]
    if not wa or not wb:
        return False
    if fuzz.token_sort_ratio(" ".join(wa), " ".join(wb)) < NAME_MATCH * 100:
        # Allow one name to contain the other when the extra words are only generic shop words.
        sa, sb = set(wa), set(wb)
        small, big = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
        if not (small <= big and (big - small) <= _GENERIC and small - _GENERIC):
            return False
    if _comparable(brand_a, brand_b) and similarity(brand_a, brand_b) < 0.8:
        # A brand can be the company ('Index') on one page and the product line ('Dibedex') on another.
        other = " ".join(x for x in (name_a, name_b, maker_a, maker_b) if x).lower()
        if brand_a.lower() not in other.replace(brand_a.lower(), "", 1) and brand_b.lower() not in other.replace(brand_b.lower(), "", 1):
            return False
    if _comparable(maker_a, maker_b):
        ma, mb = _core_company(maker_a), _core_company(maker_b)
        if ma and mb and similarity(ma, mb) < 0.8:
            return False
    return True


def name_coverage(ad_name: str, cand_name: str) -> float | None:
    """How much of the advertised name appears in a found product's name (0..1); None across scripts.

    Every distinctive word of the ad name must be present. Extra words on the shop page are allowed
    (shops add descriptors) but a distinctive extra word costs a little, since 'Saduri Plus' is not 'Saduri'.
    'Complete Hair Treatment Pack' vs 'MAYA Ready to Use Complete Care Hair Pack' -> 0.6 (treatment missing).
    """
    if script_of(ad_name) != script_of(cand_name):
        return None
    a = core_product_name(ad_name).split() or ad_name.lower().split()
    c = core_product_name(cand_name).split() or cand_name.lower().split()
    a_key = [w for w in a if w not in _GENERIC] or a
    if not a_key or not c:
        return 0.0

    def found(w: str, words: list[str]) -> bool:
        return any(fuzz.ratio(w, x) >= 85 for x in words)

    cov = sum(1 for w in a_key if found(w, c)) / len(a_key)
    extra = [x for x in c if x not in _GENERIC and not re.fullmatch(r"[\d.]+", x) and not found(x, a)]
    return round(cov * (0.9 if extra else 1.0), 2)


@dataclass
class _Cand:
    candidate_id: str
    name: str
    brand: str | None
    manufacturer: str | None
    form: str | None
    country: str | None
    source_ids: list[str]


class ProductPhase:
    def __init__(self, state: ResearchState, ad: AdExtraction, *, verify_with_llm: bool = True) -> None:
        self.s, self.ad, self.verify_with_llm = state, ad, verify_with_llm
        self.name_variants: list[str] = [ad.product_name.value] if ad.product_name else []
        self.candidates: list[_Cand] = []
        self.identity: ProductIdentity | None = None
        self.official_website: tuple[str, str] | None = None  # (url, source_id)
        self.ing_meta: dict[str, tuple[str | None, str | None, str | None]] = {}  # finding_id -> (amount, unit, per)

    # ---- loop ----------------------------------------------------------------------

    def run(self) -> ProductResearch:
        if not self.ad.product_name and not self.ad.brand:
            return self._no_name_result()
        try:
            queries = self._plan_queries()
            for round_no in range(self.s.budget.max_rounds):
                hits = []
                for q in queries:
                    hits += self.s.search(q, "identity")
                self._open_and_read(hits, round_no)
                self._decide_identity()
                gaps = self._gaps()
                if not gaps or round_no == self.s.budget.max_rounds - 1:
                    break
                queries = self._follow_up(gaps)
                if not queries:
                    break
        except BudgetExhausted as exc:
            self.s.uncertainties.append(f"Product research stopped early: {exc}.")
            self._decide_identity()
        return self._result()

    def _plan_queries(self) -> list[str]:
        summary = _ad_summary(self.ad, [])
        plan = self.s.llm_parse(P.QUERY_PLAN_SYSTEM, json.dumps(summary, ensure_ascii=False, indent=1),
                                P.QueryPlan, "identity", "plan_queries")
        base = [v for v in [summary["product_name"], summary["brand"]] if v]
        if plan:
            self.name_variants = list(dict.fromkeys(self.name_variants + plan.name_variants))
            return list(dict.fromkeys(plan.queries[:6] + base))
        return base + [f"{base[0]} ingredients"] if base else []

    def _follow_up(self, gaps: list[str]) -> list[str]:
        known = {
            "ad": _ad_summary(self.ad, self.name_variants),
            "identity": self.identity.model_dump(mode="json", include={"status", "reasoning"}) if self.identity else None,
            "candidates": [c.__dict__ for c in self.candidates][:5],
            "official_website": self.official_website[0] if self.official_website else None,
            "missing": gaps,
            "queries_already_tried": self.s.queries_tried,
        }
        out = self.s.llm_parse(P.FOLLOW_UP_SYSTEM, json.dumps(known, ensure_ascii=False, indent=1), P.FollowUp,
                               "identity", "follow_up")
        return [q for q in (out.queries if out else [])[:4]]

    def _open_and_read(self, hits, round_no: int) -> None:
        hits = [h for h in hits if h.url not in self.s.urls_seen]
        seen, unique = set(), []
        for h in hits:
            if h.url not in seen:
                seen.add(h.url)
                unique.append(h)
        if not unique:
            return
        listing = [{"i": i, "title": h.title, "url": h.url, "snippet": (h.snippet or "")[:300]} for i, h in enumerate(unique)]
        user = json.dumps({"advertised_product": _ad_summary(self.ad, self.name_variants), "results": listing},
                          ensure_ascii=False, indent=1)
        triage = self.s.llm_parse(P.TRIAGE_SYSTEM, user, P.Triage, "identity", "triage")
        picks = [p for p in (triage.open if triage else []) if 0 <= p.hit_index < len(unique)]
        if not picks:  # model unavailable or picked nothing: open the top results
            picks = [P.TriagePick(hit_index=i, expected_kind="THIRD_PARTY_OTHER", reason="top result") for i in range(min(3, len(unique)))]
        for p in picks[:FETCHES_PER_ROUND]:
            hit = unique[p.hit_index]
            doc = self.s.fetch(hit.url, "official_sources")
            if doc:
                self._read_page(doc)

    # ---- reading one page ----------------------------------------------------------

    def _read_page(self, doc) -> None:
        user = json.dumps({"advertised_product": _ad_summary(self.ad, self.name_variants), "url": doc.final_url,
                           "page_title": doc.title}, ensure_ascii=False, indent=1)
        user += f"\n\n<page_text>\n{doc.text[:PAGE_CHARS_FOR_LLM]}\n</page_text>"
        reading = self.s.llm_parse(P.READ_SYSTEM, user, P.PageReading, "composition", "read_page")
        if reading is None:
            return
        kind = SourceType(reading.page_kind)
        if kind in OFFICIAL and any(d in domain(doc.final_url) for d in NEVER_OFFICIAL):
            kind = SourceType.THIRD_PARTY_RETAIL if kind != SourceType.REGULATORY else SourceType.THIRD_PARTY_OTHER
        src = self.s.add_source(doc, kind, "fetch_page")
        page = ReadPage(source_id=src.source_id, doc=doc, page_kind=kind, about=reading.about_product.model_dump(),
                        refers=reading.refers_to_ad_product, refers_reason=reading.refers_reason)
        self.s.pages[src.source_id] = page
        items: list[tuple[str, str, str | None, str | None, str | None, str]] = [
            (f.field, f.value, None, None, None, f.passage) for f in reading.facts
        ] + [("ingredient", i.name, i.amount, i.unit, i.per, i.passage) for i in reading.ingredients]
        new_ids = []
        for field, value, amount, unit, per, passage in items:
            q_ok = quote_in_source(passage, doc.text)
            v_ok = value_in_passage(value, passage, extra_values=(amount,))
            f = Finding(
                finding_id=self.s.new_id("f"), field=field,
                value=value,
                unit=" ".join(x for x in (amount, unit) if x) or None,
                subject_id=None, source_id=src.source_id, passage=passage,
                information_class=CLASS_FOR_TYPE[kind] if (q_ok and v_ok) else InformationClass.UNVERIFIED,
                validation=Validation(
                    quote_in_source=q_ok, value_in_passage=v_ok,
                    refers_to_product={"yes": True, "no": False}.get(reading.refers_to_ad_product),
                    passage_supports_finding=q_ok and v_ok, checked_by="rule",
                ),
            )
            self.s.findings[f.finding_id] = f
            if field == "ingredient":
                self.ing_meta[f.finding_id] = (amount, unit, per)
            new_ids.append(f.finding_id)
            (page.ingredient_finding_ids if field == "ingredient" else page.finding_ids).append(f.finding_id)
        if self.verify_with_llm:
            self._llm_check(page, new_ids)
        if kind == SourceType.OFFICIAL_MANUFACTURER and reading.refers_to_ad_product != "no" and not self.official_website:
            self.official_website = (doc.final_url, src.source_id)
        for fid in page.finding_ids:
            f = self.s.findings[fid]
            if f.field == "official_website" and f.validation.passed and not self.official_website:
                self.official_website = (f.value, src.source_id)

    def _llm_check(self, page: ReadPage, finding_ids: list[str]) -> None:
        """Semantic check: does each passage really state the fact for this product?"""
        todo = [fid for fid in finding_ids if self.s.findings[fid].validation.quote_in_source]
        if not todo:
            return
        product = page.about.get("name") or (self.ad.product_name.value if self.ad.product_name else "")
        items = [{"index": n, "product": product, "field": self.s.findings[fid].field,
                  "value": " ".join(x for x in (self.s.findings[fid].value, self.s.findings[fid].unit) if x),
                  "passage": self.s.findings[fid].passage} for n, fid in enumerate(todo)]
        out = self.s.llm_parse(P.CHECK_SYSTEM, json.dumps(items, ensure_ascii=False, indent=1), P.CheckResult,
                               "validation", "check_findings")
        if out is None:
            return
        verdicts = {i.index: i for i in out.items}
        model = getattr(self.s.llm, "model", "llm")
        for n, fid in enumerate(todo):
            f = self.s.findings[fid]
            v = verdicts.get(n)
            supports = bool(v and v.supports) and f.validation.value_in_passage is not False
            val = f.validation.model_copy(update={
                "passage_supports_finding": supports, "checked_by": f"rule+{model}",
                "notes": (v.reason if v and not v.supports else f.validation.notes),
            })
            cls = f.information_class if val.passed else InformationClass.UNVERIFIED
            self.s.findings[fid] = f.model_copy(update={"validation": val, "information_class": cls})

    # ---- candidates and identity ---------------------------------------------------

    def _group_candidates(self) -> None:
        self.candidates = []
        for sid, page in self.s.pages.items():
            name = page.about.get("name")
            if not name:
                continue
            brand = page.about.get("brand")
            match = None
            for c in self.candidates:
                if same_product(c.name, name, brand_a=c.brand, brand_b=brand,
                                maker_a=c.manufacturer, maker_b=page.about.get("manufacturer")):
                    match = c
                    break
            if match is None:
                match = _Cand(self.s.new_id("prod_c"), name, brand, page.about.get("manufacturer"), page.about.get("form"),
                              page.about.get("country"), [])
                self.candidates.append(match)
            match.source_ids.append(sid)
            for attr in ("brand", "manufacturer", "form", "country"):
                if not getattr(match, attr) and page.about.get(attr):
                    setattr(match, attr, page.about.get(attr))
            page.candidate_id = match.candidate_id

    def _scores(self, c: _Cand) -> MatchScores:
        ad = self.ad
        covs = [x for x in (name_coverage(v, c.name) for v in self.name_variants) if x is not None]
        name = max(covs) if covs else max((similarity(v, c.name) for v in self.name_variants), default=0.0)
        brand = (similarity(ad.brand.value, c.brand or c.name)
                 if ad.brand and _comparable(ad.brand.value, c.brand or c.name) else None)
        man = (similarity(ad.manufacturer.name.value, c.manufacturer)
               if ad.manufacturer.name and _comparable(ad.manufacturer.name.value, c.manufacturer) else None)
        ing = None
        if ad.ingredients:
            names = [self.s.findings[f].value for sid in c.source_ids for f in self.s.pages[sid].ingredient_finding_ids]
            if names:
                hit = sum(1 for i in ad.ingredients if max((similarity(i.name_en or i.name_original, n) for n in names), default=0) >= 0.8)
                ing = hit / len(ad.ingredients)
        country = None
        if c.country:
            country = 1.0 if "bangladesh" in c.country.lower() else 0.3
        return MatchScores(name=round(name, 2), brand=None if brand is None else round(brand, 2),
                           manufacturer=None if man is None else round(man, 2),
                           ingredient_overlap=None if ing is None else round(ing, 2), country=country)

    @staticmethod
    def _overall(m: MatchScores, refers: list[str]) -> float:
        weights = {"name": 3, "brand": 2, "manufacturer": 2, "ingredient_overlap": 2, "country": 1}
        num = den = 0.0
        for k, w in weights.items():
            v = getattr(m, k)
            if v is not None:
                num += w * v
                den += w
        score = num / den if den else 0.0
        if refers:
            # pages that say "this is NOT the advertised product" pull the score down in proportion
            not_no = sum(1 for r in refers if r != "no") / len(refers)
            score *= 0.3 + 0.7 * not_no
        return round(score, 2)

    def _decide_identity(self) -> None:
        self._group_candidates()
        scored = []
        for c in self.candidates:
            m = self._scores(c)
            refers = [self.s.pages[sid].refers for sid in c.source_ids]
            scored.append((self._overall(m, refers), c, m, refers))
        scored.sort(key=lambda x: -x[0])
        cands = []
        for overall, c, m, refers in scored:
            cands.append(ProductCandidate(candidate_id=c.candidate_id, name=c.name, brand=c.brand, manufacturer=c.manufacturer,
                                          country=c.country, form=c.form, source_ids=c.source_ids, match_scores=m,
                                          overall_score=overall))
        if not cands:
            self.identity = ProductIdentity(status=IdentityStatus.NOT_FOUND, confidence=0.0, candidates=[],
                                            reasoning="No page describing a matching product was found within the search budget.",
                                            queries_tried=list(self.s.queries_tried))
            return
        best = scored[0]
        overall, c, m, refers = best
        has_official = any(self.s.pages[sid].page_kind in OFFICIAL and self.s.pages[sid].refers != "no" for sid in c.source_ids)
        close = [x for x in scored[1:] if overall - x[0] < 0.1 and x[0] >= 0.5]
        if overall < 0.6 or (m.name or 0) < 0.7:
            status, why = IdentityStatus.IDENTITY_UNCERTAIN, "the best match is weak"
        elif close:
            status, why = IdentityStatus.IDENTITY_UNCERTAIN, f"{len(close)} other candidate(s) match almost as well"
        elif has_official and (m.name or 0) >= NAME_MATCH:
            status, why = IdentityStatus.VERIFIED, "an official/regulatory source matches the advertised name"
        else:
            status, why = IdentityStatus.PROBABLE, "the name matches but no official source was found"
        for cand in cands[1:]:
            cand.rejected_reason = cand.rejected_reason or (
                "Lower match score than the chosen product" if status != IdentityStatus.IDENTITY_UNCERTAIN
                else "Not chosen; identity is uncertain")
        chosen = c.candidate_id if status in (IdentityStatus.VERIFIED, IdentityStatus.PROBABLE) else None
        kinds = sorted({self.s.pages[sid].page_kind.value for sid in c.source_ids})
        reasoning = (f"Best candidate '{c.name}' (score {overall:.2f}; name {m.name}, brand {m.brand}, "
                     f"manufacturer {m.manufacturer}, ingredients {m.ingredient_overlap}, country {m.country}) "
                     f"from {len(c.source_ids)} page(s) of type {', '.join(kinds)}. Status {status.value}: {why}.")
        self.identity = ProductIdentity(status=status, confidence=overall if chosen else min(overall, 0.5),
                                        chosen_candidate_id=chosen, candidates=cands, reasoning=reasoning,
                                        queries_tried=list(self.s.queries_tried))
        # refers_to_product on every finding now follows the identity decision
        for sid, page in self.s.pages.items():
            on_chosen = chosen is not None and page.candidate_id == chosen
            for fid in page.finding_ids + page.ingredient_finding_ids:
                f = self.s.findings[fid]
                refers_val = True if on_chosen and page.refers != "no" else (False if chosen else f.validation.refers_to_product)
                val = f.validation.model_copy(update={"refers_to_product": refers_val})
                cls = f.information_class if val.passed else InformationClass.UNVERIFIED
                if val.passed and f.information_class == InformationClass.UNVERIFIED:
                    cls = CLASS_FOR_TYPE[page.page_kind]
                self.s.findings[fid] = f.model_copy(update={"validation": val, "information_class": cls, "subject_id": page.candidate_id})

    def _chosen_pages(self) -> list[ReadPage]:
        cid = self.identity.chosen_candidate_id if self.identity else None
        return [p for p in self.s.pages.values() if cid and p.candidate_id == cid]

    def _gaps(self) -> list[str]:
        gaps = []
        if not self.identity or self.identity.status in (IdentityStatus.NOT_FOUND, IdentityStatus.IDENTITY_UNCERTAIN):
            gaps.append("product identity")
        if not self.official_website:
            gaps.append("manufacturer official website")
        if not any(p.page_kind in OFFICIAL for p in self._chosen_pages()):
            gaps.append("official product label or official product page")
        if not any(self.s.findings[f].validation.passed for p in self._chosen_pages() for f in p.ingredient_finding_ids):
            gaps.append("ingredient list")
        return gaps

    # ---- result ---------------------------------------------------------------------

    def _valid(self, fids: list[str], field: str | None = None) -> list[str]:
        return [f for f in fids if self.s.findings[f].validation.passed and (field is None or self.s.findings[f].field == field)]

    def _result(self) -> ProductResearch:
        pages = sorted(self._chosen_pages(), key=lambda p: SOURCE_RANK[p.page_kind])
        all_fids = [f for p in pages for f in p.finding_ids]
        ing_by_page = {p.source_id: self._valid(p.ingredient_finding_ids) for p in pages}

        # composition: best-ranked page group that lists ingredients
        comp_status, composed, label_sources = ResearchStatus.NOT_FOUND, [], []
        ranked = [p for p in pages if ing_by_page[p.source_id]]
        if ranked:
            top_rank = SOURCE_RANK[ranked[0].page_kind]
            top = [p for p in ranked if SOURCE_RANK[p.page_kind] == top_rank]
            primary = top[0]
            for n, fid in enumerate(ing_by_page[primary.source_id], start=1):
                f = self.s.findings[fid]
                amount, unit, per = self.ing_meta.get(fid, (None, None, None))
                composed.append(ComposedIngredient(
                    ingredient_id=f"ing_{n:02d}", name=f.value, amount=amount, unit=unit, per=per,
                    information_class=f.information_class.value, finding_ids=[fid],
                    ad_mention_id=self._ad_mention(f.value),
                ))
            # same ingredient on other pages -> extra provenance; different lists -> conflict
            names0 = {self.s.findings[f].value.lower() for f in ing_by_page[primary.source_id]}
            for p in ranked[1:]:
                names = {self.s.findings[f].value.lower() for f in ing_by_page[p.source_id]}
                for fid in ing_by_page[p.source_id]:
                    for ci in composed:
                        if similarity(ci.name, self.s.findings[fid].value) >= 0.9:
                            ci.finding_ids.append(fid)
                overlap = len(names0 & names) / max(len(names0 | names), 1)
                if overlap < 0.6 and SOURCE_RANK[p.page_kind] <= top_rank + 4:
                    self.s.conflicts.append(Conflict(
                        topic="ingredient list",
                        finding_ids=ing_by_page[primary.source_id] + ing_by_page[p.source_id],
                        description=(f"{self.s.sources[primary.source_id].site_or_publisher} and "
                                     f"{self.s.sources[p.source_id].site_or_publisher} list different ingredients."),
                    ))
            label_sources = [p.source_id for p in top if p.page_kind in OFFICIAL]
            comp_status = ResearchStatus.FOUND if primary.page_kind in OFFICIAL else ResearchStatus.PARTIAL
            if any(c.topic == "ingredient list" for c in self.s.conflicts):
                self.s.uncertainties.append("Sources disagree on the ingredient list; see conflicts.")
        if not composed and self.identity and self.identity.status == IdentityStatus.IDENTITY_UNCERTAIN:
            comp_status = ResearchStatus.IDENTITY_UNCERTAIN

        composition = Composition(
            status=comp_status, ingredients=composed,
            serving_size_finding_ids=self._valid(all_fids, "serving_size"),
            dosage_directions_finding_ids=self._valid(all_fids, "dosage"),
            intended_purpose_finding_ids=self._valid(all_fids, "intended_purpose"),
            manufacturer_description_finding_ids=self._valid(all_fids, "description"),
            label_warning_finding_ids=self._valid(all_fids, "warning"),
            label_source_ids=label_sources,
        )
        brand_ids, man_ids = self._valid(all_fids, "brand"), self._valid(all_fids, "manufacturer_name")
        site = self.official_website if (self.official_website and self._is_chosen_source(self.official_website[1])) else None
        bm_status = ResearchStatus.FOUND if (man_ids and site) else (ResearchStatus.PARTIAL if (brand_ids or man_ids or site) else ResearchStatus.NOT_FOUND)
        country = next((self.s.findings[f].value for f in self._valid(all_fids, "country")), None)
        brand_manufacturer = BrandManufacturer(status=bm_status, brand_finding_ids=brand_ids, manufacturer_finding_ids=man_ids,
                                               official_website=site[0] if site else None,
                                               official_website_source_id=site[1] if site else None, country=country)
        regulatory = []
        reg_nums = self._valid(all_fids, "registration_number")
        if reg_nums:
            auth = self._valid(all_fids, "registration_authority")
            regulatory.append(RegulatoryRecord(
                authority=self.s.findings[auth[0]].value if auth else "unknown",
                registration_number=self.s.findings[reg_nums[0]].value, finding_ids=reg_nums + auth))

        self._record_missing(composition, brand_manufacturer, regulatory)
        return ProductResearch(
            identity=self.identity, brand_manufacturer=brand_manufacturer, composition=composition, regulatory=regulatory,
            ad_vs_official=self._discrepancies(composition, pages),
            finished_product=FinishedProductResearch(status=ResearchStatus.NOT_SEARCHED,
                                                     notes="Finished-product literature search is phase 4 (next milestone)."),
        )

    def _is_chosen_source(self, source_id: str) -> bool:
        return any(p.source_id == source_id for p in self._chosen_pages())

    def _ad_mention(self, name: str) -> str | None:
        for i in self.ad.ingredients:
            if max(similarity(i.name_en, name), similarity(i.name_original, name)) >= 0.8:
                return i.mention_id
        return None

    def _discrepancies(self, comp: Composition, pages: list[ReadPage]) -> list[Discrepancy]:
        out: list[Discrepancy] = []
        if comp.status not in (ResearchStatus.FOUND, ResearchStatus.PARTIAL):
            return out
        matched = {c.ad_mention_id for c in comp.ingredients if c.ad_mention_id}
        src = "official" if comp.status == ResearchStatus.FOUND else "third-party"
        for i in self.ad.ingredients:
            if i.mention_id not in matched:
                out.append(Discrepancy(kind=DiscrepancyKind.INGREDIENT_IN_AD_NOT_IN_OFFICIAL, ad_ref=f"ingredients[{i.mention_id}]",
                                       description=f"The ad names '{i.name_original}', which is not in the {src} ingredient list found."))
        for c in comp.ingredients:
            if not c.ad_mention_id and self.ad.ingredients:
                out.append(Discrepancy(kind=DiscrepancyKind.INGREDIENT_IN_OFFICIAL_NOT_IN_AD, finding_ids=c.finding_ids,
                                       description=f"'{c.name}' is in the {src} ingredient list but not named in the ad."))
        if comp.status == ResearchStatus.FOUND and not self.ad.ingredients and comp.ingredients:
            out.append(Discrepancy(kind=DiscrepancyKind.INGREDIENT_IN_OFFICIAL_NOT_IN_AD,
                                   finding_ids=[f for c in comp.ingredients for f in c.finding_ids],
                                   description="The ad names no ingredients; the official source lists them."))
        if comp.label_warning_finding_ids and not self.ad.warnings:
            out.append(Discrepancy(kind=DiscrepancyKind.WARNING_IN_OFFICIAL_NOT_IN_AD, finding_ids=comp.label_warning_finding_ids,
                                   description="The official/third-party material has warnings the ad does not show."))
        return out

    def _record_missing(self, comp: Composition, bm: BrandManufacturer, regulatory: list) -> None:
        q = list(self.s.queries_tried)
        status_if_uncertain = (ResearchStatus.IDENTITY_UNCERTAIN
                               if self.identity and self.identity.status == IdentityStatus.IDENTITY_UNCERTAIN else None)
        exhausted = ResearchStatus.NOT_SEARCHED if self.s.budget.exhausted_reason else None
        if not bm.official_website:
            self.s.missing.append(MissingItem(item="manufacturer official website", phase="brand_manufacturer",
                                              status=status_if_uncertain or exhausted or ResearchStatus.NOT_FOUND, queries_tried=q))
        if not comp.label_source_ids:
            self.s.missing.append(MissingItem(item="official product label", phase="official_sources",
                                              status=status_if_uncertain or exhausted or ResearchStatus.NOT_FOUND, queries_tried=q))
        if not comp.ingredients:
            self.s.missing.append(MissingItem(item="ingredient list", phase="composition",
                                              status=comp.status if comp.status != ResearchStatus.FOUND else ResearchStatus.NOT_FOUND,
                                              queries_tried=q))
        if not regulatory:
            self.s.missing.append(MissingItem(item="regulatory registration", phase="official_sources",
                                              status=status_if_uncertain or exhausted or ResearchStatus.NOT_FOUND, queries_tried=q))
        if self.identity and self.identity.status != IdentityStatus.VERIFIED:
            self.s.uncertainties.append(f"Product identity is {self.identity.status.value}: {self.identity.reasoning}")

    def _no_name_result(self) -> ProductResearch:
        self.identity = ProductIdentity(status=IdentityStatus.NOT_FOUND, confidence=0.0,
                                        reasoning="The advertisement shows no readable product or brand name, so no product search was possible.")
        self.s.missing.append(MissingItem(item="product name in the advertisement", phase="identity",
                                          status=ResearchStatus.NOT_FOUND, reason="Not visible/readable in the ad."))
        return ProductResearch(
            identity=self.identity,
            brand_manufacturer=BrandManufacturer(status=ResearchStatus.NOT_SEARCHED),
            composition=Composition(status=ResearchStatus.NOT_SEARCHED),
            finished_product=FinishedProductResearch(status=ResearchStatus.NOT_SEARCHED),
        )
