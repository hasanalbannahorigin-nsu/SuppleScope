"""openFDA for the v2 agent: adverse-event reports for a finished product (CAERS, food/event) and
drug labels whose interaction or warning sections name an ingredient (drug/label).

Both are reference material, not evidence of effect:
- CAERS reports are voluntary, US-only, and "not proof that the product caused the event".
- A drug label that names an ingredient tells the verifier and the patient-risk step that the FDA label
  of that drug warns about it; it says nothing about the supplement's benefits.
openFDA answers HTTP 404 when nothing matches, which is treated as "no records", not as an error.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field

import httpx

from app.config import get_settings
from app.services.research.http import DEFAULT_TIMEOUT, USER_AGENT

BASE = "https://api.fda.gov"
LABEL_SECTIONS = ("drug_interactions", "warnings", "warnings_and_cautions", "contraindications", "precautions")
CAERS_CAVEAT = ("Voluntary US adverse-event reports (FDA CAERS). A report is not proof that the product caused the "
                "event, and the absence of reports is not proof of safety.")


def _get(path: str, params: dict, attempts: int = 3) -> dict:
    key = get_settings().openfda_api_key
    if key:
        params = {**params, "api_key": key}
    last: Exception | None = None
    for n in range(attempts):
        try:
            with httpx.Client(timeout=DEFAULT_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
                r = client.get(f"{BASE}{path}", params=params)
            if r.status_code == 404:
                return {"results": [], "meta": {"results": {"total": 0}}}
            r.raise_for_status()
            return r.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            last = exc
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500 and exc.response.status_code != 429:
                raise
            time.sleep(1.5 * (n + 1))
    raise last  # type: ignore[misc]


def _quote(term: str) -> str:
    return '"' + term.replace('"', " ").strip() + '"'


# ---- CAERS: adverse-event reports for the finished product ------------------------------------------------

@dataclass
class AdverseEventReports:
    query: str
    total_reports: int
    reports_read: int
    top_reactions: list[tuple[str, int]] = field(default_factory=list)
    outcomes: list[tuple[str, int]] = field(default_factory=list)
    product_names_seen: list[str] = field(default_factory=list)
    url: str = ""


def food_event_reports(product_name: str, limit: int = 100) -> AdverseEventReports:
    search = f"products.name_brand:{_quote(product_name)}"
    payload = _get("/food/event.json", {"search": search, "limit": limit})
    return parse_food_events(payload, search)


def parse_food_events(payload: dict, search: str) -> AdverseEventReports:
    rows = payload.get("results") or []
    total = int(((payload.get("meta") or {}).get("results") or {}).get("total") or len(rows))
    reactions, outcomes, names = Counter(), Counter(), Counter()
    for row in rows:
        for r in row.get("reactions") or []:
            reactions[str(r).strip().lower()] += 1
        for o in row.get("outcomes") or []:
            outcomes[str(o).strip().lower()] += 1
        for p in row.get("products") or []:
            if p.get("name_brand"):
                names[p["name_brand"].strip()] += 1
    return AdverseEventReports(
        query=search, total_reports=total, reports_read=len(rows),
        top_reactions=reactions.most_common(10), outcomes=outcomes.most_common(6),
        product_names_seen=[n for n, _ in names.most_common(5)],
        url=f"{BASE}/food/event.json?search={search}",
    )


def caers_text(rep: AdverseEventReports) -> str:
    """Plain text of the summary, so findings can quote it like any other source."""
    lines = [f"openFDA food/event (CAERS) search {rep.query}: {rep.total_reports} report(s) in total; "
             f"{rep.reports_read} read."]
    if rep.product_names_seen:
        lines.append("Product names in the reports: " + "; ".join(rep.product_names_seen) + ".")
    for name, n in rep.top_reactions:
        lines.append(f"Reaction reported: {name} ({n} report(s)).")
    for name, n in rep.outcomes:
        lines.append(f"Outcome reported: {name} ({n} report(s)).")
    lines.append(CAERS_CAVEAT)
    return "\n".join(lines)


# ---- drug labels naming an ingredient ----------------------------------------------------------------------

@dataclass
class LabelMention:
    drug: str
    section: str
    sentence: str
    set_id: str | None
    url: str


BOUNDARY = re.compile(r"[.;]\s+(?=[A-Z(])")


def name_pattern(name: str) -> re.Pattern:
    """Ingredient name as a loose regex: case, punctuation and spacing may differ, so "St. John's Wort" also
    matches "St John's wort", "St. John’s Wort" (curly apostrophe) and "St.John's-wort"."""
    words = [w for w in re.split(r"[\W_]+", name) if w]
    return re.compile(r"\b" + r"[\W_]*".join(re.escape(w) for w in words) + r"\b", re.I)


def sentence_around(text: str, start: int, end: int) -> str:
    """The sentence holding text[start:end]. Boundaries inside the match (the "St." of "St. John's Wort") are
    ignored, so an abbreviation in the ingredient name does not cut the sentence."""
    bounds = [m.end() for m in BOUNDARY.finditer(text)]
    s = max((b for b in bounds if b <= start), default=0)
    e = min((b for b in bounds if b >= end), default=len(text))
    return text[s:e].strip()


def label_drug_name(row: dict) -> str:
    """Drug name of a label record. openFDA leaves the `openfda` block empty for many labels, so fall back to
    the product line of the label (spl_product_data_elements: "Name generic ACTIVE INGREDIENT ...")."""
    of = row.get("openfda") or {}
    names = of.get("generic_name") or of.get("brand_name")
    if names:
        return ", ".join(names[:2]).lower()
    first = ((row.get("spl_product_data_elements") or [""])[0] or "").split()
    lead = []
    for w in first[:8]:
        if w.isupper() and len(w) > 1 and lead:
            break  # active ingredients are listed in capitals after the product name
        if w.lower() not in lead:
            lead.append(w.lower())
    return " ".join(lead[:4]) or "unnamed drug"


def quote_window(text: str, start: int, end: int, size: int = 600) -> str:
    """The sentence holding the match, or, when the "sentence" is a flattened table with no full stops, an exact
    stretch of about `size` characters centred on the match (cut at spaces, never in the middle of a word)."""
    sent = sentence_around(text, start, end)
    if len(sent) <= size:
        return sent
    half = max(0, (size - (end - start)) // 2)
    a, b = max(0, start - half), min(len(text), end + half)
    if a > 0 and text[a - 1] != " ":  # started inside a word: move to the next word
        nxt = text.find(" ", a)
        if 0 <= nxt < start:
            a = nxt + 1
    if b < len(text) and text[b] != " ":  # ended inside a word: move back to the previous space
        prv = text.rfind(" ", end, b)
        if prv != -1:
            b = prv
    return text[a:b].strip()


def label_query(names: list[str], sections: tuple[str, ...] = LABEL_SECTIONS[:3]) -> str:
    """Flat openFDA search, one `section:"name"` clause per pair. openFDA answered 400 to a curly apostrophe and
    to OR inside a field group, so only plain ASCII phrases are sent; spelling differences in the label text are
    handled afterwards by name_pattern()."""
    return " OR ".join(f"{sec}:{_quote(n)}" for sec in sections for n in names)


def drug_label_mentions(names: list[str], limit: int = 5) -> list[LabelMention]:
    """Drug labels whose interaction / warning sections name the ingredient (any of `names`)."""
    names = [n for n in dict.fromkeys(" ".join(n.replace("\\", " ").split()) for n in names if n)
             if len(n) > 2 and n.isascii()][:3]
    if not names:
        return []
    try:
        payload = _get("/drug/label.json", {"search": label_query(names), "limit": limit})
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 400:
            raise
        # The simplest query openFDA documents: one phrase in one field.
        payload = _get("/drug/label.json", {"search": label_query(names[:1], ("drug_interactions",)), "limit": limit})
    return parse_label_mentions(payload, names)


def parse_label_mentions(payload: dict, names: list[str]) -> list[LabelMention]:
    pats = [name_pattern(n) for n in names]
    out: list[LabelMention] = []
    for row in payload.get("results") or []:
        of = row.get("openfda") or {}
        drug = label_drug_name(row)
        set_id = row.get("set_id") or (of.get("spl_set_id") or [None])[0]
        url = f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={set_id}" if set_id else f"{BASE}/drug/label.json"
        for sec in LABEL_SECTIONS:
            for block in row.get(sec) or []:
                text = " ".join(str(block).split())
                hit = next((m for p in pats for m in [p.search(text)] if m), None)
                if hit:
                    out.append(LabelMention(drug=drug, section=sec, set_id=set_id, url=url,
                                            sentence=quote_window(text, hit.start(), hit.end())))
                    break
    # one mention per (drug, section)
    seen, unique = set(), []
    for m in out:
        if (m.drug, m.section) not in seen:
            seen.add((m.drug, m.section))
            unique.append(m)
    return unique


def mentions_to_json(ms: list[LabelMention]) -> list[dict]:
    return [asdict(m) for m in ms]


def mentions_from_json(rows: list[dict]) -> list[LabelMention]:
    return [LabelMention(**r) for r in rows]


def reports_to_json(r: AdverseEventReports) -> dict:
    return asdict(r)


def reports_from_json(d: dict) -> AdverseEventReports:
    d = dict(d)
    d["top_reactions"] = [tuple(x) for x in d.get("top_reactions", [])]
    d["outcomes"] = [tuple(x) for x in d.get("outcomes", [])]
    return AdverseEventReports(**d)
