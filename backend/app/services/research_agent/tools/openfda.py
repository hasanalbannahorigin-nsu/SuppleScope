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


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;])\s+(?=[A-Z(])", " ".join(text.split())) if s.strip()]


def drug_label_mentions(names: list[str], limit: int = 5) -> list[LabelMention]:
    """Drug labels whose interaction / warning sections name the ingredient (any of `names`)."""
    names = [n for n in dict.fromkeys(n.strip() for n in names if n and len(n.strip()) > 2)][:3]
    if not names:
        return []
    terms = " OR ".join(_quote(n) for n in names)
    search = " OR ".join(f"{sec}:({terms})" for sec in LABEL_SECTIONS[:3])
    payload = _get("/drug/label.json", {"search": search, "limit": limit})
    return parse_label_mentions(payload, names)


def parse_label_mentions(payload: dict, names: list[str]) -> list[LabelMention]:
    pats = [re.compile(rf"\b{re.escape(n)}\b", re.I) for n in names]
    out: list[LabelMention] = []
    for row in payload.get("results") or []:
        of = row.get("openfda") or {}
        drug = ", ".join((of.get("generic_name") or of.get("brand_name") or ["unnamed drug"])[:2])
        set_id = row.get("set_id") or (of.get("spl_set_id") or [None])[0]
        url = f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={set_id}" if set_id else f"{BASE}/drug/label.json"
        for sec in LABEL_SECTIONS:
            for block in row.get(sec) or []:
                for sent in _sentences(str(block)):
                    if any(p.search(sent) for p in pats):
                        out.append(LabelMention(drug=drug.lower(), section=sec, sentence=sent[:600], set_id=set_id, url=url))
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
