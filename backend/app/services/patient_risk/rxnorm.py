"""RxNorm (NLM RxNav REST) for medicine names the Bangladesh brand table does not know.

getApproximateMatch: /REST/approximateTerm.json?term=...&maxEntries=...  -> candidate rxcui
getRelatedByType:    /REST/rxcui/{rxcui}/related.json?tty=IN               -> ingredient (generic) name
Only names are used; RxNav's drug-interaction API is no longer offered by NLM.
"""
from __future__ import annotations

from functools import lru_cache

import httpx

from app.services.research.http import USER_AGENT

BASE = "https://rxnav.nlm.nih.gov/REST"


def _get(path: str, params: dict | None = None) -> dict:
    with httpx.Client(timeout=15.0, headers={"User-Agent": USER_AGENT}) as client:
        r = client.get(f"{BASE}{path}", params=params)
        r.raise_for_status()
        return r.json()


def parse_candidates(payload: dict) -> list[dict]:
    return ((payload or {}).get("approximateGroup") or {}).get("candidate") or []


def parse_ingredients(payload: dict) -> list[str]:
    out = []
    for group in ((payload or {}).get("relatedGroup") or {}).get("conceptGroup") or []:
        if group.get("tty") == "IN":
            out += [c.get("name") for c in group.get("conceptProperties") or [] if c.get("name")]
    return out


@lru_cache(maxsize=512)
def generic_name(term: str) -> str | None:
    """Best RxNorm ingredient name for a typed medicine name, or None."""
    cands = parse_candidates(_get("/approximateTerm.json", {"term": term, "maxEntries": 3}))
    for c in cands:
        rxcui = c.get("rxcui")
        if not rxcui:
            continue
        ings = parse_ingredients(_get(f"/rxcui/{rxcui}/related.json", {"tty": "IN"}))
        if ings:
            return ings[0]
    return None
