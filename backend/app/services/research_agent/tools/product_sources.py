"""Product sources tried in phase 1 besides the general web search (RESEARCH_PRODUCT_SOURCES).

- dsld:     NIH Dietary Supplement Label Database API (label copies of US-market products; the
            iDISK2.0 product layer is built from it). Hits are searched through the API and opened with
            `fetch_any`, which turns the label JSON into plain text the page reader can quote from.
- dgda:     targeted web queries on the Bangladesh Directorate General of Drug Administration site.
- pharmacy: targeted web queries on Bangladeshi online pharmacies (Arogga, MedEasy, Osudpotro), which
            often list the ingredients of local products. They stay THIRD_PARTY_RETAIL, never official.

The DSLD field names below follow the public v9 API. They could not be checked live from the build
machine, so every field is read defensively; run `python scripts/check_external_apis.py` once on a
machine with internet access to confirm them.
"""
from __future__ import annotations

import hashlib
import re

import httpx

from app.config import get_settings
from app.services.research.http import DEFAULT_TIMEOUT, USER_AGENT
from app.services.research_agent.tools.fetch import FetchedDoc, fetch_document
from app.services.research_agent.tools.search import SearchHit

DSLD_API = "https://api.ods.od.nih.gov/dsld/v9"
DSLD_LABEL_URL = "https://dsld.od.nih.gov/label/{id}"
DSLD_URL_RE = re.compile(r"^https?://dsld\.od\.nih\.gov/label/(\d+)")
BD_PHARMACIES = ("arogga.com", "medeasy.health", "osudpotro.com")


def enabled_sources() -> set[str]:
    return {x.strip().lower() for x in (get_settings().research_product_sources or "").split(",") if x.strip()}


def extra_queries(product: str | None, brand: str | None, sources: set[str] | None = None) -> list[str]:
    """Web queries aimed at Bangladeshi regulatory and pharmacy pages."""
    sources = enabled_sources() if sources is None else sources
    name = (product or brand or "").strip()
    if not name:
        return []
    out = []
    if "dgda" in sources:
        out.append(f'"{name}" site:dgda.gov.bd')
    if "pharmacy" in sources:
        out.append(f'"{name}" ' + " OR ".join(f"site:{d}" for d in BD_PHARMACIES))
    return out


def _get_json(path: str, params: dict) -> dict | list:
    with httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
        r = client.get(f"{DSLD_API}{path}", params=params)
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        return r.json()


class DsldSearch:
    """SearchProvider over the DSLD API, so DSLD hits go through the same triage and page reader."""

    name = "dsld"

    def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        payload = _get_json("/search-filter", {"q": query, "size": max_results})
        return parse_dsld_hits(payload)[:max_results]


def parse_dsld_hits(payload) -> list[SearchHit]:
    hits = payload.get("hits", []) if isinstance(payload, dict) else payload or []
    if isinstance(hits, dict):  # elasticsearch style {"hits": {"hits": [...]}}
        hits = hits.get("hits", [])
    out = []
    for h in hits:
        src = h.get("_source", h) if isinstance(h, dict) else {}
        lid = str(h.get("_id") or src.get("id") or "").strip()
        if not lid.isdigit():
            continue
        name = src.get("fullName") or src.get("full_name") or "unnamed product"
        brand = src.get("brandName") or src.get("brand_name")
        off = src.get("offMarket")
        snippet = "; ".join(x for x in (
            f"brand {brand}" if brand else "", f"UPC {src['upcSku']}" if src.get("upcSku") else "",
            "off market" if off in (1, True, "1") else "", "NIH Dietary Supplement Label Database (US label copy)") if x)
        out.append(SearchHit(title=f"DSLD label: {name}", url=DSLD_LABEL_URL.format(id=lid), snippet=snippet,
                             provider="dsld", rank=len(out) + 1))
    return out


def _qty(q) -> str:
    if not isinstance(q, dict):
        return ""
    amount = q.get("quantity")
    unit = q.get("unit") or ""
    op = q.get("operator") if q.get("operator") not in (None, "=") else ""
    return " ".join(str(x) for x in (op, amount, unit) if x not in (None, "")).strip()


def dsld_label_text(label: dict) -> tuple[str, str]:
    """(title, text) of a DSLD label JSON, in label order, quoting the label's own wording."""
    name = label.get("fullName") or "unnamed product"
    title = f"DSLD label: {name}"
    lines = [f"NIH Dietary Supplement Label Database (DSLD) copy of the product label. Label id {label.get('id')}.",
             f"Product name: {name}."]
    if label.get("brandName"):
        lines.append(f"Brand: {label['brandName']}.")
    for c in label.get("contacts") or []:
        d = c.get("contactDetails") if isinstance(c, dict) else None
        if isinstance(d, dict) and d.get("name"):
            place = ", ".join(x for x in (d.get("city"), d.get("state"), d.get("country")) if x)
            role = ", ".join(c.get("types") or []) if isinstance(c.get("types"), list) else ""
            lines.append(f"{role or 'Contact'}: {d['name']}" + (f", {place}" if place else "") + ".")
    sizes = []
    for ss in label.get("servingSizes") or []:
        if isinstance(ss, dict):
            q = ss.get("minQuantity") or ss.get("quantity")
            sizes.append(" ".join(str(x) for x in (q, ss.get("unit")) if x not in (None, "")))
    if any(sizes):
        lines.append("Serving size: " + "; ".join(s for s in sizes if s) + ".")
    rows = label.get("ingredientRows") or []
    if rows:
        lines.append("Supplement Facts:")

    def add_rows(rs, depth=0):
        for r in rs:
            if not isinstance(r, dict) or not r.get("name"):
                continue
            q = _qty((r.get("quantity") or [None])[0]) if isinstance(r.get("quantity"), list) else ""
            notes = f" ({r['notes']})" if r.get("notes") else ""
            lines.append(f"{'  ' * depth}- {r['name']}{': ' + q if q else ''}{notes}")
            add_rows(r.get("nestedRows") or [], depth + 1)

    add_rows(rows)
    other = label.get("otheringredients") or label.get("otherIngredients") or {}
    if isinstance(other, dict):
        names = [i.get("name") for i in other.get("ingredients") or [] if isinstance(i, dict) and i.get("name")]
        if other.get("text") or names:
            lines.append("Other ingredients: " + (other.get("text") or ", ".join(names)))
    for st in label.get("statements") or []:
        if isinstance(st, dict) and st.get("notes"):
            lines.append(f"{st.get('type') or 'Statement'}: {' '.join(str(st['notes']).split())}")
    for cl in label.get("claims") or []:
        if isinstance(cl, dict) and cl.get("langualCodeDescription"):
            lines.append(f"Claim category: {cl['langualCodeDescription']}")
    return title, "\n".join(lines)


def dsld_fetch(label_id: str, *, max_chars: int = 60000) -> FetchedDoc:
    label = _get_json(f"/label/{label_id}", {})
    if isinstance(label, list):
        label = label[0] if label else {}
    if not label:
        raise ValueError(f"DSLD label {label_id} not found")
    title, text = dsld_label_text(label)
    text = text[:max_chars]
    url = DSLD_LABEL_URL.format(id=label_id)
    return FetchedDoc(url=url, final_url=url, title=title, text=text, content_type="application/json",
                      sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), status=200)


def fetch_any(url: str, *, max_chars: int = 60000, timeout: float = 25.0) -> FetchedDoc:
    """Default fetcher of the v2 agent: DSLD label pages through the API, everything else as a web page."""
    m = DSLD_URL_RE.match(url)
    if m:
        return dsld_fetch(m.group(1), max_chars=max_chars)
    return fetch_document(url, max_chars=max_chars, timeout=timeout)
