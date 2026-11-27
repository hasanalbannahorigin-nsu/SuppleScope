"""NIH reference fact sheets used by Behnamian et al. (2026) as authoritative background:

- NIH Office of Dietary Supplements (ODS) Fact Sheet API:
  https://ods.od.nih.gov/api/index.aspx?resourcename={Resource}&readinglevel=Health%20Professional&outputformat=XML
- NCCIH "Herbs at a Glance" pages: https://www.nccih.nih.gov/health/{slug}

Only sentences about safety (interactions, medicines, pregnancy, liver/kidney, side effects, upper limits)
are kept, word for word, so the patient-risk step and the verifier can quote them. A sheet that does not
exist for an ingredient is NOT_FOUND, not an error.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field

import httpx

from app.services.research.http import DEFAULT_TIMEOUT, USER_AGENT
from app.services.research_agent.tools.fetch import html_to_text, page_text

ODS_API = "https://ods.od.nih.gov/api/index.aspx"
ODS_PAGE = "https://ods.od.nih.gov/factsheets/{resource}-HealthProfessional/"
NCCIH_PAGE = "https://www.nccih.nih.gov/health/{slug}"

# English name (lower case) -> ODS resource name. Names not listed are tried as CamelCase.
ODS_RESOURCES = {
    "ashwagandha": "Ashwagandha", "biotin": "Biotin", "calcium": "Calcium", "chromium": "Chromium", "copper": "Copper",
    "folic acid": "Folate", "folate": "Folate", "iodine": "Iodine", "iron": "Iron", "magnesium": "Magnesium",
    "manganese": "Manganese", "niacin": "Niacin", "omega-3 fatty acids": "Omega3FattyAcids", "fish oil": "Omega3FattyAcids",
    "potassium": "Potassium", "probiotics": "Probiotics", "riboflavin": "Riboflavin", "selenium": "Selenium",
    "thiamin": "Thiamin", "vitamin a": "VitaminA", "vitamin b6": "VitaminB6", "vitamin b12": "VitaminB12",
    "vitamin c": "VitaminC", "vitamin d": "VitaminD", "vitamin e": "VitaminE", "vitamin k": "VitaminK", "zinc": "Zinc",
    "choline": "Choline", "pantothenic acid": "PantothenicAcid", "multivitamin": "MVMS", "boron": "Boron",
    "molybdenum": "Molybdenum", "phosphorus": "Phosphorus", "fluoride": "Fluoride",
}
# English name -> NCCIH slug for herbs common in Bangladeshi products.
NCCIH_SLUGS = {
    "ashwagandha": "ashwagandha", "turmeric": "turmeric", "curcumin": "turmeric", "fenugreek": "fenugreek",
    "ginger": "ginger", "garlic": "garlic", "cinnamon": "cinnamon", "ginseng": "asian-ginseng", "licorice": "licorice-root",
    "green tea": "green-tea", "aloe vera": "aloe-vera", "flaxseed": "flaxseed-and-flaxseed-oil", "black seed": "nigella-sativa",
    "bitter melon": "bitter-melon", "gymnema": "gymnema", "holy basil": "holy-basil", "fennel": "fennel",
    "garcinia cambogia": "garcinia-cambogia", "psyllium husk": "psyllium", "moringa": "moringa", "neem": "neem",
    "milk thistle": "milk-thistle", "ginkgo": "ginkgo", "echinacea": "echinacea", "st. john's wort": "st-johns-wort",
    "probiotics": "probiotics", "omega-3 fatty acids": "omega3-supplements-in-depth", "triphala": "triphala",
    "shilajit": "shilajit", "bacopa": "bacopa", "brahmi": "bacopa", "spirulina": "spirulina",
}
SAFETY_WORDS = re.compile(
    r"interact|medicat|medicine|drug|warfarin|anticoagul|blood thinner|insulin|diabet|blood sugar|pregnan|breast.?feed|"
    r"lactat|liver|hepat|kidney|renal|surgery|bleed|side effect|adverse|toxic|upper limit|\bUL\b|overdose|allerg|"
    r"thyroid|blood pressure|hypotens|immunosuppress|chemotherap|antidepress|sedat|not safe|unsafe|avoid|caution",
    re.I)


@dataclass
class FactSheet:
    database: str  # "NIH ODS" | "NCCIH"
    title: str
    url: str
    text: str
    safety_sentences: list[str] = field(default_factory=list)


def _sentences(text: str) -> list[str]:
    flat = " ".join(text.split())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", flat) if 25 <= len(s.strip()) <= 700]


def safety_sentences(text: str, limit: int = 15) -> list[str]:
    out = []
    for s in _sentences(text):
        if SAFETY_WORDS.search(s) and s not in out:
            out.append(s)
        if len(out) >= limit:
            break
    return out


def _camel(name: str) -> str:
    return "".join(w[:1].upper() + w[1:] for w in re.split(r"[^A-Za-z0-9]+", name) if w)


def ods_resource(name: str) -> str | None:
    n = (name or "").strip().lower()
    if not n:
        return None
    return ODS_RESOURCES.get(n) or (_camel(n) if re.fullmatch(r"[a-z0-9 \-]+", n) else None)


def _xml_text(xml: str) -> tuple[str, str]:
    """Title and plain text of an ODS fact-sheet XML; HTML inside elements is flattened."""
    root = ET.fromstring(xml)
    title = ""
    parts = []
    for el in root.iter():
        tag = el.tag.split("}")[-1].lower()
        txt = (el.text or "").strip()
        if not txt:
            continue
        if tag in ("title", "name") and not title:
            title = txt
        parts.append(html_to_text(txt)[1] if "<" in txt else txt)
    return title, "\n".join(parts)


def _get(url: str, params: dict | None = None) -> httpx.Response | None:
    with httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
        r = client.get(url, params=params)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r


def ods_fact_sheet(english_name: str) -> FactSheet | None:
    resource = ods_resource(english_name)
    if not resource:
        return None
    r = _get(ODS_API, {"resourcename": resource, "readinglevel": "Health Professional", "outputformat": "XML"})
    if r is None or not r.text.strip():
        return None
    try:
        title, text = _xml_text(r.text)
    except ET.ParseError:
        title, text = page_text(r.text)
    # Unknown resources come back as an error page or an empty document.
    if len(text) < 500 or re.search(r"not (be )?found|invalid resource", text[:400], re.I):
        return None
    return FactSheet(database="NIH ODS", title=title or f"{resource} Fact Sheet for Health Professionals",
                     url=ODS_PAGE.format(resource=resource), text=text, safety_sentences=safety_sentences(text))


def nccih_slug(english_name: str) -> str | None:
    n = (english_name or "").strip().lower()
    if not n:
        return None
    return NCCIH_SLUGS.get(n) or re.sub(r"[^a-z0-9]+", "-", n).strip("-") or None


def nccih_fact_sheet(english_name: str) -> FactSheet | None:
    slug = nccih_slug(english_name)
    if not slug:
        return None
    url = NCCIH_PAGE.format(slug=slug)
    r = _get(url)
    if r is None:
        return None
    title, text = page_text(r.text)
    if len(text) < 500:
        return None
    return FactSheet(database="NCCIH", title=title or f"{english_name} (NCCIH)", url=str(r.url), text=text,
                     safety_sentences=safety_sentences(text))


def sheet_to_json(f: FactSheet | None) -> dict | None:
    return asdict(f) if f else None


def sheet_from_json(d: dict | None) -> FactSheet | None:
    return FactSheet(**d) if d else None
