from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

from app.services.research.clinicaltrials import search_clinical_trials
from app.services.research.crossref import search_crossref
from app.services.research.europepmc import search_europe_pmc
from app.services.research.openfda import search_openfda_label
from app.services.research.pubmed import search_pubmed
from app.services.research.websearch import search_web

logger = logging.getLogger(__name__)


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)[:12000]


@tool
def web_search(query: str) -> str:
    """Search the public web for product pages, labels, manufacturer sites, and news. Do not invent URLs."""
    try:
        return _dumps(search_web(query))
    except Exception as exc:
        logger.exception("web_search failed")
        return json.dumps({"error": str(exc)})


@tool
def pubmed_search(query: str) -> str:
    """Search PubMed for scientific papers. Returns real titles, PMIDs, DOIs, and URLs only."""
    try:
        return _dumps(search_pubmed(query))
    except Exception as exc:
        logger.exception("pubmed_search failed")
        return json.dumps({"error": str(exc)})


@tool
def europepmc_search(query: str) -> str:
    """Search Europe PMC for papers and abstracts."""
    try:
        return _dumps(search_europe_pmc(query))
    except Exception as exc:
        logger.exception("europepmc_search failed")
        return json.dumps({"error": str(exc)})


@tool
def crossref_search(query: str) -> str:
    """Search Crossref for DOI-registered scholarly works."""
    try:
        return _dumps(search_crossref(query))
    except Exception as exc:
        logger.exception("crossref_search failed")
        return json.dumps({"error": str(exc)})


@tool
def clinicaltrials_search(query: str) -> str:
    """Search ClinicalTrials.gov for registered studies."""
    try:
        return _dumps(search_clinical_trials(query))
    except Exception as exc:
        logger.exception("clinicaltrials_search failed")
        return json.dumps({"error": str(exc)})


@tool
def openfda_search(query: str) -> str:
    """Search OpenFDA drug labels for warnings and interaction text when relevant."""
    try:
        return _dumps(search_openfda_label(query))
    except Exception as exc:
        logger.exception("openfda_search failed")
        return json.dumps({"error": str(exc)})


RESEARCH_TOOLS = [
    web_search,
    pubmed_search,
    europepmc_search,
    crossref_search,
    clinicaltrials_search,
    openfda_search,
]
