"""Run the research agent v2 on one AdExtraction and return a validated ResearchDossierV2.

Milestone 1 implements phases 1-2 (identity, brand/manufacturer, official sources,
composition). Later phases are reported as NOT_SEARCHED, never filled with guesses.
"""
from __future__ import annotations

import logging
import uuid

from app.config import get_settings
from app.schemas.research.ad_extraction import AdExtraction
from app.schemas.research.common import MissingItem, ResearchStatus
from app.schemas.research.dossier import AgentInfo, ResearchDossierV2, SafetySummary
from app.services.llm.client import JsonLlm, llm_from_settings
from app.services.research_agent.budget import Budget
from app.services.research_agent.product_phase import ProductPhase
from app.services.research_agent.state import ResearchState, now
from app.services.research_agent.tools.cache import DiskCache, NullCache
from app.services.research_agent.tools.search import SearchProvider, provider_from_settings

logger = logging.getLogger(__name__)

AGENT_VERSION = "0.1.0-milestone1"
NOT_YET = [
    ("ingredient-by-ingredient scientific research", "ingredient"),
    ("studies on the finished product", "finished_product"),
    ("claim-by-claim evidence", "claim"),
    ("safety, side effects and drug interactions", "safety"),
]


def run_research(
    ad: AdExtraction,
    *,
    llm: JsonLlm | None = None,
    search: SearchProvider | None = None,
    fetcher=None,
    budget: Budget | None = None,
    use_cache: bool = True,
    verify_with_llm: bool = True,
) -> ResearchDossierV2:
    s = get_settings()
    llm = llm or llm_from_settings()
    if llm is None:
        raise RuntimeError("Research agent v2 needs an LLM: set LLM_API_KEY (free Gemini works; see .env.example).")
    search = search or provider_from_settings()
    budget = budget or Budget.from_settings()
    state = ResearchState(budget=budget, llm=llm, search_provider=search, cache=DiskCache() if use_cache else NullCache(),
                          fetch_max_chars=s.research_fetch_max_chars)
    if fetcher is not None:
        state.fetcher = fetcher

    product = ProductPhase(state, ad, verify_with_llm=verify_with_llm).run()

    for item, phase in NOT_YET:
        state.missing.append(MissingItem(item=item, phase=phase, status=ResearchStatus.NOT_SEARCHED,
                                         reason="Not part of research milestone 1."))
    return ResearchDossierV2(
        dossier_id=f"dos_{uuid.uuid4().hex[:12]}",
        ad_id=ad.ad_id,
        created_at=now(),
        agent=AgentInfo(agent_version=AGENT_VERSION, llm_provider=_provider_name(s.llm_base_url),
                        llm_model=getattr(llm, "model", "unknown"), search_provider=search.name, literature_providers=[]),
        budget=budget.snapshot(),
        advertisement=ad,
        product=product,
        safety=SafetySummary(label_warning_finding_ids=product.composition.label_warning_finding_ids),
        sources=list(state.sources.values()),
        findings=list(state.findings.values()),
        conflicts=state.conflicts,
        missing_information=state.missing,
        uncertainties=list(dict.fromkeys(state.uncertainties)),
        research_log=state.log,
    )


def _provider_name(base_url: str) -> str:
    for key in ("generativelanguage", "groq", "localhost:11434", "openai.com", "anthropic"):
        if key in (base_url or ""):
            return {"generativelanguage": "gemini", "localhost:11434": "ollama", "openai.com": "openai"}.get(key, key)
    return "openai-compatible"
