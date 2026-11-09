"""Run the research agent v2 on one AdExtraction and return a validated ResearchDossierV2.

Phases 1-2 (identity, brand/manufacturer, official sources, composition) live in product_phase;
phases 3-6 (ingredient literature, finished-product studies, claim evidence, gaps) in evidence_phase.
Anything the budget does not reach is reported as NOT_SEARCHED, never filled with guesses.
"""
from __future__ import annotations

import logging
import uuid

from app.config import get_settings
from app.schemas.research.ad_extraction import AdExtraction
from app.schemas.research.dossier import AgentInfo, ResearchDossierV2, SafetySummary
from app.services.llm.client import JsonLlm, llm_from_settings
from app.services.research_agent.budget import Budget
from app.services.research_agent.evidence_phase import EvidencePhases
from app.services.research_agent.product_phase import ProductPhase
from app.services.research_agent.state import ResearchState, now
from app.services.research_agent.tools.cache import DiskCache, NullCache
from app.services.research_agent.tools.literature import literature_providers_from_settings
from app.services.research_agent.tools.search import SearchProvider, provider_from_settings

logger = logging.getLogger(__name__)

AGENT_VERSION = "0.2.0-milestone2"


def run_research(
    ad: AdExtraction,
    *,
    llm: JsonLlm | None = None,
    search: SearchProvider | None = None,
    fetcher=None,
    literature: dict | None = None,
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

    if literature is not None:
        state.literature_search = literature
    providers = [p for p in literature_providers_from_settings() if p in state.literature_search] or list(state.literature_search)

    product = ProductPhase(state, ad, verify_with_llm=verify_with_llm).run()
    phases = EvidencePhases(state, ad, product, max_ingredients=s.research_max_ingredients,
                            papers_per_query=s.research_papers_per_query, providers=providers)
    ingredients, finished, claims = phases.run()
    product = product.model_copy(update={"finished_product": finished})
    safety = SafetySummary(
        ingredient_ids_with_interactions=[i.ingredient_id for i in ingredients if i.safety.drug_interactions],
        ingredient_ids_with_disease_cautions=[i.ingredient_id for i in ingredients if i.safety.disease_cautions],
        label_warning_finding_ids=product.composition.label_warning_finding_ids,
    )
    return ResearchDossierV2(
        dossier_id=f"dos_{uuid.uuid4().hex[:12]}",
        ad_id=ad.ad_id,
        created_at=now(),
        agent=AgentInfo(agent_version=AGENT_VERSION, llm_provider=_provider_name(s.llm_base_url),
                        llm_model=getattr(llm, "model", "unknown"), search_provider=search.name, literature_providers=providers),
        budget=budget.snapshot(),
        advertisement=ad,
        product=product,
        ingredients=ingredients,
        claims=claims,
        safety=safety,
        sources=list(state.sources.values()),
        findings=list(state.findings.values()),
        evidence=phases.records,
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
