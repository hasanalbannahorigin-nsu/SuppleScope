"""Mutable state of one research run, plus the tool wrappers that charge the budget and log."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse

from app.schemas.research.common import Conflict, Finding, MissingItem, Source, SourceType
from app.schemas.research.dossier import LogEntry
from app.services.llm.client import JsonLlm
from app.services.research_agent.budget import Budget, BudgetExhausted
from app.services.research_agent.tools.cache import NullCache
from app.services.research_agent.tools.fetch import FetchedDoc, doc_from_json, doc_to_json, fetch_document
from app.services.research_agent.tools.literature import LITERATURE_SEARCH, Paper, paper_from_json, paper_to_json
from app.services.research_agent.tools.search import SearchHit, SearchProvider, hits_from_json, hits_to_json

logger = logging.getLogger(__name__)

SOURCE_RANK = {
    SourceType.OFFICIAL_LABEL: 1,
    SourceType.OFFICIAL_MANUFACTURER: 2,
    SourceType.REGULATORY: 2,
    SourceType.SCIENTIFIC_LITERATURE: 3,
    SourceType.CLINICAL_TRIAL_REGISTRY: 3,
    SourceType.DRUG_LABEL_DATABASE: 3,
    SourceType.REFERENCE_DATABASE: 4,
    SourceType.THIRD_PARTY_RETAIL: 6,
    SourceType.THIRD_PARTY_OTHER: 7,
    SourceType.ADVERTISEMENT: 9,
}

# Pages on these domains are never "official", whatever a reader model says.
NEVER_OFFICIAL = (
    "daraz.", "arogga.", "chaldal.", "medeasy.", "osudpotro.", "lazz", "amazon.", "ebay.", "alibaba.", "aliexpress.",
    "facebook.", "fb.", "instagram.", "youtube.", "tiktok.", "pinterest.", "wikipedia.", "shajgoj.", "othoba.",
    "pickaboo.", "bikroy.", "flipkart.", "1mg.", "netmeds.", "pharmeasy.", "iherb.", "walmart.",
)


def now() -> datetime:
    return datetime.now(timezone.utc)


def domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


@dataclass
class ReadPage:
    source_id: str
    doc: FetchedDoc
    page_kind: SourceType
    about: dict
    refers: str  # yes / no / unclear
    refers_reason: str
    finding_ids: list[str] = field(default_factory=list)
    ingredient_finding_ids: list[str] = field(default_factory=list)
    candidate_id: str | None = None


@dataclass
class ResearchState:
    budget: Budget
    llm: JsonLlm
    search_provider: SearchProvider
    fetcher: callable = fetch_document
    cache: object = field(default_factory=NullCache)
    fetch_max_chars: int = 60000
    sources: dict[str, Source] = field(default_factory=dict)
    pages: dict[str, ReadPage] = field(default_factory=dict)  # source_id -> page
    findings: dict[str, Finding] = field(default_factory=dict)
    conflicts: list[Conflict] = field(default_factory=list)
    missing: list[MissingItem] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    log: list[LogEntry] = field(default_factory=list)
    queries_tried: list[str] = field(default_factory=list)
    urls_seen: set[str] = field(default_factory=set)
    literature_search: dict = field(default_factory=lambda: dict(LITERATURE_SEARCH))
    literature_queries: list[str] = field(default_factory=list)
    paper_sources: dict[str, str] = field(default_factory=dict)  # pmid/doi/title key -> source_id
    paper_texts: dict[str, str] = field(default_factory=dict)  # source_id -> title + abstract (for quote checks)
    _ids: dict[str, int] = field(default_factory=dict)

    def new_id(self, prefix: str) -> str:
        self._ids[prefix] = self._ids.get(prefix, 0) + 1
        return f"{prefix}_{self._ids[prefix]:03d}"

    def add_log(self, phase: str, action: str, **kw) -> None:
        self.log.append(LogEntry(step=len(self.log) + 1, phase=phase, action=action, at=now(), **kw))

    def _retry(self, fn, *args):
        last = None
        for _ in range(self.budget.max_retries_per_call + 1):
            try:
                return fn(*args)
            except BudgetExhausted:
                raise
            except Exception as exc:  # network errors, 4xx/5xx, parse errors
                last = exc
        raise last

    # ---- tools -------------------------------------------------------------------------

    def search(self, query: str, phase: str, max_results: int = 8) -> list[SearchHit]:
        query = " ".join(query.split())
        if not query or query.lower() in (q.lower() for q in self.queries_tried):
            return []
        self.queries_tried.append(query)
        key = f"{self.search_provider.name}|{max_results}|{query}"
        cached = self.cache.get("search", key)
        if cached is not None:
            hits = hits_from_json(cached)
            self.add_log(phase, "search", tool=f"web_search:{self.search_provider.name}(cache)", query=query, result_count=len(hits))
            return hits
        self.budget.charge_search()
        try:
            hits = self._retry(self.search_provider.search, query, max_results)
        except BudgetExhausted:
            raise
        except Exception as exc:
            self.add_log(phase, "search", tool=f"web_search:{self.search_provider.name}", query=query, error=str(exc)[:300])
            return []
        self.cache.set("search", key, hits_to_json(hits))
        self.add_log(phase, "search", tool=f"web_search:{self.search_provider.name}", query=query, result_count=len(hits))
        return hits

    def fetch(self, url: str, phase: str) -> FetchedDoc | None:
        if url in self.urls_seen:
            return None
        self.urls_seen.add(url)
        cached = self.cache.get("fetch", url)
        if cached is not None:
            doc = doc_from_json(cached)
            self.add_log(phase, "fetch", tool="fetch_page(cache)", target_url=url)
            return doc
        self.budget.charge_fetch()
        try:
            doc = self._retry(lambda u: self.fetcher(u, max_chars=self.fetch_max_chars), url)
        except BudgetExhausted:
            raise
        except Exception as exc:
            self.add_log(phase, "fetch", tool="fetch_page", target_url=url, error=str(exc)[:300])
            return None
        if not doc.text.strip():
            self.add_log(phase, "fetch", tool="fetch_page", target_url=url, error="no text extracted")
            return None
        self.cache.set("fetch", url, doc_to_json(doc))
        self.add_log(phase, "fetch", tool="fetch_page", target_url=url)
        return doc

    def literature(self, provider: str, query: str, phase: str, retmax: int = 5) -> list[Paper]:
        """PubMed / Europe PMC search returning papers with abstracts (cached, budgeted, logged)."""
        query = " ".join(query.split())
        key = f"{provider}|{retmax}|{query}"
        if not query or key in self.literature_queries:
            return []
        self.literature_queries.append(key)
        tool = f"literature:{provider}"
        cached = self.cache.get("literature", key)
        if cached is not None:
            papers = [paper_from_json(x) for x in cached]
            self.add_log(phase, "search", tool=f"{tool}(cache)", query=query, result_count=len(papers))
            return papers
        self.budget.charge_literature()
        try:
            papers = self._retry(self.literature_search[provider], query, retmax)
        except BudgetExhausted:
            raise
        except Exception as exc:
            self.add_log(phase, "search", tool=tool, query=query, error=str(exc)[:300])
            return []
        self.cache.set("literature", key, [paper_to_json(x) for x in papers])
        self.add_log(phase, "search", tool=tool, query=query, result_count=len(papers))
        return papers

    def add_paper_source(self, paper: Paper) -> Source:
        key = (paper.pmid and f"pmid:{paper.pmid}") or (paper.doi and f"doi:{paper.doi.lower()}") or f"title:{paper.title.lower()}"
        if key in self.paper_sources:
            return self.sources[self.paper_sources[key]]
        text = f"{paper.title}\n{paper.abstract}"
        src = Source(
            source_id=self.new_id("src"), source_type=SourceType.SCIENTIFIC_LITERATURE, url=paper.url, doi=paper.doi,
            pmid=paper.pmid, pmcid=paper.pmcid, title=paper.title or None, site_or_publisher=paper.provider or None,
            authors=paper.authors, journal=paper.journal, publication_year=paper.year, language="en",
            retrieved_at=now(), retrieved_by_tool=f"{paper.provider or 'literature'}_search",
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            reliability_rank=SOURCE_RANK[SourceType.SCIENTIFIC_LITERATURE],
        )
        self.sources[src.source_id] = src
        self.paper_sources[key] = src.source_id
        self.paper_texts[src.source_id] = text
        return src

    def llm_parse(self, system: str, user: str, schema, phase: str, label: str):
        try:
            out = self.llm.parse(system, user, schema, counter=self.budget)
        except BudgetExhausted:
            raise
        except Exception as exc:
            self.add_log(phase, "read" if label.startswith("read") else "plan", tool=f"llm:{label}", error=str(exc)[:300])
            return None
        self.add_log(phase, "read" if label.startswith("read") else "plan", tool=f"llm:{label}")
        return out

    def add_source(self, doc: FetchedDoc, source_type: SourceType, tool: str) -> Source:
        src = Source(
            source_id=self.new_id("src"), source_type=source_type, url=doc.final_url or doc.url, title=doc.title or None,
            site_or_publisher=domain(doc.final_url or doc.url), retrieved_at=now(), retrieved_by_tool=tool,
            content_hash=doc.sha256, reliability_rank=SOURCE_RANK[source_type],
        )
        self.sources[src.source_id] = src
        return src
