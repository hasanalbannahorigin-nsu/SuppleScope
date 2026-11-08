"""Web search behind one interface, so the provider is a setting, not a rewrite.

    WEB_SEARCH_PROVIDER=duckduckgo   (free, no key; HTML results page)
    WEB_SEARCH_PROVIDER=tavily       (free tier; set TAVILY_API_KEY)

Add a provider by writing a class with `name` and `search(query, max_results)`
and registering it in PROVIDERS.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from app.config import get_settings
from app.services.research.http import USER_AGENT

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str | None
    provider: str
    rank: int


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int = 8) -> list[SearchHit]: ...


class _DdgHtmlParser(HTMLParser):
    """Parses html.duckduckgo.com results: <a class="result__a" href="//duckduckgo.com/l/?uddg=...">."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict] = []
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class") or ""
        if tag == "a" and "result__a" in cls:
            self.results.append({"url": _ddg_target(a.get("href") or ""), "title": "", "snippet": ""})
            self._in_title = True
        elif "result__snippet" in cls and self.results:
            self._in_snippet = True

    def handle_endtag(self, tag):
        if tag == "a":
            self._in_title = False
            self._in_snippet = False

    def handle_data(self, data):
        if self._in_title and self.results:
            self.results[-1]["title"] += data
        elif self._in_snippet and self.results:
            self.results[-1]["snippet"] += data


def _ddg_target(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target)
    return href


class DuckDuckGoSearch:
    name = "duckduckgo"

    def __init__(self, timeout: float = 25.0) -> None:
        self.timeout = timeout

    def search(self, query: str, max_results: int = 8) -> list[SearchHit]:
        with httpx.Client(timeout=self.timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
            response = client.post("https://html.duckduckgo.com/html/", data={"q": query, "kl": "wt-wt"})
            response.raise_for_status()
        parser = _DdgHtmlParser()
        parser.feed(response.text)
        hits = []
        for r in parser.results:
            url = r["url"]
            if not url.startswith("http") or "duckduckgo.com/y.js" in url:  # skip ads
                continue
            hits.append(SearchHit(title=r["title"].strip(), url=url, snippet=r["snippet"].strip() or None,
                                  provider=self.name, rank=len(hits) + 1))
            if len(hits) >= max_results:
                break
        return hits


class TavilySearch:
    name = "tavily"

    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        if not api_key:
            raise ValueError("TAVILY_API_KEY is not set")
        self.api_key, self.timeout = api_key, timeout

    def search(self, query: str, max_results: int = 8) -> list[SearchHit]:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                "https://api.tavily.com/search",
                json={"api_key": self.api_key, "query": query, "search_depth": "advanced", "max_results": max_results},
            )
            response.raise_for_status()
            payload = response.json()
        return [
            SearchHit(title=item.get("title") or "", url=item.get("url") or "", snippet=item.get("content"),
                      provider=self.name, rank=i)
            for i, item in enumerate(payload.get("results", []), start=1)
            if item.get("url")
        ]


def provider_from_settings() -> SearchProvider:
    s = get_settings()
    name = (s.web_search_provider or "duckduckgo").lower()
    if name == "tavily":
        return TavilySearch(s.tavily_api_key)
    if name == "duckduckgo":
        return DuckDuckGoSearch()
    raise ValueError(f"Unknown WEB_SEARCH_PROVIDER: {name}")


def hits_to_json(hits: list[SearchHit]) -> list[dict]:
    return [asdict(h) for h in hits]


def hits_from_json(rows: list[dict]) -> list[SearchHit]:
    return [SearchHit(**r) for r in rows]
