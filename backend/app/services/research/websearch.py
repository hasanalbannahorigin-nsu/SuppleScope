from __future__ import annotations

import logging
from html.parser import HTMLParser
from urllib.parse import quote_plus

from app.config import get_settings
from app.services.research.http import get_json, get_text

logger = logging.getLogger(__name__)


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._capture = False
        self._href = ""
        self._text = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href", "")
            if href.startswith("/url?q="):
                self._capture = True
                self._href = href.split("/url?q=", 1)[1].split("&")[0]
                self._text = []

    def handle_data(self, data):
        if self._capture:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._capture:
            title = " ".join(self._text).strip()
            if title and self._href.startswith("http"):
                self.links.append((title, self._href))
            self._capture = False


def search_web(query: str, *, max_results: int = 6) -> list[dict]:
    settings = get_settings()
    provider = (settings.web_search_provider or "duckduckgo").lower()
    if provider == "tavily" and settings.tavily_api_key:
        return _tavily(query, max_results=max_results)
    return _duckduckgo(query, max_results=max_results)


def _tavily(query: str, *, max_results: int) -> list[dict]:
    settings = get_settings()
    import httpx

    with httpx.Client(timeout=25) as client:
        response = client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
            },
        )
        response.raise_for_status()
        payload = response.json()
    return [
        {
            "title": item.get("title"),
            "url": item.get("url"),
            "snippet": item.get("content"),
            "source_type": "web_search",
            "provider": "tavily",
        }
        for item in payload.get("results", [])
    ]


def _duckduckgo(query: str, *, max_results: int) -> list[dict]:
    try:
        payload = get_json(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        )
    except Exception:
        logger.exception("DuckDuckGo instant answer failed")
        payload = {}
    results: list[dict] = []
    abstract = payload.get("AbstractText")
    abstract_url = payload.get("AbstractURL")
    if abstract and abstract_url:
        results.append(
            {
                "title": payload.get("Heading") or query,
                "url": abstract_url,
                "snippet": abstract,
                "source_type": "web_search",
                "provider": "duckduckgo",
            }
        )
    for topic in payload.get("RelatedTopics") or []:
        if isinstance(topic, dict) and topic.get("FirstURL"):
            results.append(
                {
                    "title": (topic.get("Text") or "")[:120],
                    "url": topic.get("FirstURL"),
                    "snippet": topic.get("Text"),
                    "source_type": "web_search",
                    "provider": "duckduckgo",
                }
            )
        if len(results) >= max_results:
            break
    if results:
        return results[:max_results]
    # HTML fallback (unofficial). If it fails, return empty rather than fabricating.
    try:
        html = get_text(f"https://html.duckduckgo.com/html/?q={quote_plus(query)}")
        parser = _LinkParser()
        parser.feed(html)
        for title, url in parser.links[:max_results]:
            results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": None,
                    "source_type": "web_search",
                    "provider": "duckduckgo_html",
                }
            )
    except Exception:
        logger.exception("DuckDuckGo HTML search failed")
    return results[:max_results]
