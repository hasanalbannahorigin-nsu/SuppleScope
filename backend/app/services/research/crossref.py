from __future__ import annotations

from app.config import get_settings
from app.services.research.http import get_json


def search_crossref(query: str, *, rows: int = 8) -> list[dict]:
    settings = get_settings()
    payload = get_json(
        "https://api.crossref.org/works",
        params={"query": query, "rows": rows, "mailto": settings.crossref_mailto},
        headers={"User-Agent": f"SuppleScope/1.0 (mailto:{settings.crossref_mailto})"},
    )
    papers = []
    for item in payload.get("message", {}).get("items", []) or []:
        title = " ".join(item.get("title") or [])
        year = None
        date_parts = (item.get("issued") or {}).get("date-parts") or [[]]
        if date_parts and date_parts[0]:
            year = date_parts[0][0]
        authors = []
        for author in item.get("author") or []:
            name = " ".join(part for part in [author.get("given"), author.get("family")] if part)
            if name:
                authors.append(name)
        papers.append(
            {
                "title": title,
                "authors": ", ".join(authors)[:500] or None,
                "publication_year": year,
                "doi": item.get("DOI"),
                "url": item.get("URL"),
                "source_api": "crossref",
                "publisher": item.get("publisher"),
            }
        )
    return papers
