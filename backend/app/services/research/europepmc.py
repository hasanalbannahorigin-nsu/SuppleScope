from __future__ import annotations

from app.services.research.http import get_json


def search_europe_pmc(query: str, *, page_size: int = 8) -> list[dict]:
    payload = get_json(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={"query": query, "format": "json", "pageSize": page_size},
    )
    papers = []
    for item in payload.get("resultList", {}).get("result", []) or []:
        year = item.get("pubYear")
        papers.append(
            {
                "title": item.get("title"),
                "authors": item.get("authorString"),
                "publication_year": int(year) if str(year).isdigit() else None,
                "doi": item.get("doi"),
                "url": item.get("fullTextUrlList", {}).get("fullTextUrl", [{}])[0].get("url")
                if item.get("fullTextUrlList")
                else f"https://europepmc.org/article/{item.get('source')}/{item.get('id')}",
                "source_api": "europepmc",
                "study_type": item.get("pubType"),
                "evidence_passages": [item["abstractText"]] if item.get("abstractText") else [],
            }
        )
    return papers
