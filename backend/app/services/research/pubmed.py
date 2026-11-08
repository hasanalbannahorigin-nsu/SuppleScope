from __future__ import annotations

from app.config import get_settings
from app.services.research.http import get_json


def search_pubmed(query: str, *, retmax: int = 8) -> list[dict]:
    settings = get_settings()
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(retmax),
        "email": settings.ncbi_email,
    }
    if settings.ncbi_api_key:
        params["api_key"] = settings.ncbi_api_key
    search = get_json("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params)
    ids = search.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    summary = get_json(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        params={
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json",
            "email": settings.ncbi_email,
            **({"api_key": settings.ncbi_api_key} if settings.ncbi_api_key else {}),
        },
    )
    result = summary.get("result", {})
    papers = []
    for pmid in ids:
        item = result.get(pmid) or {}
        year = None
        pubdate = item.get("pubdate") or ""
        if pubdate[:4].isdigit():
            year = int(pubdate[:4])
        papers.append(
            {
                "title": item.get("title"),
                "authors": ", ".join(
                    a.get("name") for a in (item.get("authors") or []) if a.get("name")
                )[:500]
                or None,
                "publication_year": year,
                "doi": next(
                    (aid.get("value") for aid in item.get("articleids", []) if aid.get("idtype") == "doi"),
                    None,
                ),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "source_api": "pubmed",
                "pmid": pmid,
                "study_type": item.get("pubtype", [None])[0] if item.get("pubtype") else None,
            }
        )
    return papers
