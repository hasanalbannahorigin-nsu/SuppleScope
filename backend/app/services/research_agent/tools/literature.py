"""PubMed (esearch + efetch with abstracts) and Europe PMC (core results with abstracts)."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field

from app.config import get_settings
from app.services.research.http import get_json, get_text


@dataclass
class Paper:
    title: str
    abstract: str
    pmid: str | None = None
    pmcid: str | None = None
    doi: str | None = None
    journal: str | None = None
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    publication_types: list[str] = field(default_factory=list)
    url: str | None = None
    provider: str = ""


def _ncbi_params() -> dict:
    s = get_settings()
    params = {"email": s.ncbi_email}
    if s.ncbi_api_key:
        params["api_key"] = s.ncbi_api_key
    return params


def pubmed_search(query: str, retmax: int = 8) -> list[Paper]:
    search = get_json("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                      params={"db": "pubmed", "term": query, "retmode": "json", "retmax": str(retmax),
                              "sort": "relevance", **_ncbi_params()})
    ids = search.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    xml = get_text("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                   params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml", **_ncbi_params()})
    return parse_pubmed_xml(xml)


def parse_pubmed_xml(xml: str) -> list[Paper]:
    root = ET.fromstring(xml)
    papers = []
    for art in root.findall(".//PubmedArticle"):
        cit = art.find("MedlineCitation")
        pmid = cit.findtext("PMID") if cit is not None else None
        a = cit.find("Article") if cit is not None else None
        if a is None:
            continue
        title = "".join(a.find("ArticleTitle").itertext()) if a.find("ArticleTitle") is not None else ""
        parts = []
        for t in a.findall("Abstract/AbstractText"):
            label = t.get("Label")
            body = "".join(t.itertext()).strip()
            parts.append(f"{label}: {body}" if label else body)
        year_text = a.findtext("Journal/JournalIssue/PubDate/Year") or (a.findtext("Journal/JournalIssue/PubDate/MedlineDate") or "")[:4]
        authors = []
        for au in a.findall("AuthorList/Author"):
            name = " ".join(x for x in (au.findtext("LastName"), au.findtext("Initials")) if x)
            if name:
                authors.append(name)
        doi = next((i.text for i in art.findall(".//ArticleIdList/ArticleId") if i.get("IdType") == "doi"), None)
        pmcid = next((i.text for i in art.findall(".//ArticleIdList/ArticleId") if i.get("IdType") == "pmc"), None)
        papers.append(Paper(
            title=title.strip(), abstract="\n".join(parts).strip(), pmid=pmid, pmcid=pmcid, doi=doi,
            journal=a.findtext("Journal/Title"), year=int(year_text) if year_text[:4].isdigit() else None,
            authors=authors[:12], publication_types=[p.text for p in a.findall("PublicationTypeList/PublicationType") if p.text],
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None, provider="pubmed",
        ))
    return papers


def europepmc_search(query: str, page_size: int = 8) -> list[Paper]:
    payload = get_json("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                       params={"query": query, "format": "json", "pageSize": page_size, "resultType": "core"})
    out = []
    for item in payload.get("resultList", {}).get("result", []) or []:
        year = item.get("pubYear")
        out.append(Paper(
            title=(item.get("title") or "").strip(), abstract=(item.get("abstractText") or "").strip(),
            pmid=item.get("pmid"), pmcid=item.get("pmcid"), doi=item.get("doi"),
            journal=(item.get("journalInfo") or {}).get("journal", {}).get("title"),
            year=int(year) if str(year).isdigit() else None,
            authors=[a.get("fullName") for a in (item.get("authorList") or {}).get("author", []) if a.get("fullName")][:12],
            publication_types=(item.get("pubTypeList") or {}).get("pubType", []),
            url=f"https://europepmc.org/article/{item.get('source')}/{item.get('id')}", provider="europepmc",
        ))
    return out


def paper_to_json(p: Paper) -> dict:
    return asdict(p)


def paper_from_json(d: dict) -> Paper:
    return Paper(**d)


LITERATURE_SEARCH = {"pubmed": pubmed_search, "europepmc": europepmc_search}


def literature_providers_from_settings() -> list[str]:
    names = [n.strip().lower() for n in (get_settings().research_literature_providers or "").split(",") if n.strip()]
    return [n for n in names if n in LITERATURE_SEARCH] or ["pubmed"]
