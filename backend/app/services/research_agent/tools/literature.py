"""PubMed (esearch + efetch with abstracts), Europe PMC (core results with abstracts) and the
ClinicalTrials.gov registry (API v2), all returned as Paper records.

PubMed effect searches can be limited to the strongest study designs with PUBMED_STUDY_FILTER
(Behnamian et al. 2026 filtered PubMed to reviews, RCTs and meta-analyses). Registry records are
kept apart by provider == "clinicaltrials": they describe a registered trial, not its findings,
unless the registry says results were posted.
"""
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
    nct_id: str | None = None
    has_results: bool | None = None


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


PUBMED_STUDY_FILTER = ("(randomized controlled trial[pt] OR meta-analysis[pt] OR systematic review[pt] "
                       "OR clinical trial[pt] OR review[pt])")
REGISTRY_PROVIDER = "clinicaltrials"


def with_study_filter(provider: str, query: str) -> str | None:
    """The same query limited to reviews, trials and meta-analyses; None when the provider has no such filter."""
    if provider == "pubmed":
        return f"({query}) AND {PUBMED_STUDY_FILTER}"
    return None


def clinicaltrials_search(query: str, retmax: int = 3) -> list[Paper]:
    """ClinicalTrials.gov API v2 /studies; each study becomes a Paper whose 'abstract' is the registry summary."""
    payload = get_json("https://clinicaltrials.gov/api/v2/studies",
                       params={"query.term": query, "pageSize": retmax, "format": "json"})
    return parse_clinicaltrials(payload)


def parse_clinicaltrials(payload: dict) -> list[Paper]:
    out = []
    for study in (payload or {}).get("studies", []) or []:
        proto = study.get("protocolSection") or {}
        ident = proto.get("identificationModule") or {}
        nct = ident.get("nctId")
        if not nct:
            continue
        status = proto.get("statusModule") or {}
        design = proto.get("designModule") or {}
        desc = proto.get("descriptionModule") or {}
        cond = (proto.get("conditionsModule") or {}).get("conditions") or []
        arms = (proto.get("armsInterventionsModule") or {}).get("interventions") or []
        enrol = (design.get("enrollmentInfo") or {}).get("count")
        allocation = ((design.get("designInfo") or {}).get("allocation") or "").replace("_", " ").lower()
        has_results = bool(study.get("hasResults"))
        start = ((status.get("startDateStruct") or {}).get("date") or "")[:4]
        lines = [
            f"Registered trial {nct}.",
            f"Overall status: {status.get('overallStatus') or 'unknown'}.",
            f"Study type: {design.get('studyType') or 'unknown'}" + (f", allocation {allocation}" if allocation else "") + ".",
            f"Phases: {', '.join(design.get('phases') or []) or 'not given'}.",
            f"Enrollment: {enrol if enrol is not None else 'not given'}.",
            f"Conditions: {', '.join(cond) or 'not given'}.",
            "Interventions: " + ("; ".join(f"{i.get('type', '')} {i.get('name', '')}".strip() for i in arms) or "not given") + ".",
            f"Results posted on the registry: {'yes' if has_results else 'no'}.",
        ]
        if desc.get("briefSummary"):
            lines.append("Summary: " + " ".join(desc["briefSummary"].split()))
        out.append(Paper(
            title=(ident.get("briefTitle") or ident.get("officialTitle") or nct).strip(),
            abstract="\n".join(lines), year=int(start) if start.isdigit() else None,
            publication_types=["Registered clinical trial"] + (["Results posted"] if has_results else []),
            url=f"https://clinicaltrials.gov/study/{nct}", provider=REGISTRY_PROVIDER, nct_id=nct, has_results=has_results,
        ))
    return out


def paper_to_json(p: Paper) -> dict:
    return asdict(p)


def paper_from_json(d: dict) -> Paper:
    return Paper(**d)


LITERATURE_SEARCH = {"pubmed": pubmed_search, "europepmc": europepmc_search, REGISTRY_PROVIDER: clinicaltrials_search}


def literature_providers_from_settings() -> list[str]:
    names = [n.strip().lower() for n in (get_settings().research_literature_providers or "").split(",") if n.strip()]
    # The registry is searched separately (RESEARCH_TRIAL_REGISTRY), never as a fallback for papers.
    return [n for n in names if n in LITERATURE_SEARCH and n != REGISTRY_PROVIDER] or ["pubmed"]
