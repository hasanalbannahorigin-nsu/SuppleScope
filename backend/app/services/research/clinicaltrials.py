from __future__ import annotations

from app.services.research.http import get_json


def search_clinical_trials(query: str, *, page_size: int = 6) -> list[dict]:
    payload = get_json(
        "https://clinicaltrials.gov/api/v2/studies",
        params={"query.term": query, "pageSize": page_size},
    )
    rows = []
    for study in payload.get("studies", []) or []:
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        desc = proto.get("descriptionModule", {})
        status = proto.get("statusModule", {})
        nct = ident.get("nctId")
        rows.append(
            {
                "title": ident.get("briefTitle") or ident.get("officialTitle"),
                "url": f"https://clinicaltrials.gov/study/{nct}" if nct else None,
                "source_api": "clinicaltrials",
                "nct_id": nct,
                "study_type": (proto.get("designModule") or {}).get("studyType"),
                "outcomes": desc.get("briefSummary"),
                "publication_year": None,
                "status": (status.get("overallStatus")),
            }
        )
    return rows
