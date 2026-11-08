from __future__ import annotations

from app.config import get_settings
from app.services.research.http import get_json


def search_openfda_label(query: str) -> list[dict]:
    settings = get_settings()
    params = {"search": query, "limit": 5}
    if settings.openfda_api_key:
        params["api_key"] = settings.openfda_api_key
    try:
        payload = get_json("https://api.fda.gov/drug/label.json", params=params)
    except Exception:
        return []
    rows = []
    for item in payload.get("results", []) or []:
        openfda = item.get("openfda", {})
        rows.append(
            {
                "brand_name": (openfda.get("brand_name") or [None])[0],
                "generic_name": (openfda.get("generic_name") or [None])[0],
                "warnings": " ".join(item.get("warnings") or [])[:2000] or None,
                "drug_interactions": " ".join(item.get("drug_interactions") or [])[:2000] or None,
                "url": "https://open.fda.gov/apis/drug/label/",
                "source_api": "openfda",
            }
        )
    return rows
