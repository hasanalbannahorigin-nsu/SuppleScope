from __future__ import annotations

import logging

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 25.0
USER_AGENT = "SuppleScope/1.0 (university senior design; research aggregator)"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.8, min=1, max=6))
def get_json(url: str, *, params: dict | None = None, headers: dict | None = None) -> dict | list:
    merged = {"User-Agent": USER_AGENT, **(headers or {})}
    with httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
        response = client.get(url, params=params, headers=merged)
        response.raise_for_status()
        return response.json()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.8, min=1, max=6))
def get_text(url: str, *, params: dict | None = None) -> str:
    with httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
        response = client.get(url, params=params, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        return response.text
