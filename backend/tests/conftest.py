"""Test defaults: no test may reach the internet.

The completed stack adds database look-ups (DSLD, DGDA/pharmacy queries, NIH ODS, NCCIH, openFDA) that
run by default in production. Here they are switched off unless a test enables them with fakes.
"""
import os

os.environ.setdefault("RESEARCH_PRODUCT_SOURCES", "")
os.environ.setdefault("RESEARCH_REFERENCE_SOURCES", "")
os.environ.setdefault("RESEARCH_OPENFDA", "false")
os.environ.setdefault("RESEARCH_TRIAL_REGISTRY", "false")
os.environ.setdefault("EVIDENCE_SELECTOR", "bm25")
os.environ.setdefault("IDISK_DB_PATH", "./tests/_no_idisk.sqlite")

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()
