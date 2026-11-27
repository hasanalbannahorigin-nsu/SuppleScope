"""One live call to every external source the completed stack uses, to confirm the response formats
on a machine with internet access (the build machine could not reach them). No API key is printed.

  python scripts/check_external_apis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.knowledge import factsheets as F  # noqa: E402
from app.services.patient_risk import rxnorm  # noqa: E402
from app.services.research_agent.tools import openfda  # noqa: E402
from app.services.research_agent.tools import product_sources as PS  # noqa: E402
from app.services.research_agent.tools.literature import clinicaltrials_search, pubmed_search, with_study_filter  # noqa: E402


def check(name, fn, ok):
    try:
        out = fn()
        good = ok(out)
        print(f"[{'OK ' if good else 'BAD'}] {name}: {str(out)[:160]}")
    except Exception as exc:
        print(f"[ERR] {name}: {type(exc).__name__}: {str(exc)[:200]}")


def main() -> None:
    check("PubMed with study-type filter", lambda: [p.title for p in pubmed_search(
        with_study_filter("pubmed", '("Trigonella foenum-graecum"[tiab]) AND ("blood glucose")'), 3)], bool)
    check("ClinicalTrials.gov v2", lambda: [(p.nct_id, p.title) for p in clinicaltrials_search('"fenugreek"', 2)], bool)
    check("openFDA food/event (CAERS)", lambda: openfda.food_event_reports("Centrum").__dict__,
          lambda r: r["total_reports"] > 0)
    check("openFDA drug label mentions", lambda: [m.drug for m in openfda.drug_label_mentions(["St. John's Wort"])], bool)
    hits = []
    check("DSLD search", lambda: hits.extend(PS.DsldSearch().search("Centrum Silver", 3)) or [h.url for h in hits], bool)
    if hits:
        check("DSLD label", lambda: PS.fetch_any(hits[0].url).text[:300], lambda t: "Supplement Facts" in t)
    check("NIH ODS fact sheet (Iron)", lambda: (F.ods_fact_sheet("Iron") or F.FactSheet("", "", "", "")).safety_sentences[:2],
          bool)
    check("NCCIH (Ashwagandha)", lambda: (F.nccih_fact_sheet("Ashwagandha") or F.FactSheet("", "", "", "")).safety_sentences[:2],
          bool)
    check("RxNorm approximate match", lambda: rxnorm.generic_name("glucophage"), lambda g: bool(g))


if __name__ == "__main__":
    main()
