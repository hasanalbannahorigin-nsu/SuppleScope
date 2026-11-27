"""Write one VerifierInput per claim (JSON Lines) for labelling, training and evaluation.

From the dossier files of an evaluation run (scripts/eval_real_ads.py writes <ad>_dossier.json):
  python scripts/export_verifier_inputs.py eval/verifier_input_all.jsonl --dossiers eval/out/20261127_120000
From the website's database (latest v2 dossier of every advertisement), no login needed:
  python scripts/export_verifier_inputs.py eval/verifier_input_all.jsonl
The output contains advertisement text, so it is git-ignored; do not commit it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.research.dossier import ResearchDossierV2  # noqa: E402
from app.services.verification.handoff import build_verifier_inputs  # noqa: E402


def dossiers_from_folders(folders: list[str]) -> list[ResearchDossierV2]:
    out = []
    for folder in folders:
        for path in sorted(Path(folder).glob("*_dossier.json")):
            out.append(ResearchDossierV2.model_validate_json(path.read_text(encoding="utf-8")))
    return out


def dossiers_from_database() -> list[ResearchDossierV2]:
    from app.database import SessionLocal
    from app.models.pipeline_v2 import ResearchDossierRecord

    latest: dict[str, ResearchDossierRecord] = {}
    with SessionLocal() as db:
        for row in db.query(ResearchDossierRecord).order_by(ResearchDossierRecord.created_at).all():
            latest[row.advertisement_id] = row  # later rows replace earlier ones
        return [ResearchDossierV2.model_validate(r.payload) for r in latest.values()]


def write_inputs(dossiers: list[ResearchDossierV2], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for dossier in dossiers:
            for vi in build_verifier_inputs(dossier):
                fh.write(json.dumps(vi.model_dump(mode="json"), ensure_ascii=False) + "\n")
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output JSONL, e.g. eval/verifier_input_all.jsonl")
    ap.add_argument("--dossiers", nargs="*", default=None, help="folder(s) with *_dossier.json files")
    args = ap.parse_args()
    dossiers = dossiers_from_folders(args.dossiers) if args.dossiers else dossiers_from_database()
    if not dossiers:
        sys.exit("No v2 research dossiers found. Run the research agent first (RESEARCH_AGENT=v2).")
    n = write_inputs(dossiers, Path(args.out))
    print(f"Wrote {n} claims from {len(dossiers)} advertisements to {args.out}")


if __name__ == "__main__":
    main()
