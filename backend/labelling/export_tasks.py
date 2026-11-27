"""VerifierInput JSONL (GET /api/advertisements/{id}/verifier-input.jsonl) -> Label Studio tasks JSON.

  python -m labelling.export_tasks eval/verifier_input_all.jsonl eval/label_studio_tasks.json
Each task keeps the full VerifierInput under data.record so the import step can rebuild gold JSONL.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.schemas.research.verifier import VerifierInput
from app.services.verification.text_format import encode


def to_task(vi: VerifierInput) -> dict:
    _, evidence = encode(vi)
    p = vi.product
    product = (f"{p.product_name or 'unknown product'} | brand {p.brand or '-'} | identity {p.identity_status.value} "
               f"| composition {p.composition_status.value} | product studies {p.finished_product_status.value}")
    return {"data": {"id": f"{vi.ad_id}-{vi.claim.claim_id}", "claim": vi.claim.text_original,
                     "claim_en": vi.claim.text_en or "", "product": product,
                     "evidence": evidence.replace("] [", "]\n[").replace(" [", "\n["),
                     "record": vi.model_dump(mode="json")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="one or more VerifierInput JSONL files")
    ap.add_argument("out")
    args = ap.parse_args()
    tasks = []
    for path in args.inputs:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                tasks.append(to_task(VerifierInput.model_validate_json(line)))
    Path(args.out).write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(tasks)} tasks -> {args.out}")


if __name__ == "__main__":
    main()
