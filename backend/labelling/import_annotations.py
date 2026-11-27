"""Label Studio export (JSON) -> gold JSONL for training/evaluation, plus annotator agreement.

  python -m labelling.import_annotations export.json eval/verifier_gold.jsonl
Each output row is the task's VerifierInput plus "gold_label" (majority of annotators; ties -> the
first annotator listed with --adjudicator, else the item is left out and listed as a disagreement).
Prints Cohen's kappa (2 annotators) or Fleiss' kappa (3+), the agreement the evaluation plan asks for.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from app.services.evaluation.metrics import cohen_kappa, fleiss_kappa


def annotations(task: dict) -> dict[str, str]:
    """annotator -> verdict for one exported task."""
    out = {}
    for ann in task.get("annotations") or []:
        if ann.get("was_cancelled"):
            continue
        who = str((ann.get("completed_by") or {}).get("email") if isinstance(ann.get("completed_by"), dict)
                  else ann.get("completed_by"))
        for r in ann.get("result") or []:
            if r.get("from_name") == "verdict" and (r.get("value") or {}).get("choices"):
                out[who] = r["value"]["choices"][0]
    return out


def agreement(tasks: list[dict]) -> dict:
    per = [annotations(t) for t in tasks]
    people = sorted({p for a in per for p in a})
    pairs = {}
    for a, b in combinations(people, 2):
        both = [(x[a], x[b]) for x in per if a in x and b in x]
        if both:
            pairs[f"{a} vs {b}"] = {"n": len(both), "kappa": cohen_kappa([x for x, _ in both], [y for _, y in both])}
    full = [list(x.values()) for x in per if len(x) == len(people) and len(people) >= 2]
    return {"annotators": people, "pairwise_cohen": pairs,
            "fleiss": fleiss_kappa(full) if len(people) >= 3 and full else None}


def gold_rows(tasks: list[dict], adjudicator: str | None = None) -> tuple[list[dict], list[str]]:
    rows, disagreements = [], []
    for t in tasks:
        votes = annotations(t)
        if not votes:
            continue
        top = Counter(votes.values()).most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            if adjudicator and adjudicator in votes:
                label = votes[adjudicator]
            else:
                disagreements.append(t["data"].get("id", "?"))
                continue
        else:
            label = top[0][0]
        rows.append({**t["data"]["record"], "gold_label": label})
    return rows, disagreements


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("out")
    ap.add_argument("--adjudicator", default=None, help="annotator id/email whose label breaks ties")
    args = ap.parse_args()
    tasks = json.loads(Path(args.export).read_text(encoding="utf-8"))
    rows, dis = gold_rows(tasks, args.adjudicator)
    with Path(args.out).open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} gold rows -> {args.out}; {len(dis)} unresolved ties: {dis[:10]}")
    print(json.dumps(agreement(tasks), indent=2))


if __name__ == "__main__":
    main()
