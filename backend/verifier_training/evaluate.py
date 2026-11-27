"""Evaluate claim verifiers on SuppleScope's labelled claims and print the ablation table.

Input: JSONL of VerifierInput records with "gold_label" (made by labelling/import_label_studio.py).
  python -m verifier_training.evaluate --gold eval/verifier_gold.jsonl \
      --modes rules,ensemble,ensemble+rules,llm_baseline,llm_baseline+rules --out eval/out/verifier_eval.json
`ensemble*` modes need VERIFIER_MODEL_DIRS; `llm_baseline*` modes need LLM_API_KEY (free Gemini works).
Reports accuracy, macro-F1, per-class F1, Cohen's kappa against the gold labels and the confusion matrix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.schemas.research.verifier import VerdictLabel, VerifierInput
from app.services.evaluation.metrics import report
from app.services.verification.models import verifier_from_settings

ALL = [v.value for v in VerdictLabel]


def load_gold(path: Path) -> tuple[list[VerifierInput], list[str]]:
    inputs, gold = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            g = d.pop("gold_label")
            inputs.append(VerifierInput.model_validate(d))
            gold.append(g)
    return inputs, gold


def evaluate(inputs: list[VerifierInput], gold: list[str], modes: list[str], *, llm=None, models=None) -> dict:
    results = {}
    for mode in modes:
        try:
            v = verifier_from_settings(mode, llm=llm, models=models)
        except RuntimeError as exc:
            results[mode] = {"skipped": str(exc)}
            continue
        pred = [o.label.value for o in v.verify(inputs)]
        r = report(gold, pred, ALL)
        r["verifier"] = v.name
        r["predictions"] = pred
        results[mode] = r
    return results


def table(results: dict) -> str:
    lines = [f"{'mode':22} {'n':>4} {'acc':>6} {'macroF1':>8} {'kappa':>6}"]
    for mode, r in results.items():
        if "skipped" in r:
            lines.append(f"{mode:22} skipped: {r['skipped']}")
        else:
            lines.append(f"{mode:22} {r['n']:>4} {r['accuracy']:6.3f} {r['macro_f1']:8.3f} {r['cohen_kappa']:6.3f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--modes", default="rules,ensemble,ensemble+rules,llm_baseline,llm_baseline+rules")
    ap.add_argument("--out", default="eval/out/verifier_eval.json")
    args = ap.parse_args()
    inputs, gold = load_gold(Path(args.gold))
    results = evaluate(inputs, gold, [m.strip() for m in args.modes.split(",") if m.strip()])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(table(results))
    print(f"\nfull results: {out}")


if __name__ == "__main__":
    main()
