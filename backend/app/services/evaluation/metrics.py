"""Classification metrics and inter-annotator agreement, in plain Python (same numbers as scikit-learn's
precision_recall_fscore_support, confusion_matrix and cohen_kappa_score; Fleiss' kappa for 3+ annotators).
Kept dependency-free so the laptop and the test suite need no extra install; scripts use scikit-learn
when it is installed to cross-check."""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def confusion(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] | None = None) -> tuple[list[str], list[list[int]]]:
    labels = list(labels or sorted(set(gold) | set(pred)))
    idx = {lab: i for i, lab in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for g, p in zip(gold, pred):
        if g in idx and p in idx:
            m[idx[g]][idx[p]] += 1
    return labels, m


def per_class(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] | None = None) -> dict[str, dict]:
    labels, m = confusion(gold, pred, labels)
    out = {}
    for i, lab in enumerate(labels):
        tp = m[i][i]
        fp = sum(m[r][i] for r in range(len(labels))) - tp
        fn = sum(m[i]) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": sum(m[i])}
    return out


def report(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] | None = None) -> dict:
    if len(gold) != len(pred):
        raise ValueError("gold and pred differ in length")
    labels = list(labels or sorted(set(gold) | set(pred)))
    pc = per_class(gold, pred, labels)
    present = [lab for lab in labels if pc[lab]["support"] > 0]
    n = len(gold)
    acc = sum(g == p for g, p in zip(gold, pred)) / n if n else 0.0
    macro = sum(pc[lab]["f1"] for lab in present) / len(present) if present else 0.0
    weighted = sum(pc[lab]["f1"] * pc[lab]["support"] for lab in present) / n if n else 0.0
    _, m = confusion(gold, pred, labels)
    return {"n": n, "accuracy": acc, "macro_f1": macro, "weighted_f1": weighted, "per_class": pc,
            "labels": labels, "confusion": m, "cohen_kappa": cohen_kappa(gold, pred)}


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def fleiss_kappa(ratings: Sequence[Sequence[str]]) -> float:
    """ratings[i] = labels given to item i by its annotators (same number of annotators per item)."""
    items = [list(r) for r in ratings if r]
    if not items:
        return 0.0
    k = len(items[0])
    if k < 2 or any(len(r) != k for r in items):
        raise ValueError("Fleiss' kappa needs the same number (>= 2) of ratings per item")
    cats = sorted({x for r in items for x in r})
    n = len(items)
    p_j = {c: sum(r.count(c) for r in items) / (n * k) for c in cats}
    p_i = [(sum(r.count(c) ** 2 for c in cats) - k) / (k * (k - 1)) for r in items]
    p_bar = sum(p_i) / n
    pe = sum(v * v for v in p_j.values())
    return 1.0 if pe == 1 else (p_bar - pe) / (1 - pe)
