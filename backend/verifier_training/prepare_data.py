"""Build verifier training data (JSONL rows: text_a, text_b, label, source, id) from public health
fact-checking sets plus SuppleScope's own labelled claims, in the same text format used at inference.

Label mapping (NOT_ASSESSABLE is assigned only by the rules, so the model learns four labels):
  SciFact (Wadden et al. 2020)   SUPPORT -> SUPPORTED, CONTRADICT -> CONTRADICTED, no evidence -> INSUFFICIENT_EVIDENCE
  HealthVer (Sarrouti et al. 2021) Supports -> SUPPORTED, Refutes -> CONTRADICTED, Neutral -> INSUFFICIENT_EVIDENCE
  PUBHEALTH (Kotonya & Toni 2020)  true -> SUPPORTED, mixture -> PARTIALLY_SUPPORTED, false -> CONTRADICTED,
                                   unproven -> INSUFFICIENT_EVIDENCE
  SuppleScope gold JSONL          VerifierInput records with a "gold_label" field (from Label Studio, see labelling/)

Usage (from backend/):
  python -m verifier_training.prepare_data --out data/verifier --scifact --healthver --pubhealth \
      --supplescope eval/verifier_gold.jsonl
Downloads go to --cache (default data/verifier/raw). If a download is blocked, put the files there by hand:
  SciFact: https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz (claims_*.jsonl, corpus.jsonl)
  HealthVer: healthver_train.csv / healthver_dev.csv / healthver_test.csv from github.com/sarrouti/HealthVer
  PUBHEALTH: Hugging Face dataset ImperialCollegeLondon/health_fact (needs `pip install datasets`)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import tarfile
from collections import Counter
from pathlib import Path

import httpx

from app.schemas.research.verifier import VerifierInput
from app.services.evidence.selector import Doc, SentenceSelector
from app.services.verification.text_format import claim_text, encode, evidence_text

LABELS = ["SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
HEALTHVER_URL = "https://raw.githubusercontent.com/sarrouti/HealthVer/master/data/healthver_{split}.csv"
_SELECT = SentenceSelector(mode="bm25", top_k=5)


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, timeout=120, follow_redirects=True) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in r.iter_bytes():
                fh.write(chunk)
    return dest


def _top(query: str, text: str, k: int = 5) -> list[tuple[str, str]]:
    return [("STUDY", x.sentence) for x in _SELECT.select(query, [Doc("d", "d", text)], k)]


# ---- SciFact ----------------------------------------------------------------------------------------------

def scifact(cache: Path) -> list[dict]:
    tgz = _download(SCIFACT_URL, cache / "scifact.tar.gz")
    files = {}
    with tarfile.open(tgz) as tar:
        for m in tar.getmembers():
            name = Path(m.name).name
            if name in ("corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl"):
                files[name] = tar.extractfile(m).read().decode("utf-8")
    corpus = {}
    for line in files["corpus.jsonl"].splitlines():
        d = json.loads(line)
        corpus[str(d["doc_id"])] = d
    rows = []
    for split in ("claims_train.jsonl", "claims_dev.jsonl"):
        for line in files[split].splitlines():
            c = json.loads(line)
            ev = c.get("evidence") or {}
            if ev:
                for doc_id, sets in ev.items():
                    doc = corpus.get(str(doc_id))
                    if not doc:
                        continue
                    label = sets[0]["label"]
                    idx = sorted({i for s in sets for i in s["sentences"]})
                    sents = [("STUDY", doc["abstract"][i]) for i in idx if i < len(doc["abstract"])]
                    rows.append({"id": f"scifact-{c['id']}-{doc_id}", "source": "scifact",
                                 "split_hint": "dev" if "dev" in split else "train",
                                 "text_a": claim_text(c["claim"]), "text_b": evidence_text(sents),
                                 "label": "SUPPORTED" if label == "SUPPORT" else "CONTRADICTED"})
            else:
                for doc_id in (c.get("cited_doc_ids") or [])[:1]:
                    doc = corpus.get(str(doc_id))
                    if doc:
                        rows.append({"id": f"scifact-{c['id']}-{doc_id}", "source": "scifact",
                                     "split_hint": "dev" if "dev" in split else "train", "text_a": claim_text(c["claim"]),
                                     "text_b": evidence_text(_top(c["claim"], " ".join(doc["abstract"]))),
                                     "label": "INSUFFICIENT_EVIDENCE"})
    return rows


# ---- HealthVer ---------------------------------------------------------------------------------------------

def healthver(cache: Path) -> list[dict]:
    rows = []
    mapping = {"supports": "SUPPORTED", "refutes": "CONTRADICTED", "neutral": "INSUFFICIENT_EVIDENCE"}
    for split in ("train", "dev", "test"):
        path = _download(HEALTHVER_URL.format(split=split), cache / f"healthver_{split}.csv")
        with path.open(encoding="utf-8", errors="replace", newline="") as fh:
            for i, r in enumerate(csv.DictReader(fh)):
                lab = mapping.get((r.get("label") or "").strip().lower())
                if not lab or not r.get("claim") or not r.get("evidence"):
                    continue
                rows.append({"id": f"healthver-{split}-{r.get('id') or i}", "source": "healthver", "split_hint": split,
                             "text_a": claim_text(r["claim"]), "text_b": evidence_text(_top(r["claim"], r["evidence"]) or
                                                                                         [("STUDY", r["evidence"][:600])]),
                             "label": lab})
    return rows


# ---- PUBHEALTH ---------------------------------------------------------------------------------------------

def pubhealth() -> list[dict]:
    from datasets import load_dataset

    try:
        ds = load_dataset("ImperialCollegeLondon/health_fact")
    except Exception:
        ds = load_dataset("ImperialCollegeLondon/health_fact", revision="refs/convert/parquet")
    names = {0: "CONTRADICTED", 1: "PARTIALLY_SUPPORTED", 2: "SUPPORTED", 3: "INSUFFICIENT_EVIDENCE"}
    rows = []
    for split in ds:
        for i, r in enumerate(ds[split]):
            lab = names.get(r.get("label"))
            if lab is None or not r.get("claim") or not r.get("main_text"):
                continue
            rows.append({"id": f"pubhealth-{split}-{r.get('claim_id') or i}", "source": "pubhealth",
                         "split_hint": {"validation": "dev"}.get(split, split),
                         "text_a": claim_text(r["claim"]), "text_b": evidence_text(_top(r["claim"], r["main_text"])),
                         "label": lab})
    return rows


# ---- SuppleScope ------------------------------------------------------------------------------------------------

def supplescope(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        gold = d.pop("gold_label", None)
        if gold not in LABELS:  # NOT_ASSESSABLE rows train nothing (rules assign it)
            continue
        vi = VerifierInput.model_validate(d)
        a, b = encode(vi)
        rows.append({"id": f"supplescope-{vi.ad_id}-{vi.claim.claim_id}", "source": "supplescope", "split_hint": "",
                     "text_a": a, "text_b": b, "label": gold})
    return rows


def split_rows(rows: list[dict], seed: int = 13, dev_frac: float = 0.1, test_frac: float = 0.1) -> dict[str, list[dict]]:
    out = {"train": [], "dev": [], "test": []}
    rnd = random.Random(seed)
    pending = []
    for r in rows:
        h = r.get("split_hint")
        if h in out and r["source"] != "supplescope":
            out[h].append(r)
        else:
            pending.append(r)
    # SuppleScope rows: split by ad so claims of one ad never sit on both sides
    by_ad: dict[str, list[dict]] = {}
    for r in pending:
        by_ad.setdefault(r["id"].rsplit("-", 1)[0], []).append(r)
    ads = sorted(by_ad)
    rnd.shuffle(ads)
    n = len(ads)
    for i, ad in enumerate(ads):
        part = "test" if i < round(n * test_frac) else "dev" if i < round(n * (test_frac + dev_frac)) else "train"
        out[part] += by_ad[ad]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/verifier")
    ap.add_argument("--cache", default="data/verifier/raw")
    ap.add_argument("--scifact", action="store_true")
    ap.add_argument("--healthver", action="store_true")
    ap.add_argument("--pubhealth", action="store_true")
    ap.add_argument("--supplescope", default=None, help="gold JSONL (VerifierInput + gold_label)")
    args = ap.parse_args()
    out, cache = Path(args.out), Path(args.cache)
    rows: list[dict] = []
    for flag, fn in (("scifact", lambda: scifact(cache)), ("healthver", lambda: healthver(cache)), ("pubhealth", pubhealth)):
        if getattr(args, flag):
            try:
                got = fn()
                print(f"{flag}: {len(got)} rows {dict(Counter(r['label'] for r in got))}")
                rows += got
            except Exception as exc:
                print(f"{flag}: FAILED ({exc}); see the docstring for manual download")
    if args.supplescope:
        got = supplescope(Path(args.supplescope))
        print(f"supplescope: {len(got)} rows {dict(Counter(r['label'] for r in got))}")
        rows += got
    if not rows:
        raise SystemExit("No data. Pass --scifact / --healthver / --pubhealth / --supplescope.")
    parts = split_rows(rows)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in part:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(part)} -> {out / (name + '.jsonl')}")


if __name__ == "__main__":
    main()
