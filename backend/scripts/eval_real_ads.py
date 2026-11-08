"""Evaluate ad extraction + research agent v2 (milestone 1) on the real ads.

    cd backend
    python scripts/eval_real_ads.py --images D:\\CSE499\\ocr-agent\\data\\real\\images
    python scripts/eval_real_ads.py --images ... --only r03_dibedex --text-source gold

Metrics (see eval/README.md):
  ad extraction   product found, ingredient and claim precision/recall/F1 vs ad_gold
  identity        status per ad; top-1 accuracy vs research_gold (filled ads only)
  ingredients     official-ingredient recall/precision vs research_gold
  sources         share of kept findings from official / regulatory / third-party sources
  missing info    composition reported FOUND although no official source exists (should be 0)
  hallucination   findings rejected by validation; kept findings exported for manual review
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.schemas.research.common import InformationClass, ResearchStatus, SourceType  # noqa: E402
from app.schemas.research.dossier import ResearchDossierV2  # noqa: E402
from app.services.extraction.baseline import extract_from_text  # noqa: E402
from app.services.extraction.grounded import GroundedExtractor, baseline_to_ad_extraction  # noqa: E402
from app.services.llm.client import llm_from_settings  # noqa: E402
from app.services.ocr.document import ocr_document_from_text  # noqa: E402
from app.services.research_agent.runner import run_research  # noqa: E402
from app.services.text.bangla import match_key, similarity, span_score  # noqa: E402

GOLD = ROOT / "eval" / "real_ads_gold.json"
OFFICIAL = {SourceType.OFFICIAL_LABEL, SourceType.OFFICIAL_MANUFACTURER, SourceType.REGULATORY}


def prf(pred: list[str], gold: list[str], thresh: float = 80) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    used, tp = set(), 0
    for p in pred:
        for j, g in enumerate(gold):
            if j not in used and max(span_score(p, g), span_score(g, p)) >= thresh:
                used.add(j)
                tp += 1
                break
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gold) if gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def ocr_text(image: Path) -> str:
    from app.services.ocr.tesseract_provider import TesseractOcrProvider

    return TesseractOcrProvider().recognize(image).cleaned_text


def find_image(images: Path, ad_id: str) -> Path | None:
    for p in sorted(images.iterdir()):
        if p.name.startswith(ad_id):
            return p
    return None


def research_filled(rg: dict) -> bool:
    return bool(rg.get("checked_by")) and not str(rg.get("_status", "")).startswith("TO_FILL")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, help="Folder with r01_..., r02_... images (needed for --text-source ocr)")
    ap.add_argument("--text-source", choices=["ocr", "gold"], default="ocr", help="OCR the image, or use the hand transcript")
    ap.add_argument("--extractor", choices=["grounded_llm", "baseline"], default="grounded_llm")
    ap.add_argument("--only", nargs="*", help="Ad ids to run, e.g. r03_dibedex")
    ap.add_argument("--skip-research", action="store_true", help="Only evaluate ad extraction")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    if args.only:
        gold = [g for g in gold if g["id"] in args.only]
    llm = llm_from_settings()
    if llm is None and (args.extractor == "grounded_llm" or not args.skip_research):
        sys.exit("Set LLM_API_KEY / LLM_BASE_URL / LLM_MODEL in .env first (free Gemini works; see .env.example).")
    out = ROOT / "eval" / "out" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    rows, review = [], []
    failed: list[dict] = []
    for g in gold:
        ad_id = g["id"]
        print(f"== {ad_id}", flush=True)
        if args.text_source == "gold":
            text = "\n".join(g["transcript_lines"])
        else:
            img = find_image(args.images, ad_id) if args.images else None
            if img is None:
                print("   image not found, skipped")
                continue
            text = ocr_text(img)
        doc = ocr_document_from_text(ad_id, text, provider="tesseract" if args.text_source == "ocr" else "gold_transcript")
        t0 = time.time()
        if args.extractor == "grounded_llm":
            try:
                ad = GroundedExtractor(llm).run(doc).extraction
            except Exception as exc:  # keep going with the other ads; record the failure
                print(f"   extraction failed: {str(exc)[:300]}", flush=True)
                failed.append({"ad": ad_id, "error": f"extraction: {str(exc)[:300]}"})
                continue
        else:
            ad = baseline_to_ad_extraction(doc, extract_from_text(ocr_text=text))
        (out / f"{ad_id}_ad_extraction.json").write_text(ad.model_dump_json(indent=2), encoding="utf-8")
        ag = g["ad_gold"]
        row = {
            "ad": ad_id,
            "product_found": bool(ad.product_name) and max(span_score(ad.product_name.value, ag["product"]),
                                                            span_score(ag["product"], ad.product_name.value)) >= 70,
            "ing_prf": prf([i.name_original for i in ad.ingredients], ag["ingredients"]),
            "claim_prf": prf([c.text_original for c in ad.claims], ag["claims"]),
            "rejected_items": len(ad.rejected),
            "extract_seconds": round(time.time() - t0, 1),
        }
        if not args.skip_research:
            t1 = time.time()
            try:
                d = run_research(ad, llm=llm, use_cache=not args.no_cache)
            except Exception as exc:  # keep going; record the failure
                row["research_error"] = str(exc)[:300]
                rows.append(row)
                continue
            (out / f"{ad_id}_dossier.json").write_text(d.model_dump_json(indent=2), encoding="utf-8")
            row.update(score_research(d, g["research_gold"]))
            row["research_seconds"] = round(time.time() - t1, 1)
            for f in d.findings:
                if f.validation.passed:
                    src = next(s for s in d.sources if s.source_id == f.source_id)
                    review.append({"ad": ad_id, "finding_id": f.finding_id, "field": f.field, "value": f.value, "unit": f.unit or "",
                                   "information_class": f.information_class.value, "source_type": src.source_type.value,
                                   "url": src.url, "passage": f.passage, "truly_supported (y/n)": "", "about_right_product (y/n)": ""})
        rows.append(row)
        print("   " + json.dumps({k: v for k, v in row.items() if k != "ad"}, default=str), flush=True)

    with (out / "findings_review.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        if review:
            w = csv.DictWriter(fh, fieldnames=list(review[0]))
            w.writeheader()
            w.writerows(review)
    report = render_report(rows, args, failed)
    (out / "report.md").write_text(report, encoding="utf-8")
    (out / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(report)
    print(f"\nSaved to {out}")


def score_research(d: ResearchDossierV2, rg: dict) -> dict:
    ident = d.product.identity
    chosen = next((c for c in ident.candidates if c.candidate_id == ident.chosen_candidate_id), None)
    kept = [f for f in d.findings if f.validation.passed]
    src_type = {s.source_id: s.source_type for s in d.sources}
    comp = d.product.composition
    out = {
        "identity_status": ident.status.value,
        "identity_confidence": ident.confidence,
        "chosen": chosen.name if chosen else None,
        "candidates": len(ident.candidates),
        "composition_status": comp.status.value,
        "ingredients_found": [i.name for i in comp.ingredients],
        "official_website": d.product.brand_manufacturer.official_website,
        "findings_total": len(d.findings),
        "findings_rejected": len(d.findings) - len(kept),
        "rejected_not_in_source": sum(1 for f in d.findings if not f.validation.quote_in_source),
        "rejected_wrong_product": sum(1 for f in d.findings if f.validation.refers_to_product is False),
        "kept_official": sum(1 for f in kept if src_type[f.source_id] in OFFICIAL),
        "kept_third_party": sum(1 for f in kept if src_type[f.source_id] not in OFFICIAL),
        "composition_from_unverified": sum(1 for i in comp.ingredients
                                           if any(next(f for f in d.findings if f.finding_id == fid).information_class == InformationClass.UNVERIFIED
                                                  for fid in i.finding_ids)),
        "budget": d.budget.model_dump(include={"used_search_calls", "used_fetches", "used_llm_calls", "elapsed_seconds", "exhausted_reason"}),
        "gold_filled": research_filled(rg),
    }
    if out["gold_filled"]:
        ok_status = ident.status.value in (rg.get("acceptable_identity_statuses") or [])
        name_ok = bool(chosen and rg.get("product_name") and similarity(chosen.name, rg["product_name"]) >= 0.85)
        if rg.get("official_source_exists"):
            out["identity_correct"] = ok_status and name_ok
        else:
            out["identity_correct"] = ok_status and (name_ok or chosen is None)
        out["official_ing_prf"] = prf([i.name for i in comp.ingredients], rg.get("official_ingredients") or [])
        out["false_found"] = (not rg.get("official_source_exists")) and comp.status == ResearchStatus.FOUND
        out["website_correct"] = (
            (rg.get("official_website") is None and d.product.brand_manufacturer.official_website is None)
            or bool(rg.get("official_website") and d.product.brand_manufacturer.official_website
                    and match_key(rg["official_website"]).split("/")[0] in match_key(d.product.brand_manufacturer.official_website))
        )
    return out


def render_report(rows: list[dict], args, failed: list[dict] | None = None) -> str:
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    lines = [f"# Real-ad evaluation ({datetime.now():%Y-%m-%d %H:%M})", "",
             f"Text source: {args.text_source}; extractor: {args.extractor}; ads: {len(rows)}", "",
             "## Ad extraction (vs ad_gold)", "",
             "| ad | product | ingredient P/R/F1 | claim P/R/F1 | rejected |", "|---|---|---|---|---|"]
    for r in rows:
        ip, cp = r["ing_prf"], r["claim_prf"]
        lines.append(f"| {r['ad']} | {'yes' if r['product_found'] else 'no'} | {ip[0]:.2f}/{ip[1]:.2f}/{ip[2]:.2f} | "
                     f"{cp[0]:.2f}/{cp[1]:.2f}/{cp[2]:.2f} | {r['rejected_items']} |")
    lines += ["", f"Mean claim F1 {mean([r['claim_prf'][2] for r in rows]):.2f}; "
              f"product found {sum(r['product_found'] for r in rows)}/{len(rows)}", ""]
    res = [r for r in rows if "identity_status" in r]
    if res:
        lines += ["## Product research", "",
                  "| ad | identity | chosen product | composition | ingredients found | website | findings kept/rejected | official/3rd-party kept | gold? |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in res:
            lines.append(f"| {r['ad']} | {r['identity_status']} ({r['identity_confidence']:.2f}) | {r['chosen'] or '-'} | "
                         f"{r['composition_status']} | {', '.join(r['ingredients_found']) or '-'} | {r['official_website'] or '-'} | "
                         f"{r['findings_total'] - r['findings_rejected']}/{r['findings_rejected']} | "
                         f"{r['kept_official']}/{r['kept_third_party']} | {'yes' if r['gold_filled'] else 'TO_FILL'} |")
        filled = [r for r in res if r.get("gold_filled")]
        lines += [""]
        if filled:
            lines += [f"- Identity top-1 accuracy: {sum(bool(r['identity_correct']) for r in filled)}/{len(filled)}",
                      f"- Official ingredient recall (mean): {mean([r['official_ing_prf'][1] for r in filled]):.2f}; "
                      f"precision: {mean([r['official_ing_prf'][0] for r in filled]):.2f}",
                      f"- Official website correct: {sum(bool(r['website_correct']) for r in filled)}/{len(filled)}",
                      f"- Composition reported FOUND with no official source (should be 0): {sum(bool(r['false_found']) for r in filled)}"]
        else:
            lines += ["- research_gold not filled yet: accuracy metrics skipped (see eval/README.md)."]
        lines += [f"- Findings rejected by validation: {sum(r['findings_rejected'] for r in res)} of {sum(r['findings_total'] for r in res)} "
                  f"(not in source: {sum(r['rejected_not_in_source'] for r in res)}, wrong product: {sum(r['rejected_wrong_product'] for r in res)})",
                  f"- Composition entries built from unverified findings (should be 0): {sum(r['composition_from_unverified'] for r in res)}",
                  f"- Status counts: {dict(Counter(r['identity_status'] for r in res))}",
                  "- Unsupported-fact rate among KEPT findings: mark findings_review.csv by hand, then count 'n'.", ""]
    errors = [f"- {r['ad']}: research: {r['research_error']}" for r in rows if r.get("research_error")]
    errors += [f"- {f['ad']}: {f['error']}" for f in failed or []]
    if errors:
        lines += ["## Errors", "", "Re-run just these ads later with --only <ad id>.", ""] + errors
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
