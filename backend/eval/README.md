# Research agent evaluation on the 7 real ads

`real_ads_gold.json` holds, per ad:

- `ad_gold`: what the ad itself shows (product, ingredients, claims), from the hand transcripts
  of the earlier OCR project. Used to score **ad extraction**.
- `transcript_lines`: the hand transcript, so extraction can be scored without OCR errors (`--text-source gold`).
- `research_gold`: the correct real-world answer for **product research**. These are left as
  `TO_FILL` on purpose: they must be checked by a person on the web, not guessed.
  Fill in for each ad:
  - `product_name`, `brand`, `manufacturer`: the real product the ad is for
  - `official_website`: the manufacturer's site, or `null` if none exists
  - `official_source_exists`: `true` if an official label / product page with ingredients exists online, else `false`
  - `official_ingredients`: the ingredient list from that official source (English names), or `[]`
  - `acceptable_identity_statuses`: e.g. `["VERIFIED","PROBABLE"]`, or `["IDENTITY_UNCERTAIN","NOT_FOUND"]`
    when the product genuinely cannot be found online
  - `checked_by`, and delete `_status`

Ads whose `research_gold` is still `TO_FILL` are reported but left out of the research scores.

Run (from `backend`, with the venv active and `.env` set up):

    python scripts/eval_real_ads.py --images D:\CSE499\ocr-agent\data\real\images

Outputs go to `eval/out/<run>/`: one dossier per ad, `report.md` with all metrics,
and `findings_review.csv` to mark by hand whether each kept finding is truly supported.
