# Completing the SuppleScope stack

This branch (`feature/complete-stack`) implements every layer of Table 3 ("Recommended technology stack") that
was marked Planned or Partly implemented. For each layer it uses the option that gave the best results in the
related studies. P1 to P6 are the papers in the stack report:

| Code | Study | What we took from it |
|---|---|---|
| P1 | Behnamian et al. 2026, *Information* | PubMed limited to reviews, RCTs and meta-analyses; openFDA; NIH ODS and NCCIH as reference sources |
| P2 | Hou et al. 2025, iDISK2.0-RAG, *JMIR* | iDISK2.0 knowledge base, concept normalisation, DSLD product labels, one-hop safety facts (RAG accuracy 95–99%) |
| P3 | Karkera et al. 2026, MelAnalyze, *BMC MIDM* | Top-5 abstract sentences per claim; RoBERTa as the strongest verifier (F1 0.91); 5 epochs, batch 8, learning rate 1e-5 |
| P4 | Liu et al. 2024, CliVER, *JAMIA Open* | Retrieve, then re-rank; RoBERTa + PubMedBERT (+T5) majority vote (F1 0.92) |
| P5 | Nguyen et al. 2026, Med.ai ASK, *JAMIA* | ClinicalTrials.gov as an evidence source |
| P6 | Bengali fake-news VLM study | EasyOCR for Bangla text in images |

All new code works without the heavy machine-learning extras and falls back to simple methods (BM25, rules),
so the laptop setup keeps working. The heavy extras are in `backend/requirements-ml.txt`.

## What each layer now does

| Layer (Table 3) | Implemented | Where | Switch |
|---|---|---|---|
| OCR | EasyOCR (bn+en) as the second engine next to Tesseract; Docker image ships tessdata_best | `services/ocr/easyocr_provider.py` | `OCR_SECONDARY_PROVIDER=easyocr` |
| Video and speech | faster-whisper **medium**, Bangla language hint, model loaded once per process | `services/speech.py` | `STT_PROVIDER=faster_whisper`, `STT_LANGUAGE=bn` |
| Name normalisation | Bangla–Latin dictionary (70 ingredients) + RapidFuzz; linked to iDISK2.0 ids; removes the duplicate ingredient (জাম vs *Syzygium cumini*) | `services/knowledge/normalise.py`, `data/ingredient_dictionary.csv` | `INGREDIENT_MATCH_THRESHOLD` |
| Web search and retrieval | DuckDuckGo + Tavily merged and de-duplicated; trafilatura main-text extraction with the full page text kept after it | `tools/search.py`, `tools/fetch.py` | `WEB_SEARCH_PROVIDER=merged` |
| Product sources | NIH DSLD label records (API), DGDA site queries, Bangladeshi pharmacy queries (Arogga, MedEasy, Osudpotro) | `tools/product_sources.py` | `RESEARCH_PRODUCT_SOURCES=dsld,dgda,pharmacy` |
| Supplement knowledge | iDISK2.0 CSV (and 1.x RRF) loaded into SQLite; NIH ODS Fact Sheet API; NCCIH Herbs at a Glance | `services/knowledge/`, `research_agent/reference_phase.py`, `scripts/load_idisk.py` | `RESEARCH_REFERENCE_SOURCES=idisk,nccih` (ODS answers 403 to scripts, so it is off by default) |
| Literature and safety | PubMed study-type filter with fallback; ClinicalTrials.gov registry (context only); openFDA CAERS reports for the product; openFDA drug labels naming an ingredient | `tools/literature.py`, `tools/openfda.py`, `evidence_phase.py` | `RESEARCH_PUBMED_STUDY_FILTER`, `RESEARCH_TRIAL_REGISTRY`, `RESEARCH_OPENFDA` |
| Evidence selection | Top-5 sentences per claim: BM25 + biomedical bi-encoder (S-PubMedBert-MS-MARCO), cross-encoder re-ranking; BM25 alone without the extras | `services/evidence/selector.py` | `EVIDENCE_SELECTOR`, `EVIDENCE_TOP_K` |
| Claim verifier | Fine-tuned RoBERTa + PubMedBERT majority vote, then the evidence-type rule (grades A–F, ingredient-only evidence capped at PARTIALLY_SUPPORTED); Gemini zero-shot baseline; rules-only mode | `services/verification/` | `VERIFIER_MODE`, `VERIFIER_MODEL_DIRS` |
| Model training | Data from SciFact, HealthVer, PUBHEALTH and your own labels; Hugging Face Trainer; free Colab/Kaggle T4 notebook | `backend/verifier_training/` | — |
| Patient risk | Bangladesh brand → generic table, drug classes, 23 cited rules (NCCIH, NIH ODS, LiverTox), dossier safety findings, RxNorm for unknown names | `services/patient_risk/` | — |
| Evaluation and labelling | Label Studio interface, export/import, Cohen's and Fleiss' kappa, macro-F1 and confusion matrix, ablation script | `backend/labelling/`, `services/evaluation/metrics.py`, `verifier_training/evaluate.py` | — |
| Deployment | Dockerfiles for back end and web app; Compose with Postgres, back end, nginx front end | `docker-compose.yml`, `backend/Dockerfile`, `frontend/Dockerfile` | `docker compose up -d --build` |

New API endpoints: `POST/GET /api/advertisements/{id}/verify|verification`, `POST/GET /api/advertisements/{id}/patient-risk`.
The Report page has two new tabs, **Claim Verification** and **Patient Risk**. Dossier schema is now 2.2.0
(new fields are optional, so older dossiers still load).

## Rules that keep the results honest

- Registered trials from ClinicalTrials.gov are stored as `REGISTERED_TRIAL_NO_RESULTS` and can only be CONTEXT for a claim.
- CAERS reports are counts with the caveat "not proof that the product caused the event".
- Reference facts (iDISK2.0, fact sheets, drug labels) are findings of class `REFERENCE`, quoted word for word, and never count as evidence that a product works.
- Only the rules assign NOT_ASSESSABLE. A model may choose a label only inside what the evidence grade allows.
- A medicine or ingredient that cannot be checked makes the patient risk UNKNOWN, never LOW.

## What you still need to do (cannot be done from the build machine)

1. **Install and check the live APIs.** `pip install -r backend/requirements.txt`, then
   `python backend/scripts/check_external_apis.py`. The DSLD and ClinicalTrials.gov field names were written from the
   public API structure but could not be called from the build machine; this script confirms them in one run.
2. **iDISK2.0 data.** Download the CSV files from github.com/houyurain/iDISK2.0 (folder `Data/`), check the data licence,
   then `python backend/scripts/load_idisk.py inspect <folder>` and `... load <folder>`. Do not commit the data.
3. **Train the verifier.** Open `backend/verifier_training/SuppleScope_verifier_training.ipynb` in Colab or Kaggle (T4 GPU),
   run all cells, download `verifier_models.zip`, unzip into `backend/models/` and set
   `VERIFIER_MODEL_DIRS=models/verifier-roberta,models/verifier-pubmedbert`. Until then the verifier runs in rules mode.
4. **Label claims.** `python scripts/export_verifier_inputs.py eval/verifier_input_all.jsonl --dossiers eval/out/<run>`
   (or without `--dossiers` to read the website database), then `python -m labelling.export_tasks ...`, label in Label Studio
   with `labelling/label_studio_config.xml` (two annotators), import with `python -m labelling.import_annotations ...`.
   That gold file feeds training and `python -m verifier_training.evaluate`.
5. **Pharmacist review.** `services/patient_risk/data/interaction_rules.csv` and `bd_brand_generic.csv` are starter tables;
   a pharmacist must check every row (and add local brands from medex.com.bd) before the tool is used with patients.
6. **Optional extras.** `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu` then
   `pip install -r backend/requirements-ml.txt` for EasyOCR, neural evidence selection and local verifier inference.
