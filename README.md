# SuppleScope

SuppleScope is a Bangla-first Supplement Advertisement Intelligence & Evidence Platform. It is designed for analyzing Bangla supplement advertisements, extracting structure from image and video content, identifying product claims, and retrieving biomedical evidence without presenting a final medical verdict.

Tagline: "Look Beyond the Label. Discover the Evidence."

## 1. System architecture

The platform is split into modular layers so each component can be replaced independently:

- Frontend: React + TypeScript + Vite + Tailwind UI
- API layer: Python FastAPI with Pydantic schemas and route-level authorization
- Data layer: SQLAlchemy models with SQLite by default for local development, PostgreSQL recommended for production
- Storage layer: local filesystem object storage adapter with replaceable interface
- OCR services: local Tesseract Bengali-English (primary) and optional Google Cloud Vision
- Video services: FFmpeg + OpenCV frame extraction with scene detection and duplicate removal
- Speech transcription: local or API-based speech-to-text adapters
- Extraction pipeline: Bangla/English claim and product-entity extraction using a baseline extractor and adapter pattern
- Research agent: API-based tool-calling agent for product identity, ingredient discovery, and literature search
- Report layer: JSON dossier + PDF/export-friendly presentation with provenance and uncertainty tracking

## 2. Proposed architecture diagram

```text
[User Browser]
      |
      v
[React + TypeScript frontend]
      |
      v
[FastAPI REST API]
  |---- Auth / user routes
  |---- Advertisement upload routes
  |---- Analysis job routes
  |---- Research and report routes
  |---- Future model placeholder endpoints
      |
      +---- [SQLAlchemy models + DB]
      |
      +---- [Object storage]
      |
      +---- [OCR adapters]
      |        - Google Vision
      |        - Tesseract (ben)
      |
      +---- [Video processing]
      |        - FFmpeg
      |        - OpenCV
      |
      +---- [Speech-to-text adapters]
      |
      +---- [Information extraction]
      |
      +---- [Research tools]
               - Web search
               - PubMed / Europe PMC / Crossref / ClinicalTrials.gov / OpenFDA
               - Manufacturer and official label lookup
```

## 3. Folder structure

```text
SuppleScope/
├── .env.example
├── README.md
├── docker-compose.yml
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   ├── api/
│   │   │   ├── deps.py
│   │   │   └── routes/
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── jobs/
│   │   ├── logging_config.py
│   │   ├── main.py
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── security.py
│   │   ├── services/
│   │   ├── utils/
│   │   └── __init__.py
│   ├── alembic/
│   ├── alembic.ini
│   ├── requirements.txt
│   ├── pytest.ini
│   └── tests/
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── layouts/
│   │   ├── lib/
│   │   ├── pages/
│   │   ├── App.tsx
│   │   ├── index.css
│   │   ├── main.tsx
│   │   └── vite-env.d.ts
│   ├── package.json
│   ├── package-lock.json
│   ├── postcss.config.js
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   └── vite.config.ts
└── data/
    └── storage/
```

## 4. Database schema overview

The code is structured around a traceable research pipeline. The main relational entities are:

- users
- advertisements
- uploaded_files
- extracted_frames
- ocr_results
- speech_transcripts
- extracted_products
- extracted_claims
- ingredients
- product_compositions
- product_identification
- research_jobs
- research_sources
- scientific_papers
- evidence_passages
- ingredient_research
- interaction_research
- disease_risk_research
- patient_contexts
- research_reports
- future_model_configurations

Traceability flow:

Advertisement -> Extracted Claim -> Product -> Ingredient -> Research Source -> Research Report

This is enforced through explicit foreign-key relationships and modular models in the backend data layer.

## 5. Development milestones

### Phase 1: foundation
- Project structure and branding
- Frontend shell and landing page
- Authentication flow
- Dashboard and protected routes
- Database and upload system

### Phase 2: media intelligence
- Bangla OCR pipeline
- Image preprocessing and normalization
- Video frame extraction and scene filtering
- Speech transcription

### Phase 3: extraction and product identity
- Information extraction from OCR and speech
- Entity extraction from Bangla/English ads
- Claim extraction and product identification

### Phase 4: agentic research
- Web and literature tool integration
- Product and ingredient discovery
- Scientific paper retrieval and evidence tracking
- Risk and interaction research

### Phase 5: dossier and reporting
- Structured report generation
- Patient-context-aware safety section
- Dashboard presentation and report export

### Phase 6: verification and portability
- Testing and validation
- Documentation and local setup guides
- Deployment prep and VS Code portability

## 6. Required external API keys and dependencies

Required for a full production deployment:

- Google Cloud Vision API credentials
- OpenAI or equivalent API key for LLM and optionally STT
- Web search provider key if using Tavily or similar service
- PubMed / Europe PMC / Crossref / ClinicalTrials.gov access where applicable
- FFmpeg installation on the host machine
- Tesseract with Bengali data files (ben)

Optional / local-only:

- PostgreSQL service for the production database
- OpenAI-compatible provider for speech and LLM research tools
- Hugging Face model downloads for BanglaBERT or other NLP adapters

## 7. Local vs external services

### Works locally without external services
- React landing page and dashboard UI
- Authentication UI and route guard shell
- Upload handling and storage adapter
- Local file validation
- OCR normalization and extraction baseline
- Structured dossier schema validation
- SQLite local database for development
- FFmpeg/OpenCV frame extraction if installed locally
- Bangla-English OCR through Tesseract when the executable and both `ben` and `eng` traineddata are installed
- Automatic sampled/scene-change video frame selection and near-duplicate filtering with FFmpeg/OpenCV
- OCR quality measurement (CER/WER) when a ground-truth transcript is available

### Requires external APIs or models
- Google Cloud Vision OCR when enabled and service-account credentials are configured
- OpenAI and similar LLM APIs
- Search APIs and biomedical literature tools
- BanglaBERT or other transformer-based fine-tuning workflow
- Production PostgreSQL deployment
- Optional OpenAI Whisper transcription; local faster-whisper can be installed instead

### Not part of the current implementation scope
- Final custom verification model for claim validation
- Clinical diagnosis or treatment recommendation
- Prediction output from a custom model

## 8. Quick start

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Backend

```bash
cd backend
pip install -r requirements.txt
copy ../.env.example ../.env
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Windows media dependencies

The backend reports provider availability at startup and through `GET /api/system/media-status`. Install only the local tools you want to use; Google Cloud Vision and hosted speech services are optional and are not configured by these steps.

#### Tesseract Bengali-English OCR

In PowerShell, install the free UB Mannheim Tesseract build:

```powershell
winget install --id UB-Mannheim.TesseractOCR --exact
```

In the installer, include both English and Bengali language data. If Tesseract is already installed but either language is missing, download the corresponding traineddata files from the official `tessdata_fast` project into the Tesseract `tessdata` directory. For the default installation:

```powershell
$TessData = "$env:ProgramFiles\Tesseract-OCR\tessdata"
New-Item -ItemType Directory -Force $TessData | Out-Null
Invoke-WebRequest "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/ben.traineddata" -OutFile (Join-Path $TessData "ben.traineddata")
Invoke-WebRequest "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/eng.traineddata" -OutFile (Join-Path $TessData "eng.traineddata")
```

Restart PowerShell after installation. Verify the executable and both language packs:

```powershell
& "$env:ProgramFiles\Tesseract-OCR\tesseract.exe" --version
& "$env:ProgramFiles\Tesseract-OCR\tesseract.exe" --list-langs
```

The language list must contain `ben` and `eng`. Set the following in the repository `.env` if the executable is not on `PATH` or the traineddata directory is nonstandard (adjust paths for a custom installation):

```dotenv
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata
```

Restart the backend after changing `.env`. Missing Tesseract, `ben`, or `eng` is reported explicitly; the application does not fabricate OCR output.

#### FFmpeg, FFprobe, and OpenCV video processing

Install the free FFmpeg shared build:

```powershell
winget install --id Gyan.FFmpeg.Shared --exact
```

Restart PowerShell and verify both command-line tools:

```powershell
ffmpeg -version
ffprobe -version
```

If the executables are not on `PATH`, set their full paths in `.env`, for example `FFMPEG_PATH=C:\path\to\ffmpeg.exe` and `FFPROBE_PATH=C:\path\to\ffprobe.exe`. OpenCV is a backend Python dependency. Frame selection requires FFprobe and OpenCV; extracting audio for transcription requires FFmpeg.

#### Check actual provider status

With the backend running, query its live diagnostics:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/system/media-status | ConvertTo-Json -Depth 6
```

The backend startup log reports the same availability and actionable reasons. Speech transcription is optional and disabled by default (`STT_PROVIDER=none`). For API transcription, set `STT_PROVIDER=openai_whisper_api` and `OPENAI_API_KEY`; this uses an external paid service and is not required for OCR or frame selection. For local transcription, install `faster-whisper` and set `STT_PROVIDER=faster_whisper`.

Google Cloud Vision is an optional OCR fallback/compare provider. Set `GOOGLE_APPLICATION_CREDENTIALS` to a service-account JSON file and install the optional SDK with `pip install google-cloud-vision`. OCR first checks the configured provider availability and then uses the configured secondary provider. Keep credentials out of the repository.

### Phase 2 validation

Run backend tests from `backend` with `python -m pytest`. The Bangla sample-image integration test runs only when Tesseract, `ben`, and `eng` are available; it calculates CER/WER against the checked-in ground-truth text. The deterministic provider, metric, frame-selection, and mocked transcription tests do not require external APIs. To run the sample OCR test explicitly with output details:

```powershell
python -m pytest -s tests/test_ocr_providers.py
```

### PostgreSQL (optional)

```bash
docker compose up -d postgres
```

## 9. Notes on current implementation status

This repository already contains the initial SuppleScope project scaffold, including:

- landing page with branding and future-model placeholder
- dashboard and upload pages
- authentication routes and protected app shell
- FastAPI backend with upload/research endpoints
- SQLAlchemy models and service abstractions
- OCR normalization and extraction baseline tests
- Tesseract Bengali/English OCR adapter, Google Vision adapter, preprocessing, confidence, and source-linked OCR results
- FFmpeg/OpenCV frame selection with timestamps, scene-change sampling, and near-duplicate filtering
- Optional replaceable speech adapters with timestamped segments
- Sample-image CER/WER evaluation and mocked video/transcription tests
- future-model status card clearly marked as not integrated

The current project is intentionally designed to keep the custom AI verification model outside the active implementation scope while enabling the full research and evidence-gathering workflow.

# SuppleScope