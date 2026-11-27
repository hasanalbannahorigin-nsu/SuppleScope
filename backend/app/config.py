from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "SuppleScope"
    app_env: str = "development"
    secret_key: str = "change-this-to-a-long-random-string"
    access_token_expire_minutes: int = 120
    api_cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    database_url: str = "sqlite:///./data/supplescope.db"

    storage_root: str = "./data/storage"
    max_upload_mb: int = 80

    google_application_credentials: str = ""
    ocr_primary_provider: str = "tesseract"
    ocr_secondary_provider: str = "easyocr"  # or google_vision
    tesseract_cmd: str = ""
    tessdata_prefix: str = ""

    stt_provider: str = "none"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_stt_model: str = "whisper-1"
    faster_whisper_model: str = "medium"

    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"

    web_search_provider: str = "merged"  # duckduckgo | tavily | merged
    tavily_api_key: str = ""
    crossref_mailto: str = "supplescope@example.edu"
    openfda_api_key: str = ""
    ncbi_api_key: str = ""
    ncbi_email: str = "supplescope@example.edu"

    # baseline | banglabert | grounded_llm (needs an LLM key; falls back to baseline without one)
    extraction_backend: str = "baseline"
    # legacy = original research agent/scripted research; v2 = product-intelligence agent (schemas v2)
    research_agent: str = "legacy"

    # Research agent v2 limits (all configurable from .env)
    research_max_search_calls: int = 30
    research_max_fetches: int = 20
    research_max_llm_calls: int = 80
    research_max_retries_per_call: int = 2
    research_max_seconds: int = 1200
    research_max_rounds: int = 3
    research_fetch_max_chars: int = 60000
    # milestone 2: ingredient / finished-product / claim literature research
    research_max_literature_calls: int = 60
    research_max_lookup_calls: int = 40
    research_max_ingredients: int = 6
    research_papers_per_query: int = 5
    research_literature_providers: str = "pubmed,europepmc"
    research_cache_dir: str = "./data/cache/research"
    # Free LLM tiers limit requests per minute; wait this long between LLM calls.
    llm_min_seconds_between_calls: float = 4.0
    llm_timeout_seconds: float = 120.0
    banglabert_model_name: str = "csebuetnlp/banglabert"

    # ---- completed stack (see docs/STACK_COMPLETION.md) ----------------------------------
    # Literature: PubMed effect searches first try reviews / RCTs / meta-analyses / clinical trials
    # (publication-type filter, as in Behnamian et al. 2026), then fall back to an unfiltered search.
    research_pubmed_study_filter: bool = True
    # ClinicalTrials.gov registry search per ingredient and for the finished product (Med.ai ASK).
    research_trial_registry: bool = True
    research_trials_per_query: int = 3
    # openFDA: CAERS food/supplement adverse-event reports and drug labels naming an ingredient.
    research_openfda: bool = True
    # Product sources tried in phase 1 besides the web search: dsld, dgda, pharmacy (comma list).
    research_product_sources: str = "dsld,dgda,pharmacy"
    # Reference knowledge: iDISK2.0 (local SQLite built by scripts/load_idisk.py), NIH ODS, NCCIH.
    research_reference_sources: str = "idisk,nccih"  # add ods where the ODS site answers scripts
    idisk_db_path: str = "./data/knowledge/idisk.sqlite"
    # Bangla-Latin ingredient dictionary (CSV); leave empty for the bundled one.
    ingredient_dictionary_path: str = ""
    ingredient_match_threshold: float = 86.0

    # Evidence selection: top-k sentences per claim (MelAnalyze top-5 + CliVER re-ranking).
    evidence_top_k: int = 5
    # auto = sentence-transformers if installed, else BM25; bm25 = never load a neural model
    evidence_selector: str = "auto"
    evidence_embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    evidence_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # added for result/conclusion sentences, subtracted for title, background, aim and advice; 0 = relevance only
    evidence_role_bonus: float = 0.35

    # Claim verifier: fine-tuned checkpoints (comma list of local folders), majority vote.
    # Empty = rules only (plus the optional Gemini baseline below).
    verifier_model_dirs: str = ""
    verifier_max_length: int = 256
    # rules | ensemble | llm_baseline | ensemble+rules (default: ensemble when models exist, else rules)
    verifier_mode: str = "auto"

    # Speech: language hint for faster-whisper ("bn" for Bangla ads; empty = auto-detect)
    stt_language: str = "bn"
    faster_whisper_device: str = "cpu"
    faster_whisper_compute_type: str = "int8"

    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.api_cors_origins.split(",") if item.strip()]

    @property
    def storage_path(self) -> Path:
        path = Path(self.storage_root)
        if not path.is_absolute():
            path = ROOT_DIR / path
        return path

    @property
    def effective_llm_key(self) -> str:
        return self.llm_api_key or self.openai_api_key

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
