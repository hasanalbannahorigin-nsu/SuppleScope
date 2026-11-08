ANALYZE_STAGES = [
    "advertisement_uploaded",
    "video_preprocessing",
    "frames_extracted",
    "bangla_ocr_completed",
    "speech_transcription_completed",
    "information_extraction_completed",
]

RESEARCH_STAGES = [
    "product_identification_started",
    "ingredient_discovery_started",
    "scientific_research_started",
    "safety_research_completed",
    "research_dossier_generated",
]

ALL_STAGES = ANALYZE_STAGES + RESEARCH_STAGES


def empty_stages(names: list[str]) -> list[dict]:
    return [{"key": name, "status": "not_started", "detail": None} for name in names]


def set_stage(stages: list[dict], key: str, status: str, detail: str | None = None) -> list[dict]:
    updated = []
    for item in stages:
        if item["key"] == key:
            updated.append({"key": key, "status": status, "detail": detail})
        else:
            updated.append(item)
    return updated
