"""Write JSON Schema files for the research contracts to docs/schemas/.

    cd backend
    python scripts/export_research_schemas.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.research import (  # noqa: E402
    AdExtraction,
    ClaimResearch,
    IngredientResearch,
    OcrDocument,
    ProductResearch,
    ResearchDossierV2,
    VerifierInput,
    VerifierOutput,
)

OUT = Path(__file__).resolve().parents[2] / "docs" / "schemas"
MODELS = {
    "1_ocr_output": OcrDocument,
    "2_ad_extraction": AdExtraction,
    "3_product_research": ProductResearch,
    "4_ingredient_research": IngredientResearch,
    "5_claim_research": ClaimResearch,
    "6_research_dossier": ResearchDossierV2,
    "7_verifier_input": VerifierInput,
    "8_verifier_output": VerifierOutput,
}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, model in MODELS.items():
        path = OUT / f"{name}.schema.json"
        path.write_text(json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path)
