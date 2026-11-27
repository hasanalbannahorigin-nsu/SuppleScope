from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.future import FutureModelConfiguration
from app.models.user import User
from app.security import get_current_user

router = APIRouter(prefix="/api/future-model", tags=["future-model"])


@router.get("/status")
def future_model_status(
    _: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> dict:
    row = db.query(FutureModelConfiguration).first()
    if row is None:
        row = FutureModelConfiguration()
        db.add(row)
        db.commit()
        db.refresh(row)
    from app.config import get_settings
    from app.services.verification.models import load_models_from_settings

    st = get_settings()
    dirs = [x.strip() for x in (st.verifier_model_dirs or "").split(",") if x.strip()]
    loaded = [m.name for m in load_models_from_settings()] if dirs else []
    mode = (st.verifier_mode or "auto").lower()
    effective = ("ensemble+rules" if loaded else "rules") if mode == "auto" else mode
    return {
        "name": row.name,
        "status": "INTEGRATED" if loaded else "RULES_ONLY",
        "integrated": True,
        "title": "Claim Verifier and Patient Risk",
        "headline": ("Fine-tuned ensemble with the evidence-type rule." if loaded else
                     "Evidence-type rules active; set VERIFIER_MODEL_DIRS to add the fine-tuned ensemble."),
        "verifier_mode": effective,
        "models_configured": dirs,
        "models_loaded": loaded,
        "labels": ["SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE", "NOT_ASSESSABLE"],
        "risk_levels": ["LOW", "MEDIUM", "HIGH", "UNKNOWN"],
        "inputs": [
            "Advertisement claims",
            "Product identity",
            "Ingredient composition",
            "Scientific research dossier (top-5 evidence sentences per claim)",
            "Patient medication and disease context",
        ],
        "expected_outputs": [
            "Claim verification (5 labels)",
            "Evidence directness grade (A-F)",
            "Supplement–drug risk",
            "Supplement–disease risk",
        ],
        "note": "Decision support only; the patient-risk rule table must be reviewed by a pharmacist.",
    }
