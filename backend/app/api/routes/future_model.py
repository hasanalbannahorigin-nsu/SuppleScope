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
    return {
        "name": row.name,
        "status": "NOT_INTEGRATED",
        "integrated": False,
        "title": "Custom AI Verification Model",
        "headline": "Coming Soon — Planned for Future Development.",
        "inputs": [
            "Advertisement claims",
            "Product identity",
            "Ingredient composition",
            "Scientific research dossier",
            "Patient medication and disease context",
        ],
        "expected_outputs": [
            "Claim verification",
            "Effectiveness assessment",
            "Supplement–drug risk prediction",
            "Supplement–disease risk prediction",
        ],
        "note": "This interface is a placeholder. No predictions are generated in the current phase.",
    }
