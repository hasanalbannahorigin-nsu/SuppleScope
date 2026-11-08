from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.advertisement import Advertisement
from app.models.extraction import ExtractedClaim, ExtractedProduct
from app.models.research import ResearchJob, ResearchReport
from app.models.user import User
from app.security import get_current_user

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary")
def summary(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    ads = db.query(Advertisement).filter(Advertisement.user_id == user.id)
    ad_ids = [row.id for row in ads.all()]
    products = 0
    claims = 0
    reports = 0
    pending = 0
    if ad_ids:
        products = db.query(ExtractedProduct).filter(ExtractedProduct.advertisement_id.in_(ad_ids)).count()
        claims = db.query(ExtractedClaim).filter(ExtractedClaim.advertisement_id.in_(ad_ids)).count()
        reports = db.query(ResearchReport).filter(ResearchReport.advertisement_id.in_(ad_ids)).count()
        pending = (
            db.query(ResearchJob)
            .filter(ResearchJob.advertisement_id.in_(ad_ids), ResearchJob.status.in_(["queued", "running"]))
            .count()
        )
    recent = (
        db.query(Advertisement)
        .filter(Advertisement.user_id == user.id)
        .order_by(Advertisement.created_at.desc())
        .limit(8)
        .all()
    )
    return {
        "total_advertisements": len(ad_ids),
        "total_products": products,
        "total_claims": claims,
        "completed_reports": reports,
        "pending_jobs": pending,
        "recent": [
            {
                "id": row.id,
                "original_filename": row.original_filename,
                "media_type": row.media_type,
                "status": row.status,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in recent
        ],
    }
