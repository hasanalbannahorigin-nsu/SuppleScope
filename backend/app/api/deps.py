from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.advertisement import Advertisement
from app.models.user import User


def get_owned_advertisement(db: Session, user: User, advertisement_id: str) -> Advertisement:
    ad = db.get(Advertisement, advertisement_id)
    if ad is None or ad.user_id != user.id:
        raise HTTPException(status_code=404, detail="Advertisement not found")
    return ad
