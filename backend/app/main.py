from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import ROOT_DIR, get_settings
from app.database import Base, engine
from app.logging_config import configure_logging
from app.models import FutureModelConfiguration  # noqa: F401
from app.api.routes.advertisements import router as ads_router
from app.api.routes.auth import router as auth_router
from app.api.routes.dashboard import router as dashboard_router
from app.api.routes.future_model import router as future_router
from app.services.media_diagnostics import get_media_diagnostics

configure_logging()
settings = get_settings()
settings.storage_path.mkdir(parents=True, exist_ok=True)
Path(ROOT_DIR / "data").mkdir(parents=True, exist_ok=True)

limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])

app = FastAPI(
    title="SuppleScope API",
    description=(
        "Bangla-first supplement advertisement intelligence and biomedical research platform. "
        "The custom verification model is not integrated."
    ),
    version="0.1.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)
    get_media_diagnostics()


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        raise exc
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "name": "SuppleScope", "future_model": "NOT_INTEGRATED"}


@app.get("/api/system/media-status")
def media_status() -> dict:
    return get_media_diagnostics()


app.include_router(auth_router)
app.include_router(ads_router)
app.include_router(dashboard_router)
app.include_router(future_router)
