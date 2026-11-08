from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from app.config import get_settings

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".mp4", ".mov"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".mov"}

MAGIC = {
    "jpeg": (b"\xff\xd8\xff",),
    "png": (b"\x89PNG\r\n\x1a\n",),
}


def suffix_of(filename: str) -> str:
    return Path(filename).suffix.lower()


def media_type_for(filename: str) -> str:
    suffix = suffix_of(filename)
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    raise HTTPException(status_code=400, detail="Unsupported file type")


def _has_iso_bmff_file_type_box(header: bytes, file_size: int) -> bool:
    marker = header.find(b"ftyp", 4, 64)
    if marker < 0 or marker + 8 > len(header):
        return False
    box_start = marker - 4
    box_size = int.from_bytes(header[marker - 4 : marker], "big")
    return box_size >= 12 and box_start + box_size <= file_size


def validate_upload(upload: UploadFile) -> None:
    settings = get_settings()
    filename = upload.filename or ""
    suffix = suffix_of(filename)
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Allowed types: JPG, JPEG, PNG, MP4, MOV",
        )
    head = upload.file.read(64)
    upload.file.seek(0)
    upload.file.seek(0, 2)
    size = upload.file.tell()
    upload.file.seek(0)
    if suffix in {".jpg", ".jpeg"} and not any(head.startswith(sig) for sig in MAGIC["jpeg"]):
        raise HTTPException(status_code=400, detail="File content is not a valid JPEG image")
    if suffix == ".png" and not head.startswith(MAGIC["png"][0]):
        raise HTTPException(status_code=400, detail="File content is not a valid PNG image")
    if suffix in {".mp4", ".mov"} and not _has_iso_bmff_file_type_box(head, size):
        raise HTTPException(status_code=400, detail="File content is not a valid MP4/MOV video")
    if size > settings.max_upload_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds the {settings.max_upload_mb} MB size limit",
        )
    if size == 0:
        raise HTTPException(status_code=400, detail="Empty file")
