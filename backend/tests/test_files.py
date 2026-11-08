from io import BytesIO

from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.services.files import validate_upload


def test_rejects_disallowed_extension():
    upload = UploadFile(filename="note.txt", file=BytesIO(b"hello"))
    try:
        validate_upload(upload)
        assert False, "should reject"
    except HTTPException as exc:
        assert exc.status_code == 400


def test_rejects_null_prefixed_non_video():
    upload = UploadFile(filename="ad.mp4", file=BytesIO(b"\x00not-a-video"))
    try:
        validate_upload(upload)
        assert False, "should reject"
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "MP4/MOV" in exc.detail


def test_accepts_iso_bmff_video_header():
    upload = UploadFile(
        filename="ad.mov",
        file=BytesIO(b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"),
    )
    validate_upload(upload)
