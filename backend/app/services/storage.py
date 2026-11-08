from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.config import get_settings


class LocalObjectStorage:
    """Filesystem object storage. Replaceable with Replit App Storage or S3."""

    def __init__(self, root: Path | None = None) -> None:
        settings = get_settings()
        self.root = root or settings.storage_path
        self.root.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root.resolve() not in path.parents and path != self.root.resolve():
            raise ValueError("Invalid storage key")
        return path

    def save_bytes(self, data: bytes, *, suffix: str, folder: str) -> str:
        key = f"{folder}/{uuid4().hex}{suffix}"
        path = self._key_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key.replace("\\", "/")

    def save_upload(self, upload: UploadFile, *, folder: str, suffix: str) -> tuple[str, int, str]:
        data = upload.file.read()
        key = self.save_bytes(data, suffix=suffix, folder=folder)
        checksum = hashlib.sha256(data).hexdigest()
        return key, len(data), checksum

    def read_bytes(self, key: str) -> bytes:
        return self._key_path(key).read_bytes()

    def absolute_path(self, key: str) -> Path:
        return self._key_path(key)

    def exists(self, key: str) -> bool:
        return self._key_path(key).exists()


storage = LocalObjectStorage()
