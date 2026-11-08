"""Small on-disk JSON cache so repeated runs on the same ad do not repeat searches/fetches."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from app.config import ROOT_DIR, get_settings


class DiskCache:
    def __init__(self, root: str | Path | None = None, ttl_seconds: int = 7 * 24 * 3600) -> None:
        root = Path(root or get_settings().research_cache_dir)
        if not root.is_absolute():
            root = ROOT_DIR / "backend" / root if (ROOT_DIR / "backend").is_dir() else ROOT_DIR / root
        self.root, self.ttl = root, ttl_seconds

    def _path(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / namespace / f"{digest}.json"

    def get(self, namespace: str, key: str):
        path = self._path(namespace, key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if time.time() - payload.get("_at", 0) > self.ttl:
            return None
        return payload.get("value")

    def set(self, namespace: str, key: str, value) -> None:
        path = self._path(namespace, key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"_at": time.time(), "key": key, "value": value}, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass


class NullCache:
    def get(self, namespace: str, key: str):
        return None

    def set(self, namespace: str, key: str, value) -> None:
        return None
