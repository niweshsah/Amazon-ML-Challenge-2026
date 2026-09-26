from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import sha256_file


def valid_embedding_manifest(manifest: dict[str, Any], expected: dict[str, Any] | None = None) -> bool:
    if not manifest.get("success"):
        return False
    if expected and any(manifest.get(key) != value for key, value in expected.items()):
        return False
    for item in manifest.get("files", []):
        path = Path(item["path"])
        if not path.exists() or path.stat().st_size != item["bytes"]:
            return False
        if item.get("sha256") and sha256_file(path) != item["sha256"]:
            return False
    return True

