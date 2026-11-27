"""Index version folders: one folder per build, and active.json naming the one in use."""

import hashlib
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple

from llama_index.core import Document

import config

logger = logging.getLogger(__name__)

POINTER = "active.json"
MANIFEST = "manifest.json"
VERSION_NAME = re.compile(r"v\d+")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def new_version_dir(root: str) -> Path:
    """Create an empty version folder. Names are never reused: Chroma caches clients by path."""
    path = Path(root, f"v{time.time_ns()}")
    path.mkdir(parents=True)
    return path


def _settings(model_name: str) -> dict:
    return {
        "embedding_model": model_name,
        "chunk_sizes": list(config.CHUNK_SIZES),
        "chunk_overlap": config.CHUNK_OVERLAP,
    }


def _page_hashes(documents: List[Document]) -> dict:
    return {document.doc_id: text_hash(document.text) for document in documents}


def manifest(documents: List[Document], model_name: str) -> dict:
    return {**_settings(model_name), "pages": _page_hashes(documents)}


def compatible(data: dict, model_name: str) -> bool:
    """True if a version was built with the current embedding model and chunk settings."""
    return all(data.get(key) == value for key, value in _settings(model_name).items())


def pending_pages(data: dict, documents: List[Document]) -> int:
    """Number of new, changed and removed pages compared with a version's manifest."""
    old, new = data.get("pages", {}), _page_hashes(documents)
    return sum(old.get(page) != new.get(page) for page in old.keys() | new.keys())


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def _is_version_name(name) -> bool:
    return isinstance(name, str) and VERSION_NAME.fullmatch(name) is not None


def _read_pointer(root: str) -> dict:
    pointer = _read_json(Path(root, POINTER))
    if not _is_version_name(pointer.get("active")):
        raise ValueError(f"{POINTER} does not name a version folder")
    return pointer


def active_version(root: str) -> Optional[Tuple[Path, dict]]:
    """(folder, manifest) of the active version, or None if no version was ever activated."""
    if not Path(root, POINTER).exists():
        return None
    try:
        folder = Path(root, _read_pointer(root)["active"])
        return folder, _read_json(folder / MANIFEST)
    except OSError as e:
        raise ValueError(f"Cannot read the active version: {e}") from e


def activate(root: str, version_dir: Path, data: dict) -> None:
    """Write the manifest, then switch active.json to this version in one atomic step."""
    try:
        previous = _read_pointer(root)["active"]
    except (OSError, ValueError):
        previous = None
    (version_dir / MANIFEST).write_text(json.dumps(data), encoding="utf-8")
    temporary = Path(root, POINTER + ".tmp")
    temporary.write_text(
        json.dumps({"active": version_dir.name, "previous": previous}), encoding="utf-8"
    )
    os.replace(temporary, Path(root, POINTER))


def cleanup(root: str) -> None:
    """Delete version folders other than the active and previous ones."""
    keep = set()
    if Path(root, POINTER).exists():
        try:
            pointer = _read_pointer(root)
            if not Path(root, pointer["active"]).is_dir():
                raise ValueError("the active folder is missing")
        except (OSError, ValueError) as e:
            # without a trustworthy pointer, any folder could be the good one
            logger.warning("No cleanup, %s is unreadable: %s", POINTER, e)
            return
        keep = {pointer["active"], pointer.get("previous")}
    for path in Path(root).glob("v*"):
        if path.is_dir() and _is_version_name(path.name) and path.name not in keep:
            try:
                shutil.rmtree(path)
            except OSError as e:  # Windows can keep files locked; try again next time
                logger.warning("Could not delete %s yet: %s", path, e)
