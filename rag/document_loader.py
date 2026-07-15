"""
document_loader.py — Scans the Knowledge Base directory and returns
a list of document records with paths, categories, and SHA-256 hashes.

No text extraction happens here; that is delegated to text_extractor.py.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any

from rag.config import KNOWLEDGE_BASE_DIR, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


def _sha256(path: Path, chunk_bytes: int = 65536) -> str:
    """Return hex SHA-256 digest of a file without loading it all into RAM."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while True:
                block = fh.read(chunk_bytes)
                if not block:
                    break
                h.update(block)
    except OSError as exc:
        logger.warning("Cannot hash %s: %s", path, exc)
        return ""
    return h.hexdigest()


def scan_knowledge_base(base_dir: Path = KNOWLEDGE_BASE_DIR) -> list[dict[str, Any]]:
    """
    Recursively scan *base_dir* for supported documents.

    Returns a list of dicts:
        {
            "path":     Path  — absolute path to the file,
            "filename": str   — file name (no directory),
            "category": str   — immediate sub-folder name (e.g. "Air_Quality"),
            "hash":     str   — SHA-256 hex digest,
            "size":     int   — file size in bytes,
        }
    """
    if not base_dir.exists():
        logger.error(
            "Knowledge Base directory not found: %s. "
            "Create it and populate with PDF/DOCX/TXT files.",
            base_dir,
        )
        return []

    documents: list[dict[str, Any]] = []

    for file_path in sorted(base_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        # Derive category from the immediate child-folder of the knowledge base
        try:
            relative = file_path.relative_to(base_dir)
            category = relative.parts[0] if len(relative.parts) > 1 else "General"
        except ValueError:
            category = "General"

        doc = {
            "path":     file_path,
            "filename": file_path.name,
            "category": category,
            "hash":     _sha256(file_path),
            "size":     file_path.stat().st_size,
        }
        documents.append(doc)
        logger.debug("Found document: %s [%s]", file_path.name, category)

    logger.info(
        "Scanned %d document(s) across %d categor(y/ies) in '%s'.",
        len(documents),
        len({d["category"] for d in documents}),
        base_dir,
    )
    return documents


def get_file_hashes(documents: list[dict[str, Any]]) -> dict[str, str]:
    """Return a mapping of filename → hash for all documents."""
    return {d["filename"]: d["hash"] for d in documents}
