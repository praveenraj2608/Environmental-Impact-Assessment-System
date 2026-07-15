"""
text_extractor.py — Extracts plain text from PDF, DOCX, and TXT files.

Strategy:
  PDF  → PyMuPDF (fitz) — fast, no external tools required
  DOCX → python-docx
  TXT  → UTF-8 with latin-1 fallback

Unreadable files are skipped with a WARNING log entry rather than raising.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── PDF ───────────────────────────────────────────────────────────────────────

def _extract_pdf(path: Path) -> str:
    """Extract text from a PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error(
            "PyMuPDF is not installed. Run: pip install PyMuPDF"
        )
        return ""

    text_parts: list[str] = []
    try:
        with fitz.open(str(path)) as doc:
            for page in doc:
                page_text = page.get_text("text")  # type: ignore[attr-defined]
                if page_text.strip():
                    text_parts.append(page_text)
    except Exception as exc:
        logger.warning("Could not read PDF '%s': %s", path.name, exc)
        return ""

    return "\n".join(text_parts)


# ── DOCX ──────────────────────────────────────────────────────────────────────

def _extract_docx(path: Path) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        from docx import Document  # python-docx
    except ImportError:
        logger.error(
            "python-docx is not installed. Run: pip install python-docx"
        )
        return ""

    try:
        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as exc:
        logger.warning("Could not read DOCX '%s': %s", path.name, exc)
        return ""


# ── TXT ───────────────────────────────────────────────────────────────────────

def _extract_txt(path: Path) -> str:
    """Read a plain-text file (UTF-8, with latin-1 fallback)."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, OSError):
            continue
    logger.warning("Could not decode TXT file '%s'.", path.name)
    return ""


# ── Public API ────────────────────────────────────────────────────────────────

_EXTRACTORS = {
    ".pdf":  _extract_pdf,
    ".docx": _extract_docx,
    ".txt":  _extract_txt,
}


def extract_text(doc: dict[str, Any]) -> str:
    """
    Extract and return the plain text from a document record produced by
    document_loader.scan_knowledge_base().

    Returns an empty string if the file type is unsupported or unreadable.
    """
    path: Path = doc["path"]
    suffix = path.suffix.lower()
    extractor = _EXTRACTORS.get(suffix)

    if extractor is None:
        logger.warning("No extractor for file type '%s' (%s).", suffix, path.name)
        return ""

    text = extractor(path)
    if not text.strip():
        logger.warning("Extracted empty text from '%s'.", path.name)

    return text
