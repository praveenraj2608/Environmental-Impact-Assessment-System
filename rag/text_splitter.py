"""
text_splitter.py — Splits extracted text into overlapping chunks.

Uses a pure sliding-window approach (no LangChain dependency).
Each chunk carries metadata so the retriever can cite the source.
"""

import logging
from typing import Any

from rag.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def split_text(
    text: str,
    doc: dict[str, Any],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Split *text* into overlapping chunks and attach document metadata.

    Parameters
    ----------
    text         : Raw extracted text for one document.
    doc          : Document record from document_loader (must have
                   "filename" and "category" keys).
    chunk_size   : Maximum number of characters per chunk.
    chunk_overlap: Overlap in characters between adjacent chunks.

    Returns
    -------
    List of chunk dicts:
        {
            "text":        str  — chunk content,
            "source":      str  — source filename,
            "category":    str  — knowledge-base category,
            "chunk_index": int  — 0-based position within the document,
            "chunk_id":    str  — unique ID: "<filename>_chunk_<idx>",
        }
    """
    if not text.strip():
        return []

    if chunk_overlap >= chunk_size:
        logger.warning(
            "chunk_overlap (%d) >= chunk_size (%d). Resetting overlap to 0.",
            chunk_overlap, chunk_size,
        )
        chunk_overlap = 0

    filename = doc.get("filename", "unknown")
    category = doc.get("category", "General")

    chunks: list[dict[str, Any]] = []
    start = 0
    idx   = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                {
                    "text":        chunk_text,
                    "source":      filename,
                    "category":    category,
                    "chunk_index": idx,
                    "chunk_id":    f"{filename}_chunk_{idx}",
                }
            )
            idx += 1

        # Advance by (chunk_size - overlap) so the next chunk shares a tail
        step = chunk_size - chunk_overlap
        start += step

    logger.debug(
        "Split '%s' into %d chunk(s) (size=%d, overlap=%d).",
        filename, len(chunks), chunk_size, chunk_overlap,
    )
    return chunks


def split_documents(
    docs_with_text: list[tuple[dict[str, Any], str]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Convenience wrapper: split all (doc, text) pairs and return a flat list.

    Parameters
    ----------
    docs_with_text : List of (document_record, extracted_text) tuples.

    Returns
    -------
    Flat list of all chunks across all documents.
    """
    all_chunks: list[dict[str, Any]] = []
    for doc, text in docs_with_text:
        chunks = split_text(text, doc, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    logger.info(
        "Total chunks produced: %d from %d document(s).",
        len(all_chunks), len(docs_with_text),
    )
    return all_chunks
