"""
citation_service.py — Post-processes LLM responses to extract, validate,
and format citations, and computes a confidence label.

Design
------
- Deduplicates sources cited across retrieved chunks.
- Assigns a confidence label (High / Medium / Low) from average similarity.
- Formats the source list for clean display in the Streamlit UI.
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ── Confidence thresholds ─────────────────────────────────────────────────────
_HIGH_THRESHOLD   = 0.65
_MEDIUM_THRESHOLD = 0.40


def compute_confidence(chunks: list[dict[str, Any]]) -> str:
    """
    Derive a confidence label from the average similarity score of
    the retrieved chunks.

    Returns "High", "Medium", or "Low".
    """
    if not chunks:
        return "Low"

    similarities = [c.get("similarity", 0.0) for c in chunks]
    avg = sum(similarities) / len(similarities)

    if avg >= _HIGH_THRESHOLD:
        return "High"
    elif avg >= _MEDIUM_THRESHOLD:
        return "Medium"
    else:
        return "Low"


def format_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Deduplicate and sort chunks into a clean sources list.

    Each source entry:
        {
            "filename":    str   — document filename,
            "category":    str   — knowledge-base category,
            "similarity":  float — best similarity score for this source,
            "excerpt":     str   — first 200 chars of the most relevant chunk,
        }
    """
    if not chunks:
        return []

    # Group by source filename, keep the highest similarity chunk
    best: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        src = chunk.get("source", "unknown")
        if src not in best or chunk.get("similarity", 0) > best[src].get("similarity", 0):
            best[src] = chunk

    sources: list[dict[str, Any]] = []
    for src, chunk in sorted(
        best.items(), key=lambda kv: kv[1].get("similarity", 0), reverse=True
    ):
        excerpt = chunk.get("text", "")[:200].strip()
        if len(chunk.get("text", "")) > 200:
            excerpt += " …"
        sources.append(
            {
                "filename":   src,
                "category":   chunk.get("category", "General"),
                "similarity": round(chunk.get("similarity", 0.0), 3),
                "excerpt":    excerpt,
            }
        )

    return sources


def extract_cited_filenames(answer: str) -> list[str]:
    """
    Extract document names mentioned inside square brackets in the LLM answer.

    Example: "According to [WHO_Guidelines.pdf] …" → ["WHO_Guidelines.pdf"]
    """
    pattern = r"\[([^\[\]]+?\.[a-zA-Z]{2,5})\]"
    return list(set(re.findall(pattern, answer)))


def build_citation_result(
    answer: str,
    chunks: list[dict[str, Any]],
    model: str = "",
) -> dict[str, Any]:
    """
    Combine the LLM answer, retrieved sources, and confidence into the
    final result dict returned to the Streamlit page.

    Returns
    -------
    {
        "answer":     str   — LLM-generated text,
        "sources":    list  — formatted source list (from format_sources),
        "confidence": str   — "High" | "Medium" | "Low",
        "cited":      list  — filenames extracted from the answer text,
        "model":      str   — Ollama model name used,
    }
    """
    sources    = format_sources(chunks)
    confidence = compute_confidence(chunks)
    cited      = extract_cited_filenames(answer)

    # Log a warning if the LLM cited a source not in retrieved chunks
    retrieved_filenames = {c.get("source", "") for c in chunks}
    for cited_doc in cited:
        if cited_doc not in retrieved_filenames:
            logger.warning(
                "Potential hallucination: LLM cited '%s' which was not "
                "in the retrieved chunks.",
                cited_doc,
            )

    return {
        "answer":     answer,
        "sources":    sources,
        "confidence": confidence,
        "cited":      cited,
        "model":      model,
    }
