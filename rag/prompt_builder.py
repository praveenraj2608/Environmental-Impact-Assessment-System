"""
prompt_builder.py — Optimized prompt construction.

Changes vs original
───────────────────
- Compact system prompt (~300 chars vs ~700) → saves ~150 tokens per request
- Hard context limit: 2500 chars across all chunks (Step 3)
- Deduplication of identical chunk texts before formatting
- Trim whitespace in each chunk
- Prediction block compressed to one line (fewer tokens)
- max_context_chars default lowered to 2500
- Rough token estimator exposed for the debug panel
"""

import logging
import re
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)


class PromptResult(NamedTuple):
    """Returned by build_prompt — bundles the messages list with debug info."""
    messages: list[dict[str, str]]
    debug:    dict[str, Any]

# ── Compact system prompt (Step 8) ────────────────────────────────────────────
# Original was ~700 chars / ~175 tokens.
# New version is ~350 chars / ~90 tokens — saves ~85 tokens every call.
SYSTEM_PROMPT = (
    "You are an Environmental AI Consultant. "
    "Answer ONLY using the retrieved documents below. "
    "If the answer is not in the documents, say: "
    "'The knowledge base does not contain enough information.' "
    "Cite source filenames in [brackets]. "
    "Keep answers under 200 words."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (GPT rule-of-thumb)."""
    return max(1, len(text) // 4)


def _deduplicate_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove chunks with duplicate text content (keeps highest similarity)."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for chunk in sorted(chunks, key=lambda c: c.get("similarity", 0), reverse=True):
        # Normalise whitespace before fingerprinting
        fingerprint = re.sub(r"\s+", " ", chunk.get("text", "")).strip()[:120]
        if fingerprint and fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(chunk)
    return unique


def _build_context_block(
    chunks: list[dict[str, Any]],
    max_chars: int = 2500,
) -> tuple[str, int]:
    """
    Format deduplicated chunks into a numbered context block.

    Returns (context_string, total_chars_used).
    Hard-caps at max_chars to keep prompts small.
    """
    chunks = _deduplicate_chunks(chunks)

    if not chunks:
        return "No relevant documents found.", 0

    parts: list[str] = []
    total_chars = 0

    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        text   = re.sub(r"\s+", " ", chunk.get("text", "")).strip()

        # Truncate individual chunk if it would blow the budget
        remaining = max_chars - total_chars
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining] + "…"

        entry = f"[{i}] {source}:\n{text}"
        parts.append(entry)
        total_chars += len(entry)

        if total_chars >= max_chars:
            break

    return "\n\n".join(parts), total_chars


def _build_prediction_inline(ctx: dict[str, Any] | None) -> str:
    """
    One-line prediction summary to minimise tokens.
    Example: "Context: Industrial area, High health impact, AQI=120, Risk=Alert"
    """
    if not ctx:
        return ""
    fields = []
    mapping = [
        ("city_type",       "Area"),
        ("health_impact",   "Health"),
        ("pollution_level", "Pollution"),
        ("risk_level",      "Risk"),
        ("aqi",             "AQI"),
        ("trend",           "Trend"),
    ]
    for key, label in mapping:
        val = ctx.get(key)
        if val is not None:
            fields.append(f"{label}={val}")
    return "Prediction: " + ", ".join(fields) if fields else ""


# ── Public API ─────────────────────────────────────────────────────────────────

def build_prompt(
    question: str,
    chunks: list[dict[str, Any]],
    prediction_context: dict[str, Any] | None = None,
    max_context_chars: int = 2500,
) -> list[dict[str, str]]:
    """
    Build the Ollama /api/chat message list.

    Structure (Step 8)
    ------------------
    SYSTEM : compact role + rules
    USER   : [prediction line] + retrieved context + question + task

    Returns
    -------
    PromptResult(messages, debug) — messages is the list for Ollama;
    debug carries token/context metrics for the debug panel.
    """
    context_block, context_chars = _build_context_block(chunks, max_context_chars)
    prediction_line = _build_prediction_inline(prediction_context)

    user_parts: list[str] = []
    if prediction_line:
        user_parts.append(prediction_line)
    user_parts.append(f"RETRIEVED CONTEXT:\n{context_block}")
    user_parts.append(
        f"QUESTION: {question.strip()}\n\n"
        "Provide: 1) Direct answer 2) Brief explanation 3) Mitigation (if relevant) "
        "4) References [filename]. Under 200 words."
    )

    user_content = "\n\n".join(user_parts)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    total_tokens = _estimate_tokens(SYSTEM_PROMPT + user_content)

    logger.info(
        "Prompt built | chunks=%d (deduped) | context=%d chars | est_tokens≈%d",
        len(_deduplicate_chunks(chunks)), context_chars, total_tokens,
    )

    debug = {
        "context_chars": context_chars,
        "total_chars":   len(SYSTEM_PROMPT) + len(user_content),
        "est_tokens":    total_tokens,
        "chunks_used":   len(_deduplicate_chunks(chunks)),
        "system_tokens": _estimate_tokens(SYSTEM_PROMPT),
        "user_tokens":   _estimate_tokens(user_content),
    }

    return PromptResult(messages=messages, debug=debug)
