"""
rag_pipeline.py — Optimized top-level orchestrator.

Changes vs original
───────────────────
- Full per-stage timing instrumentation (Steps 1, 14)
- Prompt debug metadata extracted and stored in result
- Auto-retry with reduced TOP_K / context when generation exceeds 10s (Step 13)
- Duplicate-source deduplication in retrieval result
- timing dict attached to every return value
- ollama_status() cached for 10s so the status bar doesn't spam the server
"""

import logging
import time
from typing import Any, Callable

from rag.knowledge_manager import KnowledgeManager
from rag.retriever         import Retriever
from rag.prompt_builder    import build_prompt, _estimate_tokens, PromptResult
from rag.llm_service       import LLMService
from rag.citation_service  import build_citation_result
from rag.config import (
    RETRIEVAL_TOP_K, SIMILARITY_THRESHOLD,
    SLOW_GENERATION_THRESHOLD_SECS,
    RETRY_TOP_K, RETRY_MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHARS,
)

logger = logging.getLogger(__name__)

# Auto-retry threshold (Step 13) — driven by config
_SLOW_THRESHOLD_SECS = SLOW_GENERATION_THRESHOLD_SECS
_RETRY_TOP_K         = RETRY_TOP_K
_RETRY_MAX_CHARS     = RETRY_MAX_CONTEXT_CHARS


class RAGPipeline:
    """
    End-to-end RAG pipeline with full timing instrumentation.

    Instantiated once per Streamlit session (via @st.cache_resource).
    """

    def __init__(
        self,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> None:
        self.progress_callback = progress_callback

        self.knowledge_manager = KnowledgeManager(
            progress_callback=progress_callback
        )
        self.llm_service = LLMService()   # persistent session inside

        self.init_status: dict[str, Any] = {}
        self._initialized = False

        # Cached Ollama status (avoid hitting /api/tags on every page render)
        self._ollama_cache: dict[str, Any] = {}
        self._ollama_cache_ts: float = 0.0

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize(self, force_rebuild: bool = False) -> dict[str, Any]:
        self.init_status = self.knowledge_manager.initialize(
            force_rebuild=force_rebuild
        )
        if self.init_status.get("status") == "ok":
            self._initialized = True
            self.retriever = Retriever(
                embedding_service=self.knowledge_manager.embedding_service,
                vector_db=self.knowledge_manager.vector_db,
                top_k=RETRIEVAL_TOP_K,
                threshold=SIMILARITY_THRESHOLD,
            )
        return self.init_status

    @property
    def is_ready(self) -> bool:
        return self._initialized

    # ── Query ─────────────────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        prediction_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Answer a question with full timing instrumentation.

        Returns
        -------
        {
            "answer", "sources", "confidence", "cited", "model", "error",
            "timing": {
                "embed_secs", "retrieval_secs", "prompt_secs",
                "llm_secs", "total_secs",
                "first_token_secs",
                "est_prompt_tokens", "context_chars",
                "chunks_used",
            }
        }
        """
        t_total = time.perf_counter()

        if not self._initialized:
            return self._error_result("Knowledge base not initialized yet.")

        if not question.strip():
            return self._error_result("Please enter a question.")

        logger.info("─── RAG query: '%s…'", question[:80])

        # ── Step 1: Embed query ───────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            query_embedding = (
                self.knowledge_manager.embedding_service.embed_query(question)
            )
        except Exception as exc:
            logger.error("Embedding failed: %s", exc)
            return self._error_result(f"Embedding error: {exc}")
        embed_secs = time.perf_counter() - t0
        logger.info("  Embedding      %.3fs", embed_secs)

        # ── Step 2: Vector search ─────────────────────────────────────────────
        t0 = time.perf_counter()
        chunks = self.retriever.retrieve_with_context(
            question=question,
            prediction_context=prediction_context,
        )
        retrieval_secs = time.perf_counter() - t0
        logger.info("  Vector search  %.3fs  (%d chunks)", retrieval_secs, len(chunks))

        # ── Step 3: Build prompt ──────────────────────────────────────────────
        t0 = time.perf_counter()
        prompt_result = build_prompt(
            question=question,
            chunks=chunks,
            prediction_context=prediction_context,
            max_context_chars=MAX_CONTEXT_CHARS,
        )
        messages     = prompt_result.messages
        prompt_debug = prompt_result.debug
        prompt_secs  = time.perf_counter() - t0
        logger.info(
            "  Prompt build   %.3fs  est_tokens≈%d  context=%d chars",
            prompt_secs,
            prompt_debug.get("est_tokens", 0),
            prompt_debug.get("context_chars", 0),
        )

        # ── Step 4: LLM generation ────────────────────────────────────────────
        t0 = time.perf_counter()
        llm_result = self.llm_service.generate(messages, stream=True)
        llm_secs   = time.perf_counter() - t0
        llm_timing = llm_result.get("timing", {})
        logger.info("  LLM generation %.3fs", llm_secs)

        # ── Step 13: Auto-retry if too slow ───────────────────────────────────
        if llm_secs > _SLOW_THRESHOLD_SECS or llm_result.get("error") == "Timeout":
            logger.warning(
                "Generation took %.1fs (> %.0fs). Retrying with reduced context …",
                llm_secs, _SLOW_THRESHOLD_SECS,
            )
            t0 = time.perf_counter()
            retry_result  = build_prompt(
                question=question,
                chunks=chunks[:_RETRY_TOP_K],
                prediction_context=None,
                max_context_chars=_RETRY_MAX_CHARS,
            )
            llm_result = self.llm_service.generate(retry_result.messages, stream=True)
            llm_secs   = time.perf_counter() - t0
            logger.info("  LLM retry      %.3fs", llm_secs)

        total_secs = time.perf_counter() - t_total
        logger.info("  ─── Total      %.3fs", total_secs)

        # ── Step 5: Citations ─────────────────────────────────────────────────
        answer    = llm_result.get("answer", "")
        model     = llm_result.get("model", "")
        llm_error = llm_result.get("error")

        result = build_citation_result(answer, chunks, model)
        result["error"] = llm_error

        # Attach full timing for the debug panel (Step 11)
        result["timing"] = {
            "embed_secs":       round(embed_secs,      3),
            "retrieval_secs":   round(retrieval_secs,  3),
            "prompt_secs":      round(prompt_secs,      3),
            "llm_secs":         round(llm_secs,         3),
            "total_secs":       round(total_secs,        3),
            "first_token_secs": llm_timing.get("first_token"),
            "est_prompt_tokens": prompt_debug.get("est_tokens", 0),
            "context_chars":    prompt_debug.get("context_chars", 0),
            "chunks_used":      prompt_debug.get("chunks_used", len(chunks)),
        }

        return result

    # ── Utilities ─────────────────────────────────────────────────────────────

    def ollama_status(self) -> dict[str, Any]:
        """Return Ollama availability, cached for 10 seconds."""
        now = time.perf_counter()
        if now - self._ollama_cache_ts < 10.0 and self._ollama_cache:
            return self._ollama_cache

        available = self.llm_service.is_available()
        models    = self.llm_service.list_models() if available else []
        self._ollama_cache = {
            "available": available,
            "models":    models,
            "current":   self.llm_service.model,
        }
        self._ollama_cache_ts = now
        return self._ollama_cache

    def knowledge_base_status(self) -> dict[str, Any]:
        return {
            "initialized": self._initialized,
            "chunk_count": self.knowledge_manager.vector_db.document_count,
            "init_status": self.init_status,
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _error_result(message: str) -> dict[str, Any]:
        return {
            "answer":     f"⚠️ {message}",
            "sources":    [],
            "confidence": "Low",
            "cited":      [],
            "model":      "",
            "error":      message,
            "timing":     {},
        }
