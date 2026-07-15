"""
retriever.py — Converts a text query into ranked document chunks.

Usage
-----
    retriever = Retriever(embedding_service, vector_db)
    results   = retriever.retrieve("What is the WHO PM2.5 guideline?")
    # [{"text": "...", "source": "...", "category": "...", "similarity": 0.87}, ...]
"""

import logging
from typing import Any

from rag.embedding_service import EmbeddingService
from rag.vector_database   import VectorDatabase
from rag.config            import RETRIEVAL_TOP_K, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


class Retriever:
    """
    Embeds a query and retrieves semantically similar chunks from ChromaDB.

    Parameters
    ----------
    embedding_service : Shared EmbeddingService instance.
    vector_db         : Shared VectorDatabase instance.
    top_k             : Default number of results to return.
    threshold         : Default minimum similarity to include.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_db: VectorDatabase,
        top_k: int = RETRIEVAL_TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_db         = vector_db
        self.top_k             = top_k
        self.threshold         = threshold

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Embed *query* and return the most relevant chunks.

        Parameters
        ----------
        query     : User or enriched question string.
        top_k     : Override instance default.
        threshold : Override instance default.

        Returns
        -------
        List of chunk dicts sorted by similarity (desc):
            {"text", "source", "category", "similarity"}
        Empty list if no chunks meet the threshold.
        """
        if not query.strip():
            return []

        k   = top_k    if top_k    is not None else self.top_k
        thr = threshold if threshold is not None else self.threshold

        try:
            query_vec = self.embedding_service.embed_query(query)
            results   = self.vector_db.query(query_vec, top_k=k, threshold=thr)
        except Exception as exc:
            logger.error("Retrieval failed for query '%s': %s", query[:80], exc)
            return []

        logger.debug(
            "Retrieved %d chunk(s) for query: '%s…'",
            len(results), query[:60],
        )
        return results

    def retrieve_with_context(
        self,
        question: str,
        prediction_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Enrich the query with prediction context before retrieving.

        Combines the prediction values (city type, health impact, AQI …)
        into the query string so the embedding captures domain specifics.
        """
        if prediction_context:
            context_str = _format_context_for_embedding(prediction_context)
            enriched    = f"{context_str}\n\n{question}"
        else:
            enriched = question

        return self.retrieve(enriched)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_context_for_embedding(ctx: dict[str, Any]) -> str:
    """Serialize prediction context into a compact string for embedding."""
    parts: list[str] = []
    field_map = {
        "city_type":          "Area type",
        "health_impact":      "Health impact",
        "pollution_level":    "Pollution level",
        "risk_level":         "Risk level",
        "aqi":                "AQI",
        "trend":              "Pollution trend",
        "model_name":         "Prediction model",
        "confidence":         "Confidence",
    }
    for key, label in field_map.items():
        if key in ctx and ctx[key] is not None:
            parts.append(f"{label}: {ctx[key]}")

    return "Prediction context — " + "; ".join(parts) if parts else ""
