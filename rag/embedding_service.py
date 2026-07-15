"""
embedding_service.py — Wraps SentenceTransformers for text embedding.

Provides a single EmbeddingService class that is instantiated once
(cached via Streamlit's @st.cache_resource) to avoid re-loading the
model on every query.
"""

import logging
import numpy as np
from typing import Union

from rag.config import EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Lightweight wrapper around SentenceTransformer.

    Usage
    -----
    >>> svc = EmbeddingService()
    >>> vectors = svc.embed_texts(["hello world", "air quality"])
    >>> vectors.shape
    (2, 384)
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str = EMBEDDING_DEVICE,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.device     = device
        self.batch_size = batch_size
        self._model     = None  # lazy-loaded

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load the SentenceTransformer model (once)."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(
                "Loading embedding model '%s' on device '%s' …",
                self.model_name, self.device,
            )
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Embedding model loaded successfully.")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{self.model_name}': {exc}"
            ) from exc

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_texts(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of strings and return a float32 NumPy array of shape
        (len(texts), embedding_dim).

        Parameters
        ----------
        texts     : Non-empty list of strings to embed.
        normalize : If True (default), L2-normalise embeddings so that
                    cosine similarity equals dot product.
        """
        self._load()
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        vectors = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns a 1-D float32 array of shape (embedding_dim,).
        """
        vectors = self.embed_texts([query])
        return vectors[0]

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        self._load()
        return self._model.get_sentence_embedding_dimension()  # type: ignore[union-attr]
