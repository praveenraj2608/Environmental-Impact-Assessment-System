"""
vector_database.py — ChromaDB-backed persistent vector store.

Responsibilities
----------------
- Build the index from chunks (embed → upsert)
- Persist to disk (vector_db/ directory)
- Query by embedding vector
- Detect when a rebuild is needed (hash comparison)
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Any

from rag.config import (
    VECTOR_DB_DIR,
    VECTOR_DB_COLLECTION,
    HASH_CACHE_FILE,
    SIMILARITY_THRESHOLD,
    RETRIEVAL_TOP_K,
)
from rag.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class VectorDatabase:
    """
    Thin wrapper around a ChromaDB persistent collection.

    Parameters
    ----------
    embedding_service : Pre-instantiated EmbeddingService.
    db_path           : Directory where ChromaDB persists data.
    collection_name   : Name of the ChromaDB collection.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        db_path: Path = VECTOR_DB_DIR,
        collection_name: str = VECTOR_DB_COLLECTION,
    ) -> None:
        self.embedding_service = embedding_service
        self.db_path           = db_path
        self.collection_name   = collection_name
        self._client           = None
        self._collection       = None

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _init_client(self) -> None:
        """Initialise ChromaDB PersistentClient (creates dir if absent)."""
        if self._client is not None:
            return
        try:
            import chromadb
        except ImportError:
            raise RuntimeError(
                "chromadb is not installed. Run: pip install chromadb"
            )
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.db_path))
        logger.info("ChromaDB client initialised at '%s'.", self.db_path)

    def _get_collection(self, create: bool = False):
        """Return (or optionally create) the ChromaDB collection.

        The result is cached — subsequent calls return the same object
        without hitting ChromaDB again (Step 6 / Step 7).
        """
        self._init_client()
        # Return cached instance if already loaded
        if self._collection is not None and not create:
            return self._collection
        if create:
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            try:
                self._collection = self._client.get_collection(
                    name=self.collection_name
                )
            except Exception:
                self._collection = None
        return self._collection

    # ── Hash management ───────────────────────────────────────────────────────

    def _load_cached_hashes(self) -> dict[str, str]:
        if HASH_CACHE_FILE.exists():
            try:
                return json.loads(HASH_CACHE_FILE.read_text())
            except Exception:
                pass
        return {}

    def _save_hashes(self, hashes: dict[str, str]) -> None:
        HASH_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        HASH_CACHE_FILE.write_text(json.dumps(hashes, indent=2))

    def needs_rebuild(self, current_hashes: dict[str, str]) -> bool:
        """
        Return True if the index must be rebuilt:
        - Hash cache file absent
        - Collection absent or empty
        - Any document has changed (hash mismatch)
        - New documents added / old ones removed
        """
        collection = self._get_collection(create=False)
        if collection is None:
            logger.info("Vector DB collection absent → rebuild required.")
            return True

        try:
            count = collection.count()
        except Exception:
            count = 0

        if count == 0:
            logger.info("Vector DB collection empty → rebuild required.")
            return True

        cached = self._load_cached_hashes()
        if cached != current_hashes:
            logger.info("Document hashes changed → rebuild required.")
            return True

        logger.info("Vector DB is up-to-date (skipping rebuild).")
        return False

    # ── Index building ────────────────────────────────────────────────────────

    def build_index(
        self,
        chunks: list[dict[str, Any]],
        file_hashes: dict[str, str],
    ) -> None:
        """
        Embed all chunks and upsert into ChromaDB.

        Parameters
        ----------
        chunks      : List of chunk dicts from text_splitter.
        file_hashes : Mapping filename → SHA-256 hash for caching.
        """
        if not chunks:
            logger.warning("No chunks provided — index will be empty.")
            return

        # Drop and recreate collection for a clean rebuild
        self._init_client()
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass  # Collection may not exist yet

        collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._collection = collection

        texts      = [c["text"]     for c in chunks]
        ids        = [c["chunk_id"] for c in chunks]
        metadatas  = [
            {
                "source":      c["source"],
                "category":    c["category"],
                "chunk_index": str(c["chunk_index"]),
            }
            for c in chunks
        ]

        logger.info("Embedding %d chunks …", len(chunks))
        batch_size = 256  # ChromaDB upsert batch limit
        embeddings = self.embedding_service.embed_texts(texts)

        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=embeddings[start:end].tolist(),
                metadatas=metadatas[start:end],
            )
            logger.debug("Upserted chunks %d–%d.", start, min(end, len(chunks)))

        self._save_hashes(file_hashes)
        logger.info(
            "Index built successfully: %d chunks stored in '%s'.",
            len(chunks), self.collection_name,
        )

    # ── Querying ──────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = RETRIEVAL_TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top-k most similar chunks.

        Parameters
        ----------
        query_embedding : 1-D float32 numpy array.
        top_k           : Maximum number of results.
        threshold       : Minimum cosine similarity to include.

        Returns
        -------
        List of dicts: {"text", "source", "category", "similarity"}
        sorted by similarity descending.
        """
        collection = self._get_collection(create=False)
        # Avoid calling collection.count() on every query (expensive).
        # Use document_count property which is cached after index build.
        if collection is None or self.document_count == 0:
            logger.warning("Vector DB is empty — no results returned.")
            return []

        try:
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, self.document_count),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("ChromaDB query failed: %s", exc)
            return []

        docs      = results.get("documents", [[]])[0]
        metas     = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: list[dict[str, Any]] = []
        for text, meta, dist in zip(docs, metas, distances):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score in [0, 1]
            similarity = max(0.0, 1.0 - dist)
            if similarity < threshold:
                continue
            retrieved.append(
                {
                    "text":       text,
                    "source":     meta.get("source", "unknown"),
                    "category":   meta.get("category", "General"),
                    "similarity": round(similarity, 4),
                }
            )

        return sorted(retrieved, key=lambda x: x["similarity"], reverse=True)

    @property
    def document_count(self) -> int:
        """Return the number of chunks currently stored.

        Uses a cached value after the first successful call so that
        hot-path queries never hit ChromaDB just to get the count.
        """
        if hasattr(self, "_cached_count") and self._cached_count > 0:
            return self._cached_count  # type: ignore[return-value]
        collection = self._get_collection(create=False)
        if collection is None:
            return 0
        try:
            count = collection.count()
            if count > 0:
                self._cached_count = count
            return count
        except Exception:
            return 0
