"""
knowledge_manager.py — Orchestrates the full ingestion pipeline.

Called once at Streamlit startup (via @st.cache_resource).  Decides
whether to build a fresh index or reuse the persisted one.

Pipeline:
    Knowledge Base/ → document_loader → text_extractor → text_splitter
                    → embedding_service → vector_database (ChromaDB)
"""

import logging
from typing import Callable

from rag.document_loader  import scan_knowledge_base, get_file_hashes
from rag.text_extractor   import extract_text
from rag.text_splitter    import split_documents
from rag.embedding_service import EmbeddingService
from rag.vector_database  import VectorDatabase
from rag.config           import KNOWLEDGE_BASE_DIR, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class KnowledgeManager:
    """
    High-level manager that keeps the vector index in sync with the
    knowledge base on disk.

    Parameters
    ----------
    progress_callback : Optional callable(message: str, fraction: float)
                        for surfacing progress in a Streamlit spinner.
    """

    def __init__(
        self,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> None:
        self.progress_callback = progress_callback or (lambda msg, frac: None)

        self.embedding_service = EmbeddingService()
        self.vector_db         = VectorDatabase(self.embedding_service)

        self._initialized = False

    # ── Public API ────────────────────────────────────────────────────────────

    def initialize(self, force_rebuild: bool = False) -> dict:
        """
        Ensure the vector index is ready.

        Returns a status dict:
            {
                "status":     "ok" | "error",
                "message":    str,
                "doc_count":  int,
                "chunk_count": int,
                "rebuilt":    bool,
            }
        """
        try:
            return self._initialize(force_rebuild)
        except Exception as exc:
            logger.exception("KnowledgeManager initialization failed.")
            return {
                "status":      "error",
                "message":     str(exc),
                "doc_count":   0,
                "chunk_count": 0,
                "rebuilt":     False,
            }

    def _initialize(self, force_rebuild: bool) -> dict:
        self.progress_callback("Scanning knowledge base …", 0.05)
        documents = scan_knowledge_base(KNOWLEDGE_BASE_DIR)

        if not documents:
            return {
                "status":      "error",
                "message":     (
                    f"No documents found in '{KNOWLEDGE_BASE_DIR}'. "
                    "Add PDF/DOCX/TXT files and restart."
                ),
                "doc_count":   0,
                "chunk_count": 0,
                "rebuilt":     False,
            }

        file_hashes = get_file_hashes(documents)
        rebuilt     = False

        if force_rebuild or self.vector_db.needs_rebuild(file_hashes):
            self.progress_callback(
                f"Building index for {len(documents)} document(s) …", 0.15
            )
            rebuilt = True

            # Extract text
            docs_with_text: list[tuple] = []
            for i, doc in enumerate(documents):
                self.progress_callback(
                    f"Extracting: {doc['filename']} …",
                    0.15 + 0.35 * (i / len(documents)),
                )
                text = extract_text(doc)
                if text.strip():
                    docs_with_text.append((doc, text))

            if not docs_with_text:
                return {
                    "status":      "error",
                    "message":     "All documents extracted empty text.",
                    "doc_count":   len(documents),
                    "chunk_count": 0,
                    "rebuilt":     True,
                }

            # Split into chunks
            self.progress_callback("Splitting documents into chunks …", 0.50)
            chunks = split_documents(
                docs_with_text,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )

            # Embed + store
            self.progress_callback(
                f"Embedding {len(chunks)} chunks (this takes a few minutes on first run) …",
                0.60,
            )
            self.vector_db.build_index(chunks, file_hashes)
            chunk_count = len(chunks)
        else:
            chunk_count = self.vector_db.document_count

        self._initialized = True
        self.progress_callback("Knowledge base ready ✓", 1.0)

        return {
            "status":      "ok",
            "message":     (
                f"Loaded {len(documents)} document(s) "
                f"({chunk_count} chunks) from knowledge base."
            ),
            "doc_count":   len(documents),
            "chunk_count": chunk_count,
            "rebuilt":     rebuilt,
        }
