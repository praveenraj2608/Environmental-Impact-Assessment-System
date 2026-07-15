"""
RAG-specific configuration constants.

All RAG settings live here — utils/config.py is never modified.

Optimized for qwen2.5:1.5b (small model, CPU inference).
Tune OLLAMA_MODEL / OLLAMA_MAX_TOKENS as hardware allows.
"""

from pathlib import Path

# ─── Project root (two levels up from rag/config.py) ─────────────────────────
_RAG_DIR = Path(__file__).resolve().parent
BASE_DIR  = _RAG_DIR.parent

# ─── Knowledge Base ───────────────────────────────────────────────────────────
KNOWLEDGE_BASE_DIR = BASE_DIR / "Knowledge Base"

# Supported document extensions (lower-cased)
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# ─── Vector Database ──────────────────────────────────────────────────────────
VECTOR_DB_DIR        = BASE_DIR / "vector_db"
VECTOR_DB_COLLECTION = "eia_knowledge_base"

# Hash cache file — used to detect when docs change (skip full rebuild)
HASH_CACHE_FILE = VECTOR_DB_DIR / "doc_hashes.json"

# ─── Text Chunking ────────────────────────────────────────────────────────────
# 500 chars → ~125 tokens per chunk. Keeps context window manageable.
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100   # reduced overlap → less duplicate content in context

# ─── Embeddings ───────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: 384-dim, 80 MB, very fast on CPU (~25ms per query)
EMBEDDING_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE     = "cpu"
EMBEDDING_BATCH_SIZE = 32

# ─── Retrieval ────────────────────────────────────────────────────────────────
# TOP_K=3  → 3 × 500 chars = 1500 chars context max ≈ 375 tokens
# Raising threshold to 0.35 keeps only clearly relevant chunks
RETRIEVAL_TOP_K          = 3
SIMILARITY_THRESHOLD     = 0.30
RETRIEVAL_CACHE_ENABLED  = True

# ─── Ollama / LLM ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL    = "http://localhost:11434"
OLLAMA_MODEL       = "qwen2.5:1.5b"   # ~1 GB, fast on CPU, good for RAG

# Step 9 — Token optimization
# num_predict=256 → response ≤ 256 tokens → ~5-8s on CPU for 1.5B model
OLLAMA_TEMPERATURE = 0.2    # low = deterministic, avoids rambling
OLLAMA_MAX_TOKENS  = 256    # hard cap on response length
OLLAMA_TOP_P       = 0.9    # nucleus sampling
OLLAMA_REPEAT_PEN  = 1.05   # slight repetition penalty
OLLAMA_TIMEOUT     = 120    # seconds (generous for cold-start)

# ─── Context limits (Step 3) ──────────────────────────────────────────────────
MAX_CONTEXT_CHARS   = 2500   # hard cap across all chunks in the prompt
MAX_PROMPT_TOKENS   = 1200   # guard — pipeline logs a warning if exceeded

# ─── Auto-retry thresholds (Step 13) ─────────────────────────────────────────
SLOW_GENERATION_THRESHOLD_SECS = 10.0   # trigger retry if LLM takes this long
RETRY_TOP_K                    = 2      # use fewer chunks on retry
RETRY_MAX_CONTEXT_CHARS        = 1200   # tighter context on retry

# ─── Logging ──────────────────────────────────────────────────────────────────
import logging
RAG_LOG_LEVEL = logging.INFO
