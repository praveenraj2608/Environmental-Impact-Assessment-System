"""
llm_service.py — Optimized Ollama REST client.

Changes vs original
───────────────────
- Persistent requests.Session (reused across all calls, no TCP reconnect overhead)
- Streaming enabled by default (first-token latency visible in logs)
- Ollama options enriched: top_p, repeat_penalty for quality + speed
- Timeout uses config value (already set to 120s by user)
- _session created once at __init__, not per-call
- Timing instrumentation for every stage
"""

import json
import logging
import time
import requests
from typing import Any

from rag.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_MAX_TOKENS,
    OLLAMA_TIMEOUT,
    OLLAMA_TOP_P,
    OLLAMA_REPEAT_PEN,
)

logger = logging.getLogger(__name__)

_GENERATE_URL = "/api/chat"
_TAGS_URL     = "/api/tags"


class LLMService:
    """
    Persistent Ollama client — one requests.Session for the lifetime of the
    process.  Never recreated per-request.

    Parameters
    ----------
    base_url    : Ollama server base URL.
    model       : Model name registered in Ollama.
    temperature : Sampling temperature.
    max_tokens  : num_predict (hard token cap for the response).
    timeout     : HTTP timeout in seconds.
    """

    def __init__(
        self,
        base_url:    str   = OLLAMA_BASE_URL,
        model:       str   = OLLAMA_MODEL,
        temperature: float = OLLAMA_TEMPERATURE,
        max_tokens:  int   = OLLAMA_MAX_TOKENS,
        timeout:     int   = OLLAMA_TIMEOUT,
    ) -> None:
        self.base_url    = base_url.rstrip("/")
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout     = timeout

        # ── Persistent session (Step 5) ───────────────────────────────────────
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ── Health check ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            resp = self._session.get(self.base_url + _TAGS_URL, timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        """Return model names currently pulled in Ollama."""
        try:
            resp = self._session.get(self.base_url + _TAGS_URL, timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        messages: list[dict[str, str]],
        stream: bool = True,          # streaming ON by default (Step 5)
    ) -> dict[str, Any]:
        """
        Send messages to Ollama and return a result dict with timing info.

        Returns
        -------
        {
            "answer":            str,
            "model":             str,
            "error":             str | None,
            "timing": {
                "request_start": float,
                "first_token":   float | None,
                "total":         float,
            }
        }
        """
        payload = {
            "model":    self.model,
            "messages": messages,
            "stream":   stream,
            "options": {
                "temperature":    self.temperature,
                "num_predict":    self.max_tokens,   # hard cap
                "top_p":          OLLAMA_TOP_P,
                "repeat_penalty": OLLAMA_REPEAT_PEN,
            },
        }

        url = self.base_url + _GENERATE_URL
        t_start = time.perf_counter()

        try:
            if stream:
                result = self._generate_stream(url, payload, t_start)
            else:
                result = self._generate_single(url, payload, t_start)

            elapsed = time.perf_counter() - t_start
            logger.info(
                "LLM generation | model=%s | stream=%s | %.2fs",
                self.model, stream, elapsed,
            )
            return result

        except requests.ConnectionError:
            logger.error("Cannot connect to Ollama at %s.", self.base_url)
            return {
                "answer": (
                    "⚠️ **AI service unavailable.** "
                    f"Run `ollama serve` and `ollama pull {self.model}`."
                ),
                "model":  self.model,
                "error":  "ConnectionError",
                "timing": {"request_start": t_start, "first_token": None,
                           "total": time.perf_counter() - t_start},
            }

        except requests.Timeout:
            elapsed = time.perf_counter() - t_start
            logger.error("Ollama timed out after %.1fs.", elapsed)
            return {
                "answer": (
                    f"⚠️ **Request timed out** ({elapsed:.0f}s). "
                    "Try a shorter question, or check that the model is loaded "
                    f"(`ollama run {self.model}`)."
                ),
                "model":  self.model,
                "error":  "Timeout",
                "timing": {"request_start": t_start, "first_token": None,
                           "total": elapsed},
            }

        except Exception as exc:
            logger.exception("Unexpected LLM error: %s", exc)
            return {
                "answer": f"⚠️ **Unexpected error:** {exc}",
                "model":  self.model,
                "error":  str(exc),
                "timing": {"request_start": t_start, "first_token": None,
                           "total": time.perf_counter() - t_start},
            }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _generate_single(
        self, url: str, payload: dict, t_start: float
    ) -> dict[str, Any]:
        """Non-streaming — returns the full response in one shot."""
        resp = self._session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data   = resp.json()
        answer = (
            data.get("message", {}).get("content", "")
            or data.get("response", "")
        ).strip()
        elapsed = time.perf_counter() - t_start
        return {
            "answer": answer,
            "model":  self.model,
            "error":  None,
            "timing": {
                "request_start": t_start,
                "first_token":   None,
                "total":         elapsed,
            },
        }

    def _generate_stream(
        self, url: str, payload: dict, t_start: float
    ) -> dict[str, Any]:
        """
        Streaming — accumulates tokens.
        Records time-to-first-token for latency diagnostics.
        """
        accumulated: list[str] = []
        first_token_time: float | None = None

        with self._session.post(
            url, json=payload, timeout=self.timeout, stream=True
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                token = (
                    chunk.get("message", {}).get("content", "")
                    or chunk.get("response", "")
                )
                if token:
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - t_start
                        logger.debug("First token in %.2fs", first_token_time)
                    accumulated.append(token)

                if chunk.get("done"):
                    break

        elapsed = time.perf_counter() - t_start
        return {
            "answer": "".join(accumulated).strip(),
            "model":  self.model,
            "error":  None,
            "timing": {
                "request_start": t_start,
                "first_token":   first_token_time,
                "total":         elapsed,
            },
        }
