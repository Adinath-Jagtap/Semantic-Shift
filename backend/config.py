"""
Configuration module — loads environment variables, validates on import,
and exposes a thread-safe mutable settings singleton.

All other modules import `settings` from here. The GROQ_API_KEY is validated
at import time so the server fails fast with a clear message rather than
crashing silently on the first request.
"""

import os
import sys
import threading

from dotenv import load_dotenv
load_dotenv()


class _Settings:

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # --- Required ---
        self.groq_api_key: str = os.getenv("GROQ_API_KEY", "")
        if not self.groq_api_key:
            print(
                "\n[FATAL] GROQ_API_KEY is not set. "
                "Copy .env.example to .env and add your key.\n",
                file=sys.stderr,
            )
            sys.exit(1)

        self.groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.cache_persist_path: str = os.getenv("CACHE_PERSIST_PATH", "./cache_store.db")

        # --- Mutable thresholds (guarded by lock) ---
        self._similarity_threshold: float = float(
            os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.72")
        )
        self._bm25_min_overlap: float = float(
            os.getenv("CACHE_BM25_MIN_OVERLAP", "0.3")
        )
        self._crossencoder_min_score: float = float(
            os.getenv("CACHE_CROSSENCODER_MIN_SCORE", "0.25")
        )

        self.estimated_cost_per_call: float = 0.03

    @property
    def similarity_threshold(self) -> float:
        with self._lock:
            return self._similarity_threshold

    @similarity_threshold.setter
    def similarity_threshold(self, value: float) -> None:
        with self._lock:
            self._similarity_threshold = value

    @property
    def bm25_min_overlap(self) -> float:
        with self._lock:
            return self._bm25_min_overlap

    @bm25_min_overlap.setter
    def bm25_min_overlap(self, value: float) -> None:
        with self._lock:
            self._bm25_min_overlap = value

    @property
    def crossencoder_min_score(self) -> float:
        with self._lock:
            return self._crossencoder_min_score

    @crossencoder_min_score.setter
    def crossencoder_min_score(self, value: float) -> None:
        with self._lock:
            self._crossencoder_min_score = value

settings = _Settings()