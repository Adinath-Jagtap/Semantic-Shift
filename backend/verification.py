"""
Three-layer verification pipeline for semantic cache hit/miss decisions.

Pipeline:
  0. Exact-match short-circuit — instant return, no ML.
  1. BM25 pre-filter  — keyword overlap narrows candidates.
  2. Cosine similarity — embedding distance picks the best match.
  3. Cross-encoder     — pairwise re-ranking confirms semantic equivalence.
     (ONLY runs for borderline cases near the cosine threshold)

Every return path from decide_cache_hit() includes a debug dict so the
dashboard and tests can show *why* a decision was made, not just what it was.
"""

import logging
import time
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from backend.cache_store import CacheEntry, CacheStore
from backend.config import settings

logger = logging.getLogger("semantic_cache")
logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("Embedding model loaded.")

logger.info("Loading cross-encoder (cross-encoder/quora-distilroberta-base)...")
_crossencoder_model = CrossEncoder("cross-encoder/quora-distilroberta-base")
logger.info("Cross-encoder loaded.")

# ---------------------------------------------------------------------------
# Embedding cache — avoids redundant model.encode() calls for the same text.
# Maps query_text → numpy embedding. Populated on every embed_text() call
# and on cache load from SQLite so repeated queries cost 0ms.
# ---------------------------------------------------------------------------
_embedding_cache: dict[str, np.ndarray] = {}

def embed_text(text: str) -> np.ndarray:
    cached = _embedding_cache.get(text)
    if cached is not None:
        return cached

    vector = _embedding_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    _embedding_cache[text] = vector
    return vector


def warm_embedding_cache(entries: list[CacheEntry]) -> None:
    for entry in entries:
        _embedding_cache[entry.query_text] = entry.embedding


# ---------------------------------------------------------------------------
# Layer 0 — Exact-match short-circuit
# ---------------------------------------------------------------------------

def _normalize_query(text: str) -> str:
    return text.strip().lower().rstrip("?.!,").strip()


def _exact_match(
    query: str, entries: list[CacheEntry]
) -> CacheEntry | None:
    q_normalized = _normalize_query(query)
    for entry in entries:
        if _normalize_query(entry.query_text) == q_normalized:
            return entry
    return None


# ---------------------------------------------------------------------------
# Layer 1 — BM25 pre-filter
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


def bm25_prefilter(
    query: str, entries: list[CacheEntry]
) -> list[tuple[CacheEntry, float]]:
    if not entries:
        return []

    corpus = [_tokenize(e.query_text) for e in entries]
    bm25 = BM25Okapi(corpus)
    query_tokens = _tokenize(query)
    raw_scores = bm25.get_scores(query_tokens)

    # Normalize scores to [0, 1] for threshold comparison.
    max_score = float(max(raw_scores)) if max(raw_scores) > 0 else 1.0
    scored = [
        (entries[i], float(raw_scores[i] / max_score))
        for i in range(len(entries))
    ]

    # Keep entries above the BM25 threshold.
    above_threshold = [(e, s) for e, s in scored if s >= settings.bm25_min_overlap]

    if above_threshold:
        return above_threshold

    # Fallback: top-3 by BM25 score (spec §5.3 — don't let BM25 hard-reject).
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:3]


# ---------------------------------------------------------------------------
# Layer 2 — Cosine similarity (numpy-accelerated)
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two L2-normalized numpy vectors.
    Reduces to a single dot product (sub-microsecond).
    """
    return float(np.dot(vec_a, vec_b))


def _find_best_cosine_match(
    query_embedding: np.ndarray,
    candidates: list[tuple[CacheEntry, float]],
) -> tuple[CacheEntry | None, float, list[dict[str, Any]]]:
    """
    Among BM25-shortlisted candidates, find the one with the highest
    cosine similarity to the query embedding.

    Returns (best_entry_or_None, best_score, all_candidate_scores).
    """
    all_scores: list[dict[str, Any]] = []
    best_entry: CacheEntry | None = None
    best_score: float = -1.0

    for entry, bm25_score in candidates:
        sim = cosine_similarity(query_embedding, entry.embedding)
        all_scores.append(
            {
                "query_text": entry.query_text,
                "bm25_score": round(bm25_score, 4),
                "cosine_similarity": round(sim, 4),
            }
        )
        if sim > best_score:
            best_score = sim
            best_entry = entry

    return best_entry, best_score, all_scores


# ---------------------------------------------------------------------------
# Layer 3 — Cross-encoder re-check (ONLY for borderline cases)
# ---------------------------------------------------------------------------

def crossencoder_score(query: str, candidate_text: str) -> float:
    """
    Run the cross-encoder on a (query, candidate) pair.

    Returns a float score — higher means more semantically similar.
    The STS-B model outputs on a [0, 5] scale.
    """
    score = _crossencoder_model.predict([(query, candidate_text)])
    return float(score[0])


# ---------------------------------------------------------------------------
# Orchestrator — decide_cache_hit
# ---------------------------------------------------------------------------

# The cross-encoder is the most expensive step (~200-400ms on CPU).
# It ONLY skips for near-identical queries (cosine > 0.92) where false
# positives are essentially impossible. For everything else (0.76 - 0.92),
# the cross-encoder confirms that the INTENT matches, not just the keywords.
_BORDERLINE_UPPER = 0.92


def decide_cache_hit(
    query: str, store: CacheStore
) -> tuple[CacheEntry | None, dict[str, Any], np.ndarray | None]:
    entries = store.all_entries()

    if not entries:
        return None, {"reason": "empty_cache"}, None

    # --- Layer 0: Exact-match short-circuit ---
    exact = _exact_match(query, entries)
    if exact is not None:
        return exact, {
            "reason": "hit",
            "method": "exact_match",
            "similarity": 1.0,
            "crossencoder_score": "skipped",
            "matched_query": exact.query_text,
        }, None 

    # --- Layer 1: BM25 pre-filter ---
    t0 = time.perf_counter()
    bm25_candidates = bm25_prefilter(query, entries)
    t_bm25 = (time.perf_counter() - t0) * 1000

    if not bm25_candidates:
        return None, {"reason": "no_bm25_candidates"}, None

    # --- Layer 2: Cosine similarity ---
    t0 = time.perf_counter()
    query_embedding = embed_text(query)
    t_embed = (time.perf_counter() - t0) * 1000

    best_entry, best_sim, candidate_scores = _find_best_cosine_match(
        query_embedding, bm25_candidates
    )

    if best_entry is None or best_sim < settings.similarity_threshold:
        return None, {
            "reason": "no_similarity_match",
            "best_score": round(best_sim, 4),
            "threshold": settings.similarity_threshold,
            "candidates": candidate_scores,
            "timing_ms": {"bm25": round(t_bm25, 2), "embed": round(t_embed, 2)},
        }, query_embedding

    # --- Layer 3: Cross-encoder (ONLY for borderline cases) ---
    if best_sim >= _BORDERLINE_UPPER:
        return best_entry, {
            "reason": "hit",
            "method": "high_confidence",
            "similarity": round(best_sim, 4),
            "crossencoder_score": "skipped",
            "matched_query": best_entry.query_text,
            "candidates": candidate_scores,
            "timing_ms": {"bm25": round(t_bm25, 2), "embed": round(t_embed, 2)},
        }, query_embedding

    t0 = time.perf_counter()
    ce_score = crossencoder_score(query, best_entry.query_text)
    t_ce = (time.perf_counter() - t0) * 1000

    if ce_score < settings.crossencoder_min_score:
        return None, {
            "reason": "crossencoder_rejected",
            "similarity": round(best_sim, 4),
            "crossencoder_score": round(ce_score, 4),
            "threshold_similarity": settings.similarity_threshold,
            "threshold_crossencoder": settings.crossencoder_min_score,
            "candidates": candidate_scores,
            "timing_ms": {"bm25": round(t_bm25, 2), "embed": round(t_embed, 2), "crossencoder": round(t_ce, 2)},
        }, query_embedding

    return best_entry, {
        "reason": "hit",
        "method": "full_pipeline",
        "similarity": round(best_sim, 4),
        "crossencoder_score": round(ce_score, 4),
        "matched_query": best_entry.query_text,
        "candidates": candidate_scores,
        "timing_ms": {"bm25": round(t_bm25, 2), "embed": round(t_embed, 2), "crossencoder": round(t_ce, 2)},
    }, query_embedding