"""
Three-layer verification pipeline for semantic cache hit/miss decisions.

Pipeline:
  1. BM25 pre-filter  — keyword overlap narrows candidates.
  2. Cosine similarity — embedding distance picks the best match.
  3. Cross-encoder     — pairwise re-ranking confirms semantic equivalence.

Every return path from decide_cache_hit() includes a debug dict so the
dashboard and tests can show *why* a decision was made, not just what it was.
"""

import math
from typing import Any

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from backend.cache_store import CacheEntry, CacheStore
from backend.config import settings

_embedding_model: SentenceTransformer | None = None
_crossencoder_model: CrossEncoder | None = None


def _get_embedding_model() -> SentenceTransformer:
    """Lazy-load the bi-encoder embedding model (all-MiniLM-L6-v2)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _get_crossencoder() -> CrossEncoder:
    """Lazy-load the cross-encoder re-ranker (ms-marco-MiniLM-L-6-v2)."""
    global _crossencoder_model
    if _crossencoder_model is None:
        _crossencoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _crossencoder_model

def embed_text(text: str) -> list[float]:
    """
    Compute the dense embedding for a single string.

    Returns a plain Python list[float] (not a numpy array) so it can be
    JSON-serialized directly into SQLite.
    """
    model = _get_embedding_model()
    vector = model.encode(text, convert_to_numpy=True)
    return vector.tolist()


# ---------------------------------------------------------------------------
# Layer 1 — BM25 pre-filter
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


def bm25_prefilter(
    query: str, entries: list[CacheEntry]
) -> list[tuple[CacheEntry, float]]:
    """
    Score the query against all cached query texts using BM25Okapi.

    Returns entries that pass the BM25_MIN_OVERLAP threshold. If no entry
    passes, the top-3 candidates are returned anyway — BM25 is noisy for
    short queries and should not hard-reject on its own (spec §5.3).
    """
    if not entries:
        return []

    corpus = [_tokenize(e.query_text) for e in entries]
    bm25 = BM25Okapi(corpus)
    query_tokens = _tokenize(query)
    raw_scores = bm25.get_scores(query_tokens)

    # Normalize scores to [0, 1] for threshold comparison.
    max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
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
# Layer 2 — Cosine similarity
# ---------------------------------------------------------------------------
def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Implemented manually to avoid importing scipy/numpy for a single function.
    Returns a float in [-1, 1].
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)


def _find_best_cosine_match(
    query_embedding: list[float],
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
# Layer 3 — Cross-encoder re-check
# ---------------------------------------------------------------------------
def crossencoder_score(query: str, candidate_text: str) -> float:
    """
    Run the cross-encoder on a (query, candidate) pair.

    Returns a float score — higher means more semantically similar.
    The cross-encoder sees the raw text pair (not embeddings) and is
    more accurate than bi-encoder cosine similarity at distinguishing
    paraphrases from superficially similar but semantically different queries.
    """
    model = _get_crossencoder()
    score = model.predict([(query, candidate_text)])
    return float(score[0])


# ---------------------------------------------------------------------------
# Orchestrator — decide_cache_hit
# ---------------------------------------------------------------------------
def decide_cache_hit(
    query: str, store: CacheStore
) -> tuple[CacheEntry | None, dict[str, Any]]:
    """
    Run the full 3-layer verification pipeline to decide if a query
    can safely reuse a cached answer.

    Returns:
        (matching_entry, debug_dict) where matching_entry is None on a miss.
        The debug dict always explains *why* the decision was made.
    """
    entries = store.all_entries()

    # --- Early exit: empty cache ---
    if not entries:
        return None, {"reason": "empty_cache"}

    # --- Layer 1: BM25 pre-filter ---
    bm25_candidates = bm25_prefilter(query, entries)

    if not bm25_candidates:
        # Shouldn't happen (bm25_prefilter returns top-3 fallback), but guard.
        return None, {"reason": "no_bm25_candidates"}

    # --- Layer 2: Cosine similarity ---
    query_embedding = embed_text(query)
    best_entry, best_sim, candidate_scores = _find_best_cosine_match(
        query_embedding, bm25_candidates
    )

    if best_entry is None or best_sim < settings.similarity_threshold:
        return None, {
            "reason": "no_similarity_match",
            "best_score": round(best_sim, 4),
            "threshold": settings.similarity_threshold,
            "candidates": candidate_scores,
        }

    # --- Layer 3: Cross-encoder re-check ---
    ce_score = crossencoder_score(query, best_entry.query_text)

    if ce_score < settings.crossencoder_min_score:
        return None, {
            "reason": "crossencoder_rejected",
            "similarity": round(best_sim, 4),
            "crossencoder_score": round(ce_score, 4),
            "threshold_similarity": settings.similarity_threshold,
            "threshold_crossencoder": settings.crossencoder_min_score,
            "candidates": candidate_scores,
        }

    # --- Cache hit ---
    return best_entry, {
        "reason": "hit",
        "similarity": round(best_sim, 4),
        "crossencoder_score": round(ce_score, 4),
        "matched_query": best_entry.query_text,
        "candidates": candidate_scores,
    }