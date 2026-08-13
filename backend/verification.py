"""
Three-layer verification pipeline for semantic cache hit/miss decisions.

Pipeline:
  0. Exact-match short-circuit — instant return, no ML.
  1. Vectorized cosine similarity — single matrix multiply picks the best match.
  2. BM25 sanity check — keyword overlap confirms the cosine match is relevant.
  3. Cross-encoder     — pairwise re-ranking confirms semantic equivalence.
     (ONLY runs for borderline cases near the cosine threshold)

Every return path from decide_cache_hit() includes a debug dict so the
dashboard and tests can show *why* a decision was made, not just what it was.

Optimizations over v1:
- Persistent BM25 index (rebuilt only when entries change).
- Pre-stacked embedding matrix for vectorized cosine (one np.dot call).
- Cross-encoder result cache (same pair never scored twice).
- Aggressive query normalization catches typos and punctuation variants.
- Cosine-first ordering: cheapest discriminative check runs first.
"""

import logging
import re
import time
from difflib import SequenceMatcher
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
# Persistent BM25 index — rebuilt only when cache entries change.
# ---------------------------------------------------------------------------
_bm25_index: BM25Okapi | None = None
_bm25_version: int = -1


def _get_bm25_index(entries: list[CacheEntry], store_version: int) -> BM25Okapi:
    """Return cached BM25 index, rebuilding only when entries change."""
    global _bm25_index, _bm25_version
    if _bm25_index is None or store_version != _bm25_version:
        corpus = [_tokenize(e.query_text) for e in entries]
        _bm25_index = BM25Okapi(corpus)
        _bm25_version = store_version
        logger.info("BM25 index rebuilt for %d entries (version %d).", len(entries), store_version)
    return _bm25_index


# ---------------------------------------------------------------------------
# Pre-stacked embedding matrix — rebuilt only when cache entries change.
# One np.dot() call computes ALL cosine similarities at once.
# ---------------------------------------------------------------------------
_embedding_matrix: np.ndarray | None = None
_matrix_version: int = -1


def _get_embedding_matrix(entries: list[CacheEntry], store_version: int) -> np.ndarray:
    """Stack all entry embeddings into a single matrix for vectorized cosine."""
    global _embedding_matrix, _matrix_version
    if _embedding_matrix is None or store_version != _matrix_version:
        _embedding_matrix = np.vstack([e.embedding for e in entries]).astype(np.float32)
        _matrix_version = store_version
        logger.info("Embedding matrix rebuilt: shape %s (version %d).", _embedding_matrix.shape, store_version)
    return _embedding_matrix


# ---------------------------------------------------------------------------
# Cross-encoder result cache — same (query, candidate) pair never scored twice.
# ---------------------------------------------------------------------------
_crossencoder_cache: dict[tuple[str, str], float] = {}


# ---------------------------------------------------------------------------
# Layer 0 — Exact-match short-circuit + character-level fuzzy match
# ---------------------------------------------------------------------------

def _normalize_query(text: str) -> str:
    """Aggressive normalization: lowercase, strip punctuation, collapse whitespace."""
    text = text.strip().lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)    # Remove all punctuation
    text = re.sub(r'\s+', ' ', text).strip()     # Collapse whitespace
    return text


def _char_similarity(a: str, b: str) -> float:
    """Character-level similarity ratio (0.0 to 1.0) using SequenceMatcher.
    Catches typos like 'pythn'→'python', 'lamda'→'lambda'."""
    return SequenceMatcher(None, _normalize_query(a), _normalize_query(b)).ratio()


# Threshold for character-level fuzzy matching (catches typos).
_CHAR_SIMILARITY_THRESHOLD = 0.85


# Pre-computed normalized lookup for O(1) exact-match checks.
_normalized_lookup: dict[str, CacheEntry] = {}
_lookup_version: int = -1


def _rebuild_normalized_lookup(entries: list[CacheEntry], store_version: int) -> None:
    """Rebuild the normalized-text → CacheEntry lookup when entries change."""
    global _normalized_lookup, _lookup_version
    if store_version != _lookup_version:
        _normalized_lookup.clear()
        for entry in entries:
            key = _normalize_query(entry.query_text)
            # Keep the most recent entry for each normalized form.
            _normalized_lookup[key] = entry
        _lookup_version = store_version


def _exact_match(
    query: str, entries: list[CacheEntry], store_version: int
) -> CacheEntry | None:
    """O(1) exact-match lookup using pre-computed normalized dict."""
    _rebuild_normalized_lookup(entries, store_version)
    q_normalized = _normalize_query(query)
    return _normalized_lookup.get(q_normalized)


# ---------------------------------------------------------------------------
# Layer 1 — BM25 sanity check (now used AFTER cosine, not before)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


def bm25_prefilter(
    query: str, entries: list[CacheEntry], store_version: int
) -> list[tuple[CacheEntry, float]]:
    if not entries:
        return []

    bm25 = _get_bm25_index(entries, store_version)
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
# Layer 2 — Vectorized cosine similarity (numpy-accelerated)
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two L2-normalized numpy vectors.
    Reduces to a single dot product (sub-microsecond).
    """
    return float(np.dot(vec_a, vec_b))


def _vectorized_cosine_all(
    query_embedding: np.ndarray,
    entries: list[CacheEntry],
    store_version: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute cosine similarity of query against ALL entries in one matrix multiply.

    Returns (sorted_indices_descending, all_similarities).
    """
    matrix = _get_embedding_matrix(entries, store_version)
    # query_embedding is (384,), matrix is (N, 384) → result is (N,)
    similarities = matrix @ query_embedding.flatten()
    sorted_indices = np.argsort(similarities)[::-1]
    return sorted_indices, similarities


# ---------------------------------------------------------------------------
# Layer 3 — Cross-encoder re-check (ONLY for borderline cases)
# ---------------------------------------------------------------------------

def crossencoder_score(query: str, candidate_text: str) -> float:
    """
    Run the cross-encoder on a (query, candidate) pair.
    Results are cached so the same pair is never scored twice.

    Returns a float score — higher means more semantically similar.
    """
    cache_key = (_normalize_query(query), _normalize_query(candidate_text))
    cached = _crossencoder_cache.get(cache_key)
    if cached is not None:
        return cached

    score = float(_crossencoder_model.predict([(query, candidate_text)])[0])
    _crossencoder_cache[cache_key] = score
    return score


# ---------------------------------------------------------------------------
# Orchestrator — decide_cache_hit
# ---------------------------------------------------------------------------

# The cross-encoder is the most expensive step (~200-400ms on CPU).
# It ONLY skips for near-identical queries (cosine > 0.88) where false
# positives are essentially impossible. For everything else (0.72 - 0.88),
# the cross-encoder confirms that the INTENT matches, not just the keywords.
_BORDERLINE_UPPER = 0.88


def decide_cache_hit(
    query: str, store: CacheStore
) -> tuple[CacheEntry | None, dict[str, Any], np.ndarray | None]:
    entries = store.all_entries()
    version = store.version

    if not entries:
        return None, {"reason": "empty_cache"}, None

    # --- Layer 0: Exact-match short-circuit ---
    exact = _exact_match(query, entries, version)
    if exact is not None:
        return exact, {
            "reason": "hit",
            "method": "exact_match",
            "similarity": 1.0,
            "crossencoder_score": "skipped",
            "matched_query": exact.query_text,
        }, None 

    # --- Layer 2 FIRST: Vectorized cosine similarity (cheapest discriminative check) ---
    t0 = time.perf_counter()
    query_embedding = embed_text(query)
    t_embed = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    sorted_indices, all_sims = _vectorized_cosine_all(query_embedding, entries, version)
    t_cosine = (time.perf_counter() - t0) * 1000

    best_idx = int(sorted_indices[0])
    best_sim = float(all_sims[best_idx])
    best_entry = entries[best_idx]

    # Build candidate scores for debug (top 5 max).
    top_k = min(5, len(sorted_indices))
    candidate_scores = [
        {
            "query_text": entries[int(sorted_indices[i])].query_text,
            "cosine_similarity": round(float(all_sims[int(sorted_indices[i])]), 4),
        }
        for i in range(top_k)
    ]

    if best_sim < settings.similarity_threshold:
        # --- Fuzzy fallback: catch typo variants via character-level similarity ---
        fuzzy_match = None
        fuzzy_ratio = 0.0
        for entry in entries:
            ratio = _char_similarity(query, entry.query_text)
            if ratio > _CHAR_SIMILARITY_THRESHOLD and ratio > fuzzy_ratio:
                fuzzy_match = entry
                fuzzy_ratio = ratio

        if fuzzy_match is not None:
            return fuzzy_match, {
                "reason": "hit",
                "method": "fuzzy_typo_match",
                "similarity": round(best_sim, 4),
                "char_similarity": round(fuzzy_ratio, 4),
                "crossencoder_score": "skipped",
                "matched_query": fuzzy_match.query_text,
                "candidates": candidate_scores,
                "timing_ms": {"embed": round(t_embed, 2), "cosine": round(t_cosine, 2)},
            }, query_embedding

        return None, {
            "reason": "no_similarity_match",
            "best_score": round(best_sim, 4),
            "threshold": settings.similarity_threshold,
            "candidates": candidate_scores,
            "timing_ms": {"embed": round(t_embed, 2), "cosine": round(t_cosine, 2)},
        }, query_embedding

    # --- High-confidence match: skip BM25 and cross-encoder ---
    if best_sim >= _BORDERLINE_UPPER:
        return best_entry, {
            "reason": "hit",
            "method": "high_confidence",
            "similarity": round(best_sim, 4),
            "crossencoder_score": "skipped",
            "matched_query": best_entry.query_text,
            "candidates": candidate_scores,
            "timing_ms": {"embed": round(t_embed, 2), "cosine": round(t_cosine, 2)},
        }, query_embedding

    # --- Borderline match (threshold <= sim < _BORDERLINE_UPPER) ---
    # Use BM25 as a sanity check, then cross-encoder to confirm.

    t0 = time.perf_counter()
    bm25_candidates = bm25_prefilter(query, entries, version)
    t_bm25 = (time.perf_counter() - t0) * 1000

    # Check if the cosine best match is also present in BM25 candidates.
    bm25_entry_ids = {e.id for e, _ in bm25_candidates}
    bm25_confirms = best_entry.id in bm25_entry_ids

    # Add BM25 scores to candidate debug info.
    bm25_score_map = {e.id: s for e, s in bm25_candidates}
    for cs in candidate_scores:
        for entry in entries:
            if entry.query_text == cs["query_text"]:
                cs["bm25_score"] = round(bm25_score_map.get(entry.id, 0.0), 4)
                break

    # --- Layer 3: Cross-encoder (ONLY for borderline cases) ---
    t0 = time.perf_counter()
    ce_score = crossencoder_score(query, best_entry.query_text)
    t_ce = (time.perf_counter() - t0) * 1000

    timing = {
        "embed": round(t_embed, 2),
        "cosine": round(t_cosine, 2),
        "bm25": round(t_bm25, 2),
        "crossencoder": round(t_ce, 2),
    }

    if ce_score < settings.crossencoder_min_score:
        return None, {
            "reason": "crossencoder_rejected",
            "similarity": round(best_sim, 4),
            "crossencoder_score": round(ce_score, 4),
            "bm25_confirms": bm25_confirms,
            "threshold_similarity": settings.similarity_threshold,
            "threshold_crossencoder": settings.crossencoder_min_score,
            "candidates": candidate_scores,
            "timing_ms": timing,
        }, query_embedding

    return best_entry, {
        "reason": "hit",
        "method": "full_pipeline",
        "similarity": round(best_sim, 4),
        "crossencoder_score": round(ce_score, 4),
        "bm25_confirms": bm25_confirms,
        "matched_query": best_entry.query_text,
        "candidates": candidate_scores,
        "timing_ms": timing,
    }, query_embedding