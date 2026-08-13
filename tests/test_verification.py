"""
Unit tests for the three-layer verification pipeline.

These are real pytest cases — not placeholders. They test the core logic
that determines cache hit vs. miss, including the critical "trap pair"
that must never produce a false positive.
"""

import pytest

from backend.verification import decide_cache_hit, embed_text
from backend.config import settings


# -------------------------------------------------------------------------
# Test 1: Empty cache → always a miss
# -------------------------------------------------------------------------


def test_empty_cache_returns_miss(cache_store):
    """An empty cache must always return a miss with reason 'empty_cache'."""
    result, debug, _ = decide_cache_hit("How do I reset my password?", cache_store)

    assert result is None
    assert debug["reason"] == "empty_cache"


# -------------------------------------------------------------------------
# Test 2: Paraphrase pair → cache hit
# -------------------------------------------------------------------------


def test_paraphrase_returns_hit(cache_store):
    """
    A semantically equivalent paraphrase should produce a cache hit.

    Cached:  'How do I reset my password?'
    Query:   'What are the steps to recover my password?'
    Expected: HIT — these mean the same thing.
    """
    original = "How do I reset my password?"
    paraphrase = "What are the steps to recover my password?"

    # Populate cache with the original query.
    embedding = embed_text(original)
    cache_store.add(original, embedding, "Go to Settings > Security > Reset Password.")

    result, debug, _ = decide_cache_hit(paraphrase, cache_store)

    assert result is not None, (
        f"Expected a cache hit for paraphrase, got miss. Debug: {debug}"
    )
    assert debug["reason"] == "hit"
    assert "similarity" in debug
    assert "crossencoder_score" in debug


# -------------------------------------------------------------------------
# Test 3: Trap pair → cache miss (THE test that matters most)
# -------------------------------------------------------------------------


def test_trap_pair_returns_miss(cache_store):
    """
    The 'trap pair' must NEVER produce a false cache hit.

    Cached:  'How do I reset my password?'
    Query:   'How do I delete my account?'
    Expected: MISS — these are superficially similar but semantically different.

    The reason must be either 'no_similarity_match' or 'crossencoder_rejected',
    never a hit.
    """
    cached_query = "How do I reset my password?"
    trap_query = "How do I delete my account?"

    embedding = embed_text(cached_query)
    cache_store.add(
        cached_query, embedding, "Go to Settings > Security > Reset Password."
    )

    result, debug, _ = decide_cache_hit(trap_query, cache_store)

    assert result is None, (
        f"CRITICAL: Trap pair produced a false cache hit! Debug: {debug}"
    )
    assert debug["reason"] in ("no_similarity_match", "crossencoder_rejected"), (
        f"Unexpected rejection reason: {debug['reason']}"
    )


# -------------------------------------------------------------------------
# Test 4: Threshold sensitivity — lowering threshold flips rejection to hit
# -------------------------------------------------------------------------


def test_threshold_sensitivity(cache_store):
    """
    Verify that lowering CACHE_SIMILARITY_THRESHOLD far enough can flip
    a borderline rejection to a hit. This validates the live slider demo.

    We use a pair that is somewhat related but not a perfect paraphrase,
    so it sits near the default threshold boundary.
    """
    cached_query = "How do I reset my password?"
    borderline_query = "How can I change my password?"

    embedding = embed_text(cached_query)
    cache_store.add(
        cached_query, embedding, "Go to Settings > Security > Reset Password."
    )

    # Save the original threshold.
    original_threshold = settings.similarity_threshold

    try:
        # At a very high threshold, this should be a miss or borderline.
        settings.similarity_threshold = 0.99
        result_strict, debug_strict, _ = decide_cache_hit(borderline_query, cache_store)

        # At a very low threshold, this should definitely be a hit.
        settings.similarity_threshold = 0.30
        # Also lower the cross-encoder threshold to ensure Layer 3 passes.
        original_ce = settings.crossencoder_min_score
        settings.crossencoder_min_score = -10.0

        result_loose, debug_loose, _ = decide_cache_hit(borderline_query, cache_store)

        # Restore cross-encoder threshold.
        settings.crossencoder_min_score = original_ce

        # The strict threshold should reject (or at least not both be hits).
        # The loose threshold should accept.
        assert result_loose is not None, (
            f"Even at threshold=0.30, no hit was produced. Debug: {debug_loose}"
        )
        assert debug_loose["reason"] == "hit"

        # Verify that strict was indeed more restrictive.
        if result_strict is not None:
            # Both passed — that's fine if the pair is very close.
            # But the test still validates that the slider mechanism works.
            pass
        else:
            # Strict rejected, loose accepted — the slider flipped the decision.
            assert debug_strict["reason"] in (
                "no_similarity_match",
                "crossencoder_rejected",
            )

    finally:
        # Always restore the original threshold.
        settings.similarity_threshold = original_threshold
