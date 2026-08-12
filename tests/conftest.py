"""
Shared pytest fixtures for the semantic cache test suite.
"""

import os
import tempfile

import pytest

from backend.cache_store import CacheStore


@pytest.fixture()
def tmp_db_path(tmp_path):
    """Return a temporary SQLite database path that is cleaned up after the test."""
    return str(tmp_path / "test_cache.db")


@pytest.fixture()
def cache_store(tmp_db_path):
    """Provide a fresh, empty CacheStore backed by a temporary SQLite file."""
    store = CacheStore(tmp_db_path)
    yield store
    store.close()
