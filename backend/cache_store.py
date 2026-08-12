"""
Cache storage layer — SQLite-backed with an in-memory hot path.

Design rationale:
- SQLite is the durable backing store (crash-safe, one-row-at-a-time writes).
- On startup, ALL rows are loaded into self.entries (an in-memory list).
- The hot-path similarity/BM25 scoring runs against the in-memory list —
  SQLite is never queried per-request.
- add() writes to SQLite AND appends to the in-memory list atomically,
  so a newly cached answer is usable on the very next request.
"""

import json
import sqlite3
import time
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """A single cached (query, answer) pair with its precomputed embedding."""

    id: int
    query_text: str
    embedding: list[float]
    answer: str
    created_at: float


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS cache_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT NOT NULL,
    embedding TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""

_INSERT_SQL = """
INSERT INTO cache_entries (query_text, embedding, answer, created_at)
VALUES (?, ?, ?, ?);
"""

_SELECT_ALL_SQL = """
SELECT id, query_text, embedding, answer, created_at
FROM cache_entries
ORDER BY id;
"""


class CacheStore:
    """
    SQLite-backed cache with an in-memory list for fast scoring.

    Thread safety note: sqlite3.connect(check_same_thread=False) is used
    because FastAPI's async handlers may be invoked from different threads
    in the dev server. A single connection with immediate commits is
    sufficient at hackathon scale — no connection pool needed.
    """

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()

        # Load all existing rows into memory on startup.
        self.entries: list[CacheEntry] = self._load_all()

    def _load_all(self) -> list[CacheEntry]:
        """Read every row from SQLite into CacheEntry objects."""
        cursor = self._conn.execute(_SELECT_ALL_SQL)
        entries: list[CacheEntry] = []
        for row in cursor.fetchall():
            entries.append(
                CacheEntry(
                    id=row[0],
                    query_text=row[1],
                    embedding=json.loads(row[2]),
                    answer=row[3],
                    created_at=row[4],
                )
            )
        return entries

    def add(self, query_text: str, embedding: list[float], answer: str) -> CacheEntry:
        """
        Persist a new cache entry to SQLite and append it to the in-memory list.

        Uses parameterized SQL (? placeholders) — never string-formatted queries.
        Commits immediately so the row survives a crash.
        Returns the newly created CacheEntry.
        """
        created_at = time.time()
        embedding_json = json.dumps(embedding)

        cursor = self._conn.execute(
            _INSERT_SQL, (query_text, embedding_json, answer, created_at)
        )
        self._conn.commit()

        entry = CacheEntry(
            id=cursor.lastrowid,  # type: ignore[arg-type]
            query_text=query_text,
            embedding=embedding,
            answer=answer,
            created_at=created_at,
        )
        self.entries.append(entry)
        return entry

    def all_entries(self) -> list[CacheEntry]:
        """Return the in-memory entry list for the verification layer to scan."""
        return self.entries

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
