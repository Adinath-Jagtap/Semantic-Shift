# Semantic-Shift — Build Spec

- Semantic Cache Middleware with Hybrid Verification — Build Spec

## 0. What this project IS and IS NOT

- **IS:** A caching proxy that sits in front of an LLM API. It stores past (query, answer) pairs and
  decides, using three verification layers, whether a new query can safely reuse a past answer.
- **IS NOT:** A RAG / document-retrieval system. There is no document corpus, no chunking, no
  "chat with your files." Do not add a document ingestion pipeline. Do not add FAISS/Qdrant/
  Elasticsearch/LangChain — they are unnecessary at this scale and out of scope.
- Storage is intentionally simple: an in-process Python list of cache entries, persisted to a
  local SQLite file. No external database, no Redis.

---

## 1. Tech Stack

- Python 3.11+, FastAPI, uvicorn
- `sentence-transformers` with model `all-MiniLM-L6-v2` — for embeddings
- `rank_bm25` (`BM25Okapi`) — for keyword overlap
- `sentence-transformers` `CrossEncoder` with model `cross-encoder/ms-marco-MiniLM-L-6-v2` — for
  final re-verification
- `groq` Python SDK as the underlying LLM — model `llama-3.1-8b-instant` by default (configurable via env var)
- Vanilla HTML/CSS/JS — for the dashboard UI (served from FastAPI, no build step)
- `python-dotenv` for loading `GROQ_API_KEY` from a `.env` file

---

## 2. Environment Variables

`.env.example`:

```
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant
CACHE_SIMILARITY_THRESHOLD=0.85
CACHE_BM25_MIN_OVERLAP=0.3
CACHE_CROSSENCODER_MIN_SCORE=0.5
CACHE_PERSIST_PATH=./cache_store.db
```

Never hardcode the API key. Fail fast with a clear error message on startup if `GROQ_API_KEY` is
missing, rather than failing silently on the first request.

---

## 3. Project Structure

```
semantic-cache/
  .env.example
  requirements.txt
  README.md
  CODE_FLOW.md
  TECHNICAL_DECISIONS.md
  backend/
    __init__.py
    config.py           # loads env vars, validates on startup
    cache_store.py      # CacheStore class: add(), all_entries(), close()
    verification.py     # embed_text(), bm25_prefilter(), crossencoder_score(), decide_cache_hit()
    llm_client.py       # call_groq(query) -> answer text
    main.py             # FastAPI app + serves frontend/index.html at GET /
  frontend/
    index.html          # Single-page dashboard UI (HTML/CSS/JS, no build step)
  tests/
    conftest.py         # Shared test fixtures
    test_verification.py # Unit tests including the "reset password" vs "delete account" trap case
```

---

## 4. `backend/cache_store.py` — the storage layer

Use **SQLite** via the `sqlite3` standard-library module — not JSON. At this scale (hundreds of
entries) raw speed is identical either way, but SQLite writes one row at a time instead of rewriting
a whole file, so a crash mid-demo can't corrupt the entire cache. No server, no extra dependency.

- Schema (created on startup if the file at `CACHE_PERSIST_PATH` doesn't exist yet):
  ```sql
  CREATE TABLE IF NOT EXISTS cache_entries (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      query_text TEXT NOT NULL,
      embedding TEXT NOT NULL,   -- JSON-encoded list[float], e.g. json.dumps(embedding.tolist())
      answer TEXT NOT NULL,
      created_at REAL NOT NULL
  );
  ```
- `CacheEntry` is a plain Python dataclass: `{id, query_text, embedding: list[float], answer, created_at}`.
  The `embedding` column stores it as a JSON string (`json.dumps`/`json.loads`).
- `CacheStore.__init__` opens the SQLite connection, runs `CREATE TABLE IF NOT EXISTS`, then
  loads all existing rows into an in-memory `self.entries: list[CacheEntry]` list on startup.
- `add(query_text, embedding, answer)` inserts one row via a parameterized `INSERT`, commits
  immediately, and appends the new entry to `self.entries`.
- `all_entries()` returns `self.entries` for the verification layer to scan.
- Use `sqlite3.connect(path, check_same_thread=False)`.

---

## 5. `backend/verification.py` — the three-layer check

Implement `decide_cache_hit(query: str, store: CacheStore) -> tuple[CacheEntry | None, dict]` that:

1. If `store.entries` is empty, return `(None, {"reason": "empty_cache"})` immediately.
2. **Layer 1 — BM25 pre-filter**: score the query against all cached query texts. Keep entries
   above `CACHE_BM25_MIN_OVERLAP`. If no entry passes, still allow the top-3 BM25 candidates
   through to layer 2 (BM25 alone is noisy for short queries — don't let it hard-reject).
3. Compute the query's embedding once (after BM25, so it's skipped on empty cache).
4. **Layer 2 — cosine similarity**: among the BM25-shortlisted candidates, compute cosine similarity
   between the query embedding and each candidate's stored embedding. Keep the single best match if
   its score is >= `CACHE_SIMILARITY_THRESHOLD`. If none qualifies, return
   `(None, {"reason": "no_similarity_match", "best_score": <float>})`.
5. **Layer 3 — cross-encoder re-check**: run the cross-encoder on `(query, best_candidate.query_text)`.
   If the score is >= `CACHE_CROSSENCODER_MIN_SCORE`, this is a real cache hit — return
   `(best_candidate, {"reason": "hit", "similarity": ..., "crossencoder_score": ...})`.
   Otherwise return `(None, {"reason": "crossencoder_rejected", "similarity": ..., "crossencoder_score": ...})`.
6. Every return path must include the debug dict — the dashboard and tests depend on it to show
   _why_ a decision was made, not just what the decision was.

This is the function that must correctly reject the "reset password" vs "delete account" trap case.

---

## 6. `backend/llm_client.py`

- `call_groq(query: str) -> str`: sends `{"role": "user", "content": query}` to the Groq chat
  completions endpoint using `GROQ_MODEL`, returns the response text.
- Wrap the network call in try/except. On failure, raise a `GroqAPIError` with the error body
  attached. Catch it in `main.py` and return a proper 502 with a JSON error body.
- Support non-streaming responses only for v1.

---

## 7. `backend/main.py` — the API

- `GET /` — serves `frontend/index.html` as an `HTMLResponse`. This is the dashboard entry point.
- `POST /v1/chat/completions` — accepts `{"messages": [{"role": "user", "content": "..."}]}`
  (OpenAI-compatible shape, but only the last user message is used for cache lookup/storage in v1).
  Flow: extract query text → `decide_cache_hit()` → on hit, return the cached answer with
  `{"cached": true, "latency_ms": ..., "debug": {...}}` → on miss, call Groq, `store.add(...)`,
  return `{"cached": false, "latency_ms": ..., "debug": {...}}`.
- Track running totals in memory: total requests, cache hits, cache misses, estimated dollars saved,
  estimated time saved.
- `GET /stats` — returns running totals + `recent_queries` (last 20), for the dashboard to poll.
- `GET /health` — returns `{"status": "ok"}`.
- `POST /config` — updates verification thresholds at runtime (no server restart needed).
  Returns the current config values regardless of what was sent, so the dashboard can initialize
  its sliders from the live backend state.
- CORS: allow `*` (dashboard is served from the same origin; open CORS lets Postman/curl work too).

---

## 8. `frontend/index.html` — the dashboard UI

A self-contained single-page app served from FastAPI at `GET /`. No build step, no npm.

Features:

- **Query Tester** (default view): chat-style interface with user/assistant bubbles.
  Each assistant reply shows a `CACHE HIT` or `CACHE MISS` chip, latency, markdown-rendered
  answer, and a collapsible "Details" panel showing the full debug dict.
- **Dashboard** view: 6 KPI cards (Requests, Hits, Misses, Hit Rate, $ Saved, Time Saved) +
  a scrollable query log table of the last 20 entries.
- **Settings panel** (top-right gear icon): 3 range sliders for the verification thresholds.
  "Save" sends a `POST /config` to the backend — changes take effect immediately without restart.
- **Backend status indicator** (top-right dot): green (online) / red (offline) with pulsing
  animation. Hover shows "Backend is connected" / "Backend is not connected". Polls every 15s.
- **Sidebar**: app name + nav (Query Tester / Dashboard) + Last Result card (HIT/MISS + latency
  of the most recent query) + Repo Link.
- Markdown rendering via `marked.js` (CDN). XSS prevention via `DOMPurify` (CDN).
- Mobile-responsive with a hamburger menu drawer on small screens.

`BACKEND_URL` in the JS is set to `window.location.origin` — since the HTML is served from the
same FastAPI process, no hardcoded URLs are needed.

---

## 9. Known Gaps — do not silently pretend these don't exist

Listed at the top of `backend/main.py`:

- No streaming response support yet (only whole-response replies).
- Only the single most recent user message is used for cache matching — multi-turn conversation
  context is not embedded.
- No cache eviction/TTL — entries persist indefinitely. Fine for a hackathon demo, not production.
- Single-process only — cache and stats are not shared across multiple server instances.

---

## 10. Tests (`tests/test_verification.py`)

At minimum, these four pytest cases:

1. **Empty cache** → always a miss (`reason: empty_cache`).
2. **Paraphrase pair** ("How do I reset my password?" cached, query "What are the steps to recover
   my password?") → must return a HIT.
3. **The trap pair** ("How do I reset my password?" cached, query "How do I delete my account?")
   → must return a MISS with `reason` of `no_similarity_match` or `crossencoder_rejected`.
   This test is the one that matters most.
4. **Threshold sensitivity**: same pair, verify that lowering `CACHE_SIMILARITY_THRESHOLD` far
   enough can flip a rejected match to a hit (used for the live slider demo).

---

## 11. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env and add GROQ_API_KEY

# Start (serves API + dashboard on the same port)
uvicorn backend.main:app --reload

# Open dashboard
# Navigate to http://localhost:8000

# Run tests
pytest tests/ -v
```

---

## 12. Definition of Done

- `uvicorn backend.main:app --reload` starts cleanly with a clear error if `GROQ_API_KEY` is
  unset, and clear success log otherwise.
- `http://localhost:8000` loads the dashboard UI with backend status showing green.
- The Settings sliders actually change live verification behavior (verified manually against the
  trap-pair query).
- All four tests in section 10 pass (`pytest tests/ -v`).
- README documents: setup steps, environment variables, how to run, the trap-pair demo, and the
  four known limitations.
