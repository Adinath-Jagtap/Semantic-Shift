# Code Flow — Function Map & Data Flow

This document maps every module, every function, its inputs/outputs, where it's called, and how data flows through the system.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HTTP Request (POST /v1/chat/completions)                 │
└─────────────────┬───────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────┐
│  backend/main.py                │
│  chat_completions()             │
│  - Extracts last user message   │
│  - Measures latency             │
│  - Updates stats                │
└────────┬───────────┬────────────┘
         │           │
    (cache check)  (cache miss)
         │           │
         ▼           ▼
┌────────────────┐  ┌─────────────────┐
│verification.py │  │ llm_client.py   │
│decide_cache    │  │ call_groq()     │
│_hit()          │  │ → Groq API      │
└──┬──┬──┬───────┘  └────────┬────────┘
   │  │  │                   │
   │  │  │                   ▼
   │  │  │          ┌─────────────────┐
   │  │  │          │ cache_store.py  │
   │  │  │          │ store.add()     │
   │  │  │          │ → SQLite + RAM  │
   │  │  │          └─────────────────┘
   │  │  │
   │  │  └── Layer 3: crossencoder_score()
   │  └───── Layer 2: _find_best_cosine_match() → cosine_similarity()
   └──────── Layer 1: bm25_prefilter()
```

---

## Module: `backend/config.py`

**Purpose:** Load environment variables, validate at import time, expose a thread-safe mutable settings singleton.

### `class _Settings`

| Property / Method | Type | Description |
|---|---|---|
| `__init__()` | constructor | Reads `.env`, validates `GROQ_API_KEY` (exits if missing), sets defaults |
| `groq_api_key` | `str` | The Groq API key (never logged, never returned in responses) |
| `groq_model` | `str` | LLM model name, default `llama-3.1-8b-instant` |
| `cache_persist_path` | `str` | SQLite file path, default `./cache_store.db` |
| `similarity_threshold` | `float` (property) | Thread-safe getter/setter, default `0.76` |
| `bm25_min_overlap` | `float` (property) | Thread-safe getter/setter, default `0.3` |
| `crossencoder_min_score` | `float` (property) | Thread-safe getter/setter, default `0.5` |
| `estimated_cost_per_call` | `float` | Fixed at `$0.03` for "dollars saved" metric |

### `settings` (module-level singleton)

- **Created at:** import time
- **Used by:** every other module imports `from backend.config import settings`
- **Thread safety:** all mutable thresholds use `threading.Lock`

### Call Graph

```
config.py (import) → load_dotenv() → _Settings.__init__()
                                        ├── os.getenv("GROQ_API_KEY") → sys.exit(1) if missing
                                        ├── os.getenv("GROQ_MODEL")
                                        ├── os.getenv("CACHE_PERSIST_PATH")
                                        ├── os.getenv("CACHE_SIMILARITY_THRESHOLD")
                                        ├── os.getenv("CACHE_BM25_MIN_OVERLAP")
                                        └── os.getenv("CACHE_CROSSENCODER_MIN_SCORE")
```

---

## Module: `backend/cache_store.py`

**Purpose:** SQLite-backed persistent cache with an in-memory list for fast hot-path scoring.

### `class CacheEntry` (dataclass)

| Field | Type | Description |
|---|---|---|
| `id` | `int` | SQLite autoincrement primary key |
| `query_text` | `str` | The original user query |
| `embedding` | `np.ndarray` (float32) | 384-dimensional L2-normalized dense vector from `all-MiniLM-L6-v2` |
| `answer` | `str` | The LLM's response text |
| `created_at` | `float` | Unix timestamp (`time.time()`) |

### `class CacheStore`

| Method | Signature | Description | Called By |
|---|---|---|---|
| `__init__` | `(db_path: str)` | Opens SQLite, creates table, loads all rows into `self.entries` | `main.py` lifespan |
| `_load_all` | `() → list[CacheEntry]` | Reads all SQLite rows, deserializes embeddings via `json.loads` | `__init__` |
| `add` | `(query_text, embedding, answer) → CacheEntry` | Parameterized INSERT + immediate commit + append to in-memory list | `main.py` on cache miss |
| `all_entries` | `() → list[CacheEntry]` | Returns `self.entries` (the in-memory list) | `verification.py` |
| `close` | `() → None` | Closes SQLite connection | `main.py` lifespan shutdown |

### Data Flow

```
Startup:
  SQLite file → _load_all() → self.entries (in-memory list)

On cache miss:
  main.py → store.add(query, embedding, answer)
          → INSERT INTO cache_entries (parameterized)
          → conn.commit()
          → self.entries.append(new_entry)

On cache check:
  verification.py → store.all_entries() → returns self.entries
```

### SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS cache_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT NOT NULL,
    embedding TEXT NOT NULL,   -- JSON-encoded list[float], loaded as np.ndarray in memory
    answer TEXT NOT NULL,
    created_at REAL NOT NULL
);
```

---

## Module: `backend/verification.py`

**Purpose:** The 3-layer verification pipeline — the core intelligence of the system.

### Model Singletons

| Function | Returns | Model | Loaded When |
|---|---|---|---|
| `_get_embedding_model()` | `SentenceTransformer` | `all-MiniLM-L6-v2` | First call to `embed_text()` |
| `_crossencoder_model` | `CrossEncoder` | `cross-encoder/quora-distilroberta-base` | Loaded eagerly at import time |

### Functions

| Function | Signature | Layer | Description | Called By |
|---|---|---|---|---|
| `embed_text` | `(text: str) → np.ndarray` | In-memory cache | Computes 384-dim L2-normalized embedding, cached per query text | `main.py` (on miss, to store), `decide_cache_hit` (for query) |
| `_tokenize` | `(text: str) → list[str]` | — | Lowercase whitespace tokenizer | `bm25_prefilter` |
| `bm25_prefilter` | `(query, entries) → list[tuple[CacheEntry, float]]` | 1 | BM25Okapi scoring, threshold filter, top-3 fallback | `decide_cache_hit` |
| `cosine_similarity` | `(vec_a, vec_b) → float` | — | Manual dot-product cosine, returns [-1, 1] | `_find_best_cosine_match` |
| `_find_best_cosine_match` | `(query_emb, candidates) → (entry, score, all_scores)` | 2 | Finds highest cosine match among BM25 candidates | `decide_cache_hit` |
| `crossencoder_score` | `(query, candidate_text) → float` | 3 | Cross-encoder pairwise re-ranking score | `decide_cache_hit` |
| `decide_cache_hit` | `(query, store) → (CacheEntry\|None, debug_dict)` | ALL | Orchestrates the full pipeline | `main.py` |

### `decide_cache_hit` — Complete Flow

```
Input: query (str), store (CacheStore)
  │
  ├─ store.all_entries() → entries
  │    └─ Empty? → return (None, {"reason": "empty_cache"})
  │
  ├─ Layer 1: bm25_prefilter(query, entries)
  │    ├─ Tokenize all cached queries
  │    ├─ BM25Okapi.get_scores()
  │    ├─ Normalize to [0, 1]
  │    ├─ Filter: score >= BM25_MIN_OVERLAP
  │    └─ Fallback: top-3 if none pass
  │         → bm25_candidates
  │
  ├─ embed_text(query) → query_embedding  [computed ONCE]
  │
  ├─ Layer 2: _find_best_cosine_match(query_embedding, bm25_candidates)
  │    ├─ cosine_similarity() for each candidate
  │    ├─ Select best match
  │    └─ best_sim < SIMILARITY_THRESHOLD?
  │         → return (None, {"reason": "no_similarity_match", ...})
  │
  ├─ Layer 3: crossencoder_score(query, best_entry.query_text)
  │    └─ ce_score < CROSSENCODER_MIN_SCORE?
  │         → return (None, {"reason": "crossencoder_rejected", ...})
  │
  └─ return (best_entry, {"reason": "hit", ...})
```

### Debug Dict — All Return Paths

| Reason | Returned When | Fields |
|---|---|---|
| `empty_cache` | No entries in store | `reason` |
| `no_bm25_candidates` | BM25 returned empty (guard) | `reason` |
| `no_similarity_match` | Best cosine sim < threshold | `reason`, `best_score`, `threshold`, `candidates` |
| `crossencoder_rejected` | Cross-encoder score < threshold | `reason`, `similarity`, `crossencoder_score`, `threshold_*`, `candidates` |
| `hit` | All 3 layers passed | `reason`, `similarity`, `crossencoder_score`, `matched_query`, `candidates` |

---

## Module: `backend/llm_client.py`

**Purpose:** Groq API client — sends chat completions, handles errors.

### `class GroqAPIError(Exception)`

| Field | Type | Description |
|---|---|---|
| `status_code` | `int` | HTTP status code (or synthetic 503/500) |
| `body` | `str` | Error message body |

### Functions

| Function | Signature | Description | Called By |
|---|---|---|---|
| `_get_client` | `() → Groq` | Lazy-init Groq SDK client singleton | `call_groq` |
| `call_groq` | `(query: str) → str` | Sends chat completion, returns response text | `main.py` on cache miss |

### Error Handling Flow

```
call_groq(query)
  │
  ├─ APIConnectionError → GroqAPIError(503, "Could not connect...")
  ├─ APIError           → GroqAPIError(exc.status_code, exc.body)
  └─ Exception (any)    → GroqAPIError(500, "Unexpected error...")
        │
        └─ Caught by main.py → JSONResponse(502, {"error": "llm_upstream_error", ...})
```

---

## Module: `backend/main.py`

**Purpose:** FastAPI application — the HTTP API layer and dashboard HTML server.

### Pydantic Models

| Model | Fields | Used For |
|---|---|---|
| `ChatMessage` | `role: str`, `content: str` | Individual message in the OpenAI-compatible shape |
| `ChatRequest` | `messages: list[ChatMessage]` | POST `/v1/chat/completions` request body |
| `ChatResponse` | `answer, cached, latency_ms, debug` | POST `/v1/chat/completions` response body |
| `StatsResponse` | `total_requests, cache_hits, ...` | GET `/stats` response body |
| `ConfigUpdate` | `similarity_threshold?, bm25_min_overlap?, crossencoder_min_score?` | POST `/config` request body |
| `ConfigResponse` | `similarity_threshold, bm25_min_overlap, crossencoder_min_score` | POST `/config` response body |
| `HealthResponse` | `status: str` | GET `/health` response body |
| `ErrorResponse` | `error, detail` | 502 error response |

### `class _Stats`

| Property / Method | Description |
|---|---|
| `total_requests` | Running counter |
| `cache_hits` | Running counter |
| `cache_misses` | Running counter |
| `total_hit_latency_ms` | Sum of hit latencies |
| `total_miss_latency_ms` | Sum of miss latencies |
| `estimated_dollars_saved` | `cache_hits × $0.03` |
| `average_miss_latency_ms` | `total_miss_latency_ms / cache_misses` |
| `estimated_time_saved_ms` | `cache_hits × average_miss_latency_ms` |
| `recent_queries` | Rolling list of last 20 query log entries |
| `record(entry)` | Appends to `recent_queries`, trims to 20 |

### Endpoints

| Endpoint | Method | Handler | Flow |
|---|---|---|---|
| `/` | GET | `serve_dashboard()` | Read `frontend/index.html` → return HTMLResponse |
| `/v1/chat/completions` | POST | `chat_completions()` | Extract query → `decide_cache_hit()` → hit/miss logic → return |
| `/stats` | GET | `get_stats()` | Return `_Stats` snapshot |
| `/health` | GET | `health_check()` | Return `{"status": "ok"}` |
| `/config` | POST | `update_config()` | Update `settings.*` thresholds → return current values |

### `chat_completions()` — Complete Flow

```
POST /v1/chat/completions { "messages": [...] }
  │
  ├─ Extract last user message (reversed scan for role="user")
  │    └─ No user message? → return empty with reason "no_user_message"
  │
  ├─ time.perf_counter() → start
  │
  ├─ decide_cache_hit(query, store)
  │    ├─ HIT → record stats → return ChatResponse(cached=True, ...)
  │    └─ MISS ↓
  │
  ├─ call_groq(query)
  │    └─ GroqAPIError? → record stats → return JSONResponse(502, ...)
  │
  ├─ embed_text(query) → query_embedding
  ├─ store.add(query, query_embedding, answer)
  │
  ├─ time.perf_counter() → end → latency_ms
  ├─ Record miss stats
  │
  └─ return ChatResponse(cached=False, answer=..., latency_ms=..., debug=...)
```

### Lifecycle

```
Startup (lifespan):
  1. Config already validated (config.py exits on missing GROQ_API_KEY)
  2. Log all configuration values
  3. CacheStore(settings.cache_persist_path) → loads existing entries
  4. Log entry count + "ready" message

Shutdown (lifespan):
  1. store.close() → closes SQLite connection
```

---

## Module: `frontend/index.html`

**Purpose:** Single-page dashboard UI served directly from FastAPI at the root URL (`/`). Pure HTML/CSS/JS — no build step, no framework.

### Views

| View | Activated By | Backend Interaction |
|---|---|---|
| **Query Tester** (chat UI) | Default / "Query Tester" nav | POST `/v1/chat/completions` |
| **Dashboard** | "Dashboard" nav | GET `/stats` |

### UI Components

| Component | Description | Backend Interaction |
|---|---|---|
| Settings dropdown | 3 range sliders (similarity, BM25, cross-encoder) + Save | POST `/config` on save |
| Backend status box | Green/red dot with pulsing animation and hover tooltip | GET `/config` (health check) every 15s |
| Chat scroll area | Renders user messages + assistant replies with HIT/MISS chip | — |
| Typing indicator | Animated 3-dot bounce while waiting for response | — |
| Composer | Auto-resize textarea + "Query tester" tag + Send button | — |
| KPI grid | 6 cards: Requests, Hits, Misses, Hit Rate, $ Saved, Time Saved | GET `/stats` |
| Query log table | Scrollable table of last 20 queries, newest first | GET `/stats` → `recent_queries` |
| Last Result card (sidebar) | Shows HIT/MISS + latency of the most recent query | Updated after each chat send |
| Repo Link (sidebar) | External link to the GitHub repo | — |

### JavaScript Architecture

All logic runs in a single IIFE (`(function(){ ... })()`). Key functions:

| Function | Purpose |
|---|---|
| `init()` | On page load: fetch config, set status, start 15s health poll |
| `refreshHealth()` | Polls `/config` every 15s, updates status dot |
| `sendQuery(text)` | POST to `/v1/chat/completions`, returns response data |
| `addAssistantMessage(data)` | Renders cache HIT/MISS chip, latency, markdown answer, details panel |
| `loadDashboard()` | GET `/stats`, renders KPI grid + query table |
| `populateSettingsInputs(cfg)` | Syncs sliders to current backend config values |
| `switchView(view)` | Toggles between chat and dashboard panels |

---

## Module: `tests/conftest.py`

### Fixtures

| Fixture | Scope | Returns | Description |
|---|---|---|---|
| `tmp_db_path` | function | `str` | Temporary SQLite path via `tmp_path` |
| `cache_store` | function | `CacheStore` | Fresh empty store, closed after test |

---

## Module: `tests/test_verification.py`

### Test Cases

| Test | Cached Query | Test Query | Expected |
|---|---|---|---|
| `test_empty_cache_returns_miss` | *(none)* | "How do I reset my password?" | `(None, {reason: "empty_cache"})` |
| `test_paraphrase_returns_hit` | "How do I reset my password?" | "What are the steps to recover my password?" | HIT |
| `test_trap_pair_returns_miss` | "How do I reset my password?" | "How do I delete my account?" | MISS |
| `test_threshold_sensitivity` | "How do I reset my password?" | "How can I change my password?" | MISS at 0.99, HIT at 0.30 |

---

## Cross-Module Data Flow (Full Request Lifecycle)

```
1. HTTP POST arrives at backend/main.py:chat_completions()
2. Pydantic validates ChatRequest
3. Last user message extracted from messages list
4. decide_cache_hit(query, store) called in backend/verification.py
   a. store.all_entries() returns in-memory list from cache_store.py
   b. bm25_prefilter() tokenizes and scores (Layer 1)
   c. embed_text() computes query embedding via sentence-transformers
   d. _find_best_cosine_match() scores candidates (Layer 2)
   e. crossencoder_score() re-ranks best match (Layer 3)
   f. Returns (entry_or_None, debug_dict)
5. If HIT: return cached answer immediately
6. If MISS:
   a. call_groq(query) in backend/llm_client.py → Groq API
   b. embed_text(query) for storage
   c. store.add(query, embedding, answer) in cache_store.py
      - INSERT INTO SQLite (parameterized)
      - Append to in-memory list
   d. Return fresh answer
7. Stats updated, response returned with latency_ms + debug dict
8. frontend/index.html displays result in chat bubble with HIT/MISS chip
```
