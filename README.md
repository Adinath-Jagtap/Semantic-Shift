# Semantic-Shift : Semantic Cache Pipeline

> **Slash LLM API costs and latency by intelligently reusing past answers — without sacrificing accuracy.**

A caching proxy that sits in front of an LLM API (Groq) and uses a **four-layer verification pipeline** to decide whether a new query can safely reuse a cached answer. Unlike naive string-matching caches, this system understands that _"How do I reset my password?"_ and _"What are the steps to recover my password?"_ mean the same thing — while correctly rejecting _"How do I delete my account?"_ even though it shares 80 % of the same words.

<p align="center">
  <img src="docs/semantic_cache_architecture.png" alt="Semantic-Shift Architecture" width="700" />
</p>

---

## Key Features

- **4-Layer Semantic Verification** — Exact Match → BM25 keyword pre-filter → Cosine Similarity → Cross-Encoder re-ranking
- **Duplicate Question Detection** — Cross-encoder trained on Quora Question Pairs distinguishes _same intent_ from _same topic_
- **Embedding Cache** — In-memory dict caches all embeddings, so repeated queries cost < 1ms
- **Real-Time Dashboard** — KPI cards, query log table, and a live query tester (Streamlit)
- **Live Threshold Tuning** — Adjust all three verification thresholds via sliders without restarting the server
- **Backend Health Monitor** — Live connection status indicator
- **Cost & Time Tracking** — Estimated dollars saved and time saved updated per request
- **SQLite Persistence** — Cache survives server restarts with ACID guarantees
- **Single-Command Startup** — One `uvicorn` command serves both the API and the dashboard UI

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (backend/main.py)                      │
│                                                                          │
│   ┌──────────────┐                                                       │
│   │  Layer 0      │                                                       │
│   │  Exact & Typo │  ── string/char match ──▶  INSTANT HIT (< 1ms)      │
│   │  Match        │                                                       │
│   └──────┬───────┘                                                       │
│          │ no match                                                       │
│          ▼                                                                │
│   ┌──────────────┐    ┌──────────────────┐    ┌───────────────────────┐  │
│   │  Layer 1      │    │  Layer 2          │    │  Layer 3               │  │
│   │  Cosine       │ ──▶│  BM25             │ ──▶│  Cross-Encoder         │  │
│   │  Similarity   │    │  Sanity Check     │    │  Re-ranking            │  │
│   │  (embedding)  │    │  (keyword)        │    │  (intent matching)     │  │
│   └──────────────┘    └──────────────────┘    └───────────────────────┘  │
│         │                     │                        │                  │
│    Fast vector match    Confirms overlap        Confirms intent           │
│    (threshold: 0.72)    for borderline cases    (catches traps)          │
│                                                                          │
│   ┌──────────────┐    ┌──────────────────┐                               │
│   │  Groq LLM    │    │  SQLite Cache     │                               │
│   │  (on miss)   │    │  (persistent)     │                               │
│   └──────────────┘    └──────────────────┘                               │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│              Streamlit/HTML Dashboard (frontend/index.html)               │
└──────────────────────────────────────────────────────────────────────────┘
```

### Why Four Layers?

| Layer                           | Algorithm                                        | Speed    | Purpose                                                                                                     |
| ------------------------------- | ------------------------------------------------ | -------- | ----------------------------------------------------------------------------------------------------------- |
| **Layer 0** — Exact/Typo Match  | Normalized string compare & SequenceMatcher      | < 1ms    | Instant return for identical queries or minor typos (`pythn` → `python`). No ML involved.                   |
| **Layer 1** — Cosine Similarity | `all-MiniLM-L6-v2` embeddings (384-dim, L2-norm) | ~10-40ms | Vectorized `np.dot` finds the semantically closest match instantly among all cached entries.                |
| **Layer 2** — BM25              | Keyword overlap (TF-IDF)                         | ~1-5ms   | Sanity check for borderline cosine scores. Confirms vocabulary overlap before expensive cross-encoder step. |
| **Layer 3** — Cross-Encoder     | `quora-distilroberta-base` pairwise scoring      | ~200ms   | Confirms **intent** equivalence — not just topic similarity. Catches "roadmap for X" vs "what is X" traps.  |

**Layer 3 is selectively skipped** when cosine similarity >= 0.88 (high-confidence queries where false positives are essentially impossible), saving ~200ms.

**Total complexity:** Vectorized numpy operations + fallback ML = **fast AND accurate**.

---

## Quick Start

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com/) (free tier works)

### 1. Clone & Install

```bash
git clone https://github.com/Adinath-Jagtap/Semantic-Shift.git
cd Semantic-Shift

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```env
GROQ_API_KEY=gsk_your_key_here
```

### 3. Start the Server

```bash
uvicorn backend.main:app --reload
```

The server will:

- Load both ML models at startup (embedding + cross-encoder)
- Warm the embedding cache from SQLite
- **Fail fast** with a clear error if `GROQ_API_KEY` is missing
- Log all loaded configuration values
- Serve the dashboard UI at the root URL

### 4. Open the Dashboard

**Option A — Built-in Dashboard:** Navigate to **http://localhost:8000** in your browser.

**Option B — Streamlit Dashboard (richer UI):**

```bash
# In a second terminal:
cd dashboard
streamlit run app.py
```

Navigate to **http://localhost:8501**.

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## The Trap-Pair Demo

This is the demo that proves the system works. Try these queries in the Query Tester:

1. **First query:** `"How do I reset my password?"`
   - Result: **CACHE MISS** — no cached entries yet, calls Groq LLM (~1-2s)

2. **Second query:** `"What are the steps to recover my password?"`
   - Result: **CACHE HIT** — the system recognizes this is the same question, returns the cached answer in ~130ms instead of ~1-2s (10x faster)

3. **Third query:** `"How do I delete my account?"`
   - Result: **CACHE MISS** — despite sharing 80% of the words with query 1, the Cross-Encoder correctly identifies this as a **different intent** and rejects the cache match

4. **Fourth query:** `"How do I reset my password?"` (exact repeat)
   - Result: **CACHE HIT** — exact match short-circuit, **< 1ms latency**, zero ML

This is what naive string-matching or single-layer embedding caches get wrong.

---

## Latency Breakdown

| Scenario                                     | Latency        | Path                                  | Speedup vs LLM |
| -------------------------------------------- | -------------- | ------------------------------------- | -------------- |
| Exact or typo match                          | **< 1ms**      | Layer 0 (string/char match)           | ~2000x         |
| Paraphrase, repeated                         | **< 5ms**      | Embedding cache hit + numpy dot       | ~400x          |
| Paraphrase, high confidence (cosine >= 0.88) | **~15-45ms**   | Embed + cosine (CE skipped)           | ~30x           |
| Paraphrase, first time (borderline pipeline) | **~130-200ms** | Embed + cosine + BM25 + cross-encoder | ~10x           |
| Cache miss                                   | **~1-2s**      | Embed once + Groq API call            | 1x (baseline)  |

> **Note:** Semantic match latency (~40-130ms) is bounded by CPU neural network inference. A GPU would bring this under 5ms. Even on CPU, cache hits are **10-30x faster** than calling the LLM.

---

## Environment Variables

| Variable                       | Default                | Description                           |
| ------------------------------ | ---------------------- | ------------------------------------- |
| `GROQ_API_KEY`                 | _(required)_           | Your Groq API key                     |
| `GROQ_MODEL`                   | `llama-3.1-8b-instant` | LLM model to use                      |
| `CACHE_SIMILARITY_THRESHOLD`   | `0.72`                 | Min cosine similarity for Layer 1     |
| `CACHE_BM25_MIN_OVERLAP`       | `0.3`                  | Min normalized BM25 score for Layer 2 |
| `CACHE_CROSSENCODER_MIN_SCORE` | `0.25`                 | Min cross-encoder score for Layer 3   |
| `CACHE_PERSIST_PATH`           | `./cache_store.db`     | SQLite database file path             |

All thresholds can also be **tuned live** from the dashboard settings panel without restarting the server.

---

## API Endpoints

| Endpoint               | Method | Description                                                     |
| ---------------------- | ------ | --------------------------------------------------------------- |
| `/`                    | GET    | Dashboard UI (single-page app)                                  |
| `/v1/chat/completions` | POST   | Main proxy — OpenAI-compatible request shape                    |
| `/stats`               | GET    | Running totals (requests, hits, $ saved, time saved, query log) |
| `/health`              | GET    | Liveness probe                                                  |
| `/config`              | POST   | Read/update verification thresholds at runtime                  |

### Example API Call

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is machine learning?"}]}'
```

**Response:**

```json
{
  "answer": "Machine learning is a subset of artificial intelligence...",
  "cached": false,
  "latency_ms": 1842.3,
  "debug": {
    "reason": "empty_cache",
    "timing_ms": {}
  }
}
```

**Cache hit response:**

```json
{
  "answer": "Machine learning is a subset of artificial intelligence...",
  "cached": true,
  "latency_ms": 0.05,
  "debug": {
    "reason": "hit",
    "method": "exact_match",
    "similarity": 1.0,
    "crossencoder_score": "skipped",
    "matched_query": "What is machine learning?"
  }
}
```

---

## Tech Stack

| Technology                                             | Purpose                                               |
| ------------------------------------------------------ | ----------------------------------------------------- |
| **FastAPI** + **uvicorn**                              | High-performance async API server                     |
| **sentence-transformers** (`all-MiniLM-L6-v2`)         | 384-dim L2-normalized semantic embeddings             |
| **sentence-transformers** (`quora-distilroberta-base`) | Cross-encoder for duplicate question detection        |
| **rank-bm25** (`BM25Okapi`)                            | Keyword overlap pre-filtering                         |
| **numpy**                                              | Vectorized dot product for sub-microsecond cosine sim |
| **Groq SDK**                                           | Ultra-low-latency LLM inference (LPU hardware)        |
| **SQLite**                                             | Zero-dependency persistent cache with ACID guarantees |
| **Streamlit**                                          | Dashboard UI with live threshold sliders              |
| **python-dotenv**                                      | Environment variable loading                          |
| **pydantic**                                           | Request/response validation                           |
| **pytest**                                             | Test framework                                        |

---

## Project Structure

```
semantic-cache/
├── .env.example              # Environment variable template
├── requirements.txt          # Pinned Python dependencies
├── README.md                 # This file
├── CODE_FLOW.md              # Function-level data flow documentation
├── TECHNICAL_DECISIONS.md    # Library & algorithm rationale
├── CHANGES.md                # Optimization changelog
├── cache_store.db            # SQLite cache (auto-created on first run)
│
├── backend/                  # Python backend (FastAPI)
│   ├── __init__.py
│   ├── config.py             # Env loading, validation, mutable settings
│   ├── cache_store.py        # SQLite-backed cache with in-memory hot path
│   ├── verification.py       # 4-layer verification pipeline
│   ├── llm_client.py         # Groq API client
│   └── main.py               # FastAPI app + serves dashboard HTML
│
├── dashboard/                # Streamlit dashboard
│   └── app.py                # Streamlit app (query tester + stats)
│
├── frontend/                 # Built-in dashboard UI (served by FastAPI)
│   └── index.html            # Single-page app (HTML/CSS/JS)
│
└── tests/                    # Test suite
    ├── __init__.py
    ├── conftest.py            # Shared test fixtures
    └── test_verification.py   # 4 pytest cases (incl. trap-pair test)
```

---

## Test Cases

| Test                            | Cached Query                  | Test Query                                   | Expected                     |
| ------------------------------- | ----------------------------- | -------------------------------------------- | ---------------------------- |
| `test_empty_cache_returns_miss` | _(none)_                      | "How do I reset my password?"                | MISS (`reason: empty_cache`) |
| `test_paraphrase_returns_hit`   | "How do I reset my password?" | "What are the steps to recover my password?" | **HIT**                      |
| `test_trap_pair_returns_miss`   | "How do I reset my password?" | "How do I delete my account?"                | **MISS**                     |
| `test_threshold_sensitivity`    | "How do I reset my password?" | "How can I change my password?"              | MISS at 0.99, HIT at 0.30    |

---

## Performance Optimizations

| Optimization              | What                                               | Impact                               |
| ------------------------- | -------------------------------------------------- | ------------------------------------ |
| Exact-match short-circuit | Layer 0 string comparison before any ML            | < 1ms for repeated queries           |
| Embedding cache           | In-memory dict avoids redundant `model.encode()`   | < 0.01ms for cached embeddings       |
| Warm cache at startup     | Pre-populates embedding cache from SQLite          | First request after restart is fast  |
| L2-normalized embeddings  | Cosine sim = `np.dot()` (one C call)               | ~100x faster than Python loop        |
| Selective cross-encoder   | Only runs for borderline cosine scores (0.76–0.92) | Saves ~200ms on confident matches    |
| Reuse embedding on miss   | `decide_cache_hit()` returns computed embedding    | Avoids duplicate `embed_text()` call |
| Eager model loading       | Models load at server startup, not first request   | No cold-start spike                  |

---

## Known Limitations (v1)

1. **No streaming response support** — only whole-response replies.
2. **Single-message cache matching** — only the most recent user message is used; multi-turn conversation context is not embedded.
3. **No cache eviction / TTL** — entries persist indefinitely. Fine for a demo, not production.
4. **Single-process only** — cache and stats are not shared across multiple server instances.

These are documented in the code and acknowledged by design — the project never claims capabilities it doesn't have.

---

## Scalability Roadmap

| Current Design        | Production Change              | When Needed                  |
| --------------------- | ------------------------------ | ---------------------------- |
| In-memory Python list | FAISS index (IVF-PQ) or Qdrant | > 10,000 cache entries       |
| Single SQLite file    | PostgreSQL with pgvector       | Multi-instance deployment    |
| No cache eviction     | TTL-based eviction + LRU       | Cache > 1GB or stale answers |
| Single-process stats  | Redis counters or Prometheus   | Horizontal scaling           |
| No streaming          | SSE / WebSocket streaming      | Production chat UIs          |

---
