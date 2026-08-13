# Technical Decisions — Library Choices, Algorithm Rationale & Design Notes

This document explains **why** each technical decision was made — the libraries chosen, the algorithms used, the threshold values, and the trade-offs accepted.

---

## Table of Contents

1. [Why These Libraries](#1-why-these-libraries)
2. [Why SQLite Over Alternatives](#2-why-sqlite-over-alternatives)
3. [Why all-MiniLM-L6-v2 for Embeddings](#3-why-all-minilm-l6-v2-for-embeddings)
4. [Why Vectorized Cosine as Layer 1](#4-why-vectorized-cosine-as-layer-1)
5. [Why a Cross-Encoder as the Final Gate](#5-why-a-cross-encoder-as-the-final-gate)
6. [Why Four Layers Instead of One](#6-why-four-layers-instead-of-one)
7. [How Default Thresholds Were Chosen](#7-how-default-thresholds-were-chosen)
8. [Why In-Memory List + SQLite Persistence](#8-why-in-memory-list--sqlite-persistence)
9. [Security Decisions](#9-security-decisions)
10. [Why Groq as the LLM Backend](#10-why-groq-as-the-llm-backend)
11. [Why a Native HTML/JS Dashboard](#11-why-a-native-htmljs-dashboard)
12. [What Was Deliberately Not Used](#12-what-was-deliberately-not-used)
13. [Scalability Roadmap](#13-scalability-roadmap)

---

## 1. Why These Libraries

| Library | Version | Purpose | Why This One |
|---|---|---|---|
| **FastAPI** | 0.115.6 | HTTP API framework | Async-native, automatic OpenAPI docs, Pydantic-integrated. The industry standard for Python ML APIs. |
| **uvicorn** | 0.34.0 | ASGI server | The canonical server for FastAPI. The `[standard]` extra adds `uvloop` and `httptools` for better performance. |
| **sentence-transformers** | 3.3.1 | Embeddings + cross-encoder | Single library provides both the bi-encoder (embeddings) and cross-encoder (re-ranking). Hugging Face ecosystem, well-maintained, battle-tested. |
| **rank-bm25** | 0.2.2 | BM25 keyword scoring | Minimal, pure-Python BM25Okapi implementation. No heavy dependencies. Does exactly one thing well. |
| **numpy** | >=1.24.0 | Vectorized math | Sub-microsecond cosine similarity via `np.dot()` on L2-normalized embeddings. Also stores embeddings as `float32` arrays in memory. |
| **groq** | 0.15.0 | LLM API client | Official Groq SDK with typed responses, retry logic, and proper error classes. Safer than raw `requests`. |
| **python-dotenv** | 1.0.1 | `.env` file loading | Standard, lightweight, universally understood. |
| **pydantic** | 2.10.4 | Input validation | Already a FastAPI dependency — used for request/response models with constrained types. |
| **pytest** | 8.3.4 | Testing | The standard Python test framework. Supports fixtures, parametrization, and clear assertion output. |
| **httpx** | 0.28.1 | Test client | FastAPI's recommended test client via `TestClient`. Async-compatible. |

### Why exact version pins?

Floating versions (`>=`) can introduce breaking changes between installs. Pinning to exact versions ensures:
- Reproducible builds across machines
- No surprise breakage from upstream updates
- Known-good dependency graph

---

## 2. Why SQLite Over Alternatives

| Alternative | Why Not |
|---|---|
| **JSON file** | Rewrites the entire file on every add. A crash mid-write corrupts the whole cache. SQLite writes one row at a time with ACID guarantees. |
| **Redis** | External server dependency. Overkill at hackathon scale (hundreds of entries). Adds operational complexity. |
| **PostgreSQL** | Same as Redis — external server, connection management, migrations. Way beyond what's needed. |
| **FAISS / Qdrant** | Vector databases are for millions of vectors with approximate nearest neighbor search. At hundreds of entries, a Python `for` loop is just as fast and far simpler. |

### Why SQLite specifically:

- **Zero dependencies** — it's in Python's standard library (`sqlite3`).
- **ACID transactions** — a crash can't corrupt the cache.
- **Single-file storage** — portable, easy to inspect, easy to delete and restart.
- **WAL mode** — enables concurrent reads while writing, which helps with the async server.

---

## 3. Why `all-MiniLM-L6-v2` for Embeddings

| Property | Value |
|---|---|
| Dimensions | 384 |
| Parameters | 22.7M |
| Max sequence length | 256 tokens |
| Speed | ~14,000 sentences/sec on CPU |
| Quality (STS benchmark) | 0.8489 |

### Why this model:

1. **Speed/accuracy sweet spot**: 384 dimensions is enough for semantic matching without the memory/compute cost of 768 or 1024-dim models.
2. **CPU-friendly**: 22M parameters means it loads fast and runs fast on a laptop without a GPU.
3. **Well-benchmarked**: Consistently top-tier on the MTEB leaderboard for its size class.
4. **Sentence-level**: Designed for comparing whole sentences (not paragraphs or documents), which is exactly our use case — short user queries.

### What was considered and rejected:

| Model | Why Not |
|---|---|
| `all-mpnet-base-v2` | Better quality but 2.5x slower (110M params). Overkill for demo-scale. |
| `text-embedding-3-small` (OpenAI) | External API call for embeddings would add latency to every request, defeating the purpose of caching. |
| `e5-small-v2` | Comparable quality but less ecosystem support and fewer benchmarks. |

---

## 4. Why Vectorized Cosine as Layer 1 (and BM25 as Layer 2)

### The old way (BM25 first):
Originally, BM25 was used to pre-filter candidates because string matching was assumed to be faster than neural network comparisons. But BM25 is actually quite noisy and misses paraphrases that use different words ("reset" vs "recover").

### The new way (Cosine first):
Because the embeddings are L2-normalized (`all-MiniLM-L6-v2`), cosine similarity reduces to a single dot product. By pre-stacking all cached embeddings into a single numpy matrix (`np.vstack`), we can compute the similarity against ALL cached queries in a **single C-optimized BLAS operation**.
- Vectorized numpy dot product takes **< 0.01ms** for hundreds of entries.
- It is actually faster than BM25 tokenization and scoring.
- It is far more accurate than BM25 for finding semantic matches.

### What BM25 does now (Sanity Check):
BM25 (Best Matching 25) is now used as a **sanity check** (Layer 2) only for borderline cosine scores (0.72 - 0.88). If the cosine similarity is unsure, we check if the words overlap. It adds evidence to the debug trace before the expensive cross-encoder step.

---

## 5. Why a Cross-Encoder as the Final Gate

### The bi-encoder limitation:

Bi-encoders (like `all-MiniLM-L6-v2`) encode each sentence **independently** and then compare their embeddings via cosine similarity. This is fast but has a known weakness:

> Sentences that share structure and topic but differ in **intent** can have high cosine similarity.

Example:
- "How do I reset my password?" → embedding A
- "How do I delete my account?" → embedding B
- `cosine_sim(A, B)` ≈ 0.75-0.85 — **dangerously close to the threshold**

### Why the cross-encoder fixes this:

A cross-encoder takes **both sentences as input simultaneously** through a single transformer pass. It sees the relationship between the words directly:

```
Input: [CLS] How do I reset my password? [SEP] How do I delete my account? [SEP]
Output: 0.12 (low — these are different requests)
```

vs.

```
Input: [CLS] How do I reset my password? [SEP] What are the steps to recover my password? [SEP]
Output: 8.7 (high — these are paraphrases)
```

The cross-encoder is ~10x slower than a bi-encoder comparison, but it only runs **once** — on the single best candidate from Layer 2. This is the layer that catches the "reset password" vs "delete account" trap.

### Why `quora-distilroberta-base`:

- **Trained on Quora Question Pairs** — a dataset of 400K question pairs labeled as duplicate or not. This is exactly our use case: "Are these two questions asking the same thing?"
- Outputs a **probability [0, 1]** where > 0.5 means "duplicate question".
- Correctly distinguishes **same intent** from **same topic** — e.g., "roadmap for AI engineer" vs "what is AI engineer" share the topic (AI) but have different intents (roadmap ≠ definition).

### What was considered and rejected:

| Model | Why Not |
|---|---|
| `ms-marco-MiniLM-L-6-v2` | Trained for passage retrieval (query → document). Returns nonsensical negative scores (-5.88) when given two short questions. Wrong task entirely. |
| `stsb-distilroberta-base` | Trained on Semantic Textual Similarity. Measures **topic overlap**, not **intent equivalence**. Scored "roadmap for AI" vs "what is AI" at 0.514 — barely above threshold — because they share the topic. |

---

## 6. Why Four Layers Instead of One

A single layer would be either too fast-and-inaccurate or too slow-and-accurate:

| Approach | Speed | Accuracy | Problem |
|---|---|---|---|
| BM25 only | ⚡⚡⚡ | ❌ | Misses paraphrases, false positives on keyword overlap |
| Cosine only | ⚡⚡ | ⚠️ | "Reset password" vs "delete account" has dangerously similar embeddings |
| Cross-encoder only | ⚡ | ✅ | O(n) cross-encoder calls per request — too slow with 100+ cache entries |

The four-layer pipeline gives us:

```
Layer 0 (Exact/Typo):    O(n) string comparison & char overlap — instant return, zero ML
Layer 1 (Cosine):        O(1) matrix multiply — picks the best candidate instantly
Layer 2 (BM25):          Sanity check for borderline candidates
Layer 3 (Cross-encoder): O(1) — one call on the single best candidate
```

**Total: Vectorized numpy operations + fallback ML = fast AND accurate.**

Layer 0 (exact/typo match) acts as a zero-cost optimization — if the user sends the exact same query text, or a minor typo variant (e.g. `pythn`), no ML models are invoked at all. This brings repeat-query latency from ~40ms to < 1ms.

---

## 7. How Default Thresholds Were Chosen

### `CACHE_SIMILARITY_THRESHOLD = 0.72`

- Cosine similarity of L2-normalized `all-MiniLM-L6-v2` embeddings on paraphrase pairs typically falls in the **0.75–0.95** range.
- Non-paraphrase but topic-related pairs fall in the **0.60–0.72** range.
- 0.72 lets more candidates through to Layer 3 (the cross-encoder), where intent is verified accurately. It also leaves room for typo matching to take over if below 0.72.
- The dashboard settings panel lets you tune this live.

### `CACHE_BM25_MIN_OVERLAP = 0.3`

- Normalized BM25 scores are in [0, 1] after dividing by the max score.
- 0.3 is intentionally permissive — BM25 is just a pre-filter, not a decision-maker.
- The top-3 fallback ensures even entries below 0.3 can reach Layer 2.

### `CACHE_CROSSENCODER_MIN_SCORE = 0.25`

- The `quora-distilroberta-base` cross-encoder outputs probabilities in [0, 1].
- While > 0.5 usually means "duplicate question", abbreviations (like "ML" vs "machine learning") can artificially lower the score to the 0.3-0.5 range.
- A threshold of 0.25 correctly accepts valid paraphrases while still firmly rejecting truly different intents (which score < 0.1).

### Cross-Encoder Skip Threshold = 0.88

- When cosine similarity >= 0.88, the match is near-identical — the cross-encoder is skipped to save ~200ms.
- Below 0.88, the cross-encoder confirms intent equivalence.
- This prevents false positives like "roadmap for AI" vs "what is AI" from being accepted without cross-encoder verification.

---

## 8. Why In-Memory List + SQLite Persistence

### The hybrid design:

```
Hot path (per-request):     In-memory Python list — zero I/O
Cold path (startup + add):  SQLite reads/writes — durable
```

### Why not just SQLite for everything?

At hackathon scale (hundreds of entries), reading from SQLite per-request would be fine in terms of speed. But:
- The verification pipeline needs random access to all embeddings (for cosine similarity) — easier with a Python list than SQLite cursors.
- BM25Okapi is built from a corpus array — it expects a Python list of token lists.
- The in-memory list is the natural data structure for this workload.

### Why not just an in-memory list (no SQLite)?

Without persistence, restarting the server loses all cached entries. SQLite gives crash-safe persistence with zero operational overhead.

---

## 9. Security Decisions

| Decision | Rationale |
|---|---|
| **Parameterized SQL only** | SQL injection is the #1 web vulnerability (OWASP). All queries use `?` placeholders — no string formatting ever touches SQL. |
| **GROQ_API_KEY never logged** | `config.py` reads the key but never prints it. `main.py` logs the model name and thresholds, never the key. API responses never include the key. |
| **Pydantic request validation** | Every endpoint uses typed Pydantic models with constraints (`min_length`, `ge`, `le`). Malformed requests are rejected before reaching business logic. |
| **CORS open (intentional)** | The dashboard is served from the same FastAPI origin, so CORS is set to `*` to allow the browser's same-origin fetch calls and any external tooling (Postman, curl). |
| **No eval/pickle/exec** | Embeddings are serialized via `json.dumps`/`json.loads` — safe, deterministic, no code execution. |
| **Error bodies don't leak internals** | Groq errors are wrapped in `GroqAPIError` with a sanitized body. Stack traces are logged server-side, not returned to the client. |
| **Fail-fast on missing config** | `sys.exit(1)` with a clear message if `GROQ_API_KEY` is missing — never silently run with an empty key. |
| **DOMPurify on rendered markdown** | The dashboard uses `DOMPurify.sanitize()` before injecting any LLM response HTML into the DOM, preventing XSS from malicious model output. |

---

## 10. Why Groq as the LLM Backend

Groq provides:
- **Ultra-low latency** (~200ms for short responses) via custom LPU hardware.
- **OpenAI-compatible API shape** — easy to swap for other providers later.
- **Free tier** with generous rate limits for hackathon use.
- **Official Python SDK** with typed responses and built-in retry logic.

The `GROQ_MODEL` is configurable via env var — defaulting to `llama-3.1-8b-instant` (fast, capable for general Q&A).

---

## 11. Why a Native HTML/JS Dashboard

The dashboard was originally built with Streamlit but was replaced with a custom HTML/CSS/JS single-page app for the following reasons:

| Requirement | Streamlit | Native HTML/JS |
|---|---|---|
| Chat-style UI with bubbles | ❌ Not possible | ✅ Full DOM control |
| Smooth animations & micro-interactions | ❌ Limited | ✅ Pure CSS/keyframes |
| Custom HIT/MISS chips inline with responses | ❌ Requires hacks | ✅ Trivial |
| Animated typing indicator | ❌ No native support | ✅ CSS animation |
| Served from same port as API | ❌ Requires separate process | ✅ `GET /` route in FastAPI |
| No separate process to manage | ❌ `streamlit run` needed | ✅ Single `uvicorn` command |
| Mobile-responsive layout | ❌ Limited | ✅ Full CSS media queries |

### Why no framework (React/Vue)?

The dashboard is a single HTML file with embedded CSS and JS. It has:
- No build step
- No `node_modules`
- No transpilation
- No bundler

This is intentional — a frontend build pipeline would add operational complexity for zero end-user benefit at hackathon scale.

### Why `marked.js` + `DOMPurify` from CDN?

- **marked.js** renders LLM responses as formatted markdown (code blocks, lists, etc.) — essential for readable answers.
- **DOMPurify** sanitizes the rendered HTML before injecting it into the DOM — prevents XSS from adversarial model output.
- Both loaded from CDN — no npm, no build step.

---

## 12. What Was Deliberately Not Used

| Technology | Why Excluded |
|---|---|
| **FAISS** | Approximate nearest neighbor search. At hundreds of entries, brute-force cosine is just as fast and simpler. FAISS adds a C extension dependency and index management complexity. |
| **Qdrant / Pinecone / Weaviate** | Vector databases for million-scale datasets. Massive operational overhead for zero benefit at this scale. |
| **Redis** | In-memory cache server. We already have an in-memory Python list — Redis would add a network hop and a separate process for no gain. |
| **LangChain** | Framework for LLM application orchestration. This project has exactly one LLM call and one cache check — LangChain would add layers of abstraction over a 10-line function. |
| **Elasticsearch** | Full-text search engine. BM25Okapi in a 200-line Python library does the same thing at this scale without running a JVM. |
| **scipy** | Would be used for `cosine_similarity()`, but importing scipy for a single dot-product/magnitude calculation adds a 150MB dependency. A 4-line manual implementation is equivalent. |
| **numpy** | `sentence-transformers` already uses it internally. We call `.tolist()` to convert to plain Python lists for JSON serialization, but we don't import numpy directly — keeping the dependency surface minimal. |
| **Streamlit** | Originally used for the dashboard. Replaced with a native HTML/CSS/JS SPA because Streamlit cannot produce a chat-style UI, smooth animations, or inline HIT/MISS chips. See section 11. |
| **React / Vue / Angular** | Frontend frameworks require a build step, Node.js, and a separate dev server. The dashboard's requirements are met with a single self-contained HTML file. |

---

## 13. Scalability Roadmap

What you'd change for production (beyond hackathon scale):

| Current Design | Production Change | When Needed |
|---|---|---|
| In-memory Python list | FAISS index (IVF-PQ) or Qdrant | > 10,000 cache entries |
| Single SQLite file | PostgreSQL with pgvector | Multi-instance deployment |
| No cache eviction | TTL-based eviction + LRU | Cache > 1GB or stale answers |
| Single-process stats | Redis counters or Prometheus | Horizontal scaling |
| No streaming | SSE / WebSocket streaming | Production chat UIs |
| Single user message | Full conversation hashing | Multi-turn applications |
| `threading.Lock` | `asyncio.Lock` or per-request isolation | High concurrency (> 100 RPS) |
| Manual cosine similarity | Batch matrix multiply (numpy) | > 1,000 candidates per request |
| CDN-loaded JS libraries | Bundled + minified assets | Offline/air-gapped deployments |
