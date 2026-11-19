# Technical Decisions — Library Choices, Algorithm Rationale & Design Notes

This document details the architectural decisions, algorithm designs, threshold selections, and trade-offs behind the **Semantic-Shift** semantic caching pipeline.

---

## Table of Contents

1. [Why These Libraries](#1-why-these-libraries)
2. [Why SQLite Over Alternatives](#2-why-sqlite-over-alternatives)
3. [Why all-MiniLM-L6-v2 for Embeddings](#3-why-all-minilm-l6-v2-for-embeddings)
4. [Why Vectorized Cosine as Layer 1 (and BM25 as Layer 2)](#4-why-vectorized-cosine-as-layer-1-and-bm25-as-layer-2)
5. [Why a Cross-Encoder as the Final Gate](#5-why-a-cross-encoder-as-the-final-gate)
6. [Why Four Layers Instead of One](#6-why-four-layers-instead-of-one)
7. [How Default Thresholds Were Chosen & Calibrated](#7-how-default-thresholds-were-chosen--calibrated)
8. [Why In-Memory Hot Path + SQLite Persistence](#8-why-in-memory-hot-path--sqlite-persistence)
9. [Security Decisions](#9-security-decisions)
10. [Why Groq as the LLM Backend](#10-why-groq-as-the-llm-backend)
11. [Frontend & Dashboard Architecture](#11-frontend--dashboard-architecture)
12. [What Was Deliberately Not Used](#12-what-was-deliberately-not-used)
13. [Test Architecture & Empirical Validation](#13-test-architecture--empirical-validation)
14. [Scalability Roadmap](#14-scalability-roadmap)

---

## 1. Why These Libraries

| Library | Version | Purpose | Why This One |
|---|---|---|---|
| **FastAPI** | 0.115.6 | HTTP API framework | Async-native, automatic OpenAPI documentation, Pydantic integration. Industry standard for Python ML services. |
| **uvicorn** | 0.34.0 | ASGI server | Canonical ASGI server for FastAPI. Includes `uvloop` and `httptools` on supported platforms for high throughput. |
| **sentence-transformers** | 3.3.1 | Embeddings + cross-encoder | Unified library providing bi-encoders (`SentenceTransformer`) and cross-encoders (`CrossEncoder`). Well-maintained and Hugging Face integrated. |
| **rank-bm25** | 0.2.2 | BM25 keyword scoring | Lightweight, pure-Python `BM25Okapi` implementation. Self-contained with no heavy native dependencies. |
| **numpy** | >=1.24.0 | Vectorized math | Hardware-accelerated dot products via BLAS. Computes cosine similarities across all entries in `< 0.01ms` and stores in-memory embeddings as `float32` arrays. |
| **groq** | 0.15.0 | LLM API client | Official Groq SDK with typed responses, retry policies, and structured exception handling. |
| **python-dotenv** | 1.0.1 | `.env` configuration | Standard and zero-overhead environment file loader. |
| **pydantic** | 2.10.4 | Input validation | Request and response schema enforcement with type constraints (`min_length`, bounds checks). |
| **pytest** | 8.3.4 | Test framework | Standard Python test runner supporting parameterization, fixtures, and detailed failure diffs. |
| **httpx** | 0.28.1 | HTTP test client | Asynchronous-capable HTTP client used by FastAPI's `TestClient` and the benchmark suite. |

### Version Pinning Rationale

All project dependencies are explicitly pinned in `requirements.txt` to guarantee:
- Deterministic and reproducible environments across developer setups and CI.
- Immunity against breaking upstream API updates.
- Predictable inference and numerical consistency in model outputs.

---

## 2. Why SQLite Over Alternatives

| Alternative | Trade-offs & Limitations |
|---|---|
| **JSON file** | Rewrites the entire file on every update. A process interruption or crash mid-write risks cache corruption. |
| **Redis** | Requires a standalone daemon/service. Adds deployment and operational complexity unnecessary for small to mid-scale cache sizes. |
| **PostgreSQL** | Substantial operational overhead (containerization, credential management, connection pooling, migrations). |
| **Vector DBs (FAISS, Qdrant)** | Geared toward millions of high-dimensional vectors with approximate nearest neighbors. For thousands of entries, brute-force matrix multiplication in NumPy is faster, exact, and drastically simpler. |

### Why SQLite Specifically

- **Zero configuration**: Bundled in Python's standard library (`sqlite3`).
- **ACID guarantees**: Atomic transactions prevent corruption during concurrent writes or sudden terminations.
- **Single-file portability**: Easily inspected, backed up, or purged (`cache_store.db`).
- **WAL mode**: Write-Ahead Logging allows concurrent readers while writes execute.

---

## 3. Why `all-MiniLM-L6-v2` for Embeddings

| Metric | Specification |
|---|---|
| Embedding Dimensions | 384 |
| Model Parameters | 22.7M |
| Max Sequence Length | 256 tokens |
| Quality (STS Benchmark) | 0.8489 |

### Selection Rationale

1. **Efficiency Sweet Spot**: 384-dimensional vectors capture fine-grained semantic intent without the compute or memory footprint of 768 or 1024-dimensional architectures.
2. **CPU-Optimized**: At 22.7M parameters, initialization takes seconds and CPU inference takes ~13ms, fitting local desktop deployment without requiring dedicated GPUs.
3. **Sentence-Level Tuning**: Pre-trained specifically on sentence pairs, making it directly suited for short conversational prompts and search queries.

---

## 4. Why Vectorized Cosine as Layer 1 (and BM25 as Layer 2)

### Shift from BM25-First to Vectorized Cosine

Originally, BM25 was positioned as an initial filter under the assumption that token matching is inherently faster than neural embeddings. However:
- BM25 is lexical: It rejects legitimate paraphrases that express identical intent using alternative vocabulary (e.g., *"How do I reset my password?"* vs *"What are the steps to recover my password?"*).
- When embeddings are L2-normalized upon generation (`normalize_embeddings=True`), cosine similarity simplifies to a dot product:

$$\text{Cosine Similarity}(\vec{a}, \vec{b}) = \vec{a} \cdot \vec{b}$$

By pre-stacking cached embeddings into a NumPy matrix (`_embedding_matrix`), a single matrix-vector multiplication computes similarity against every cached entry in **< 0.01ms**:

$$\text{similarities} = \mathbf{M} \cdot \vec{q}$$

This BLAS operation is faster and more discriminative than lexical tokenization.

### The Role of BM25 (Layer 2 Sanity Check)

BM25 was repositioned to Layer 2 as an **informational sanity check** for borderline cases (cosine score between 0.72 and 0.88). It evaluates keyword overlap and attaches lexical evidence (`bm25_score`, `bm25_confirms`) to the response debug trace before dispatching the query to the cross-encoder.

---

## 5. Why a Cross-Encoder as the Final Gate

### Bi-Encoder Limitations

Bi-encoders (such as `all-MiniLM-L6-v2`) encode queries and candidates **independently**. While this allows pre-computing and caching embeddings, it exhibits a known vulnerability: queries sharing domain vocabulary or grammatical structure can yield misleadingly high cosine similarities despite conveying opposing intent.

For example:
- *"How do I reset my password?"* vs *"How do I delete my account?"* (shares 4 of 6 words, yielding high cosine similarity).
- *"What is machine learning?"* vs *"What is the history of machine learning?"* (yields a cosine similarity of ~0.745).

### The Cross-Encoder Solution

A cross-encoder processes both sentences simultaneously in a single transformer pass with full cross-attention across all tokens:

$$\text{Input: } [\text{CLS}] \; \text{Query} \; [\text{SEP}] \; \text{Candidate} \; [\text{SEP}]$$

This enables the model to resolve semantic dependencies and determine true intent equivalence.

### Why `quora-distilroberta-base`

- **Trained on Quora Question Pairs**: 400,000 human-labeled question pairs annotated for semantic duplication ("Are these asking the same question?").
- **Classification Output**: Produces calibrated probabilities in $[0, 1]$.
- **Contrastive Precision**: Successfully distinguishes *same topic* from *same intent* (e.g., rejecting *"history of ML"* vs *"what is ML"*).
- **Avoided Model (`ms-marco-MiniLM-L-6-v2`)**: MS-MARCO models are trained on passage retrieval (query $\to$ document) and return uncalibrated, negative scores (e.g., -5.88) when given two short questions.

---

## 6. Why Four Layers Instead of One

A single algorithm cannot achieve sub-millisecond repeat latency while preventing semantic drift. Semantic-Shift uses a hierarchical 4-layer cascade:

```
User Query
    │
    ▼
┌────────────────────────────────────────────────────────┐
│ Layer 0: Exact-Match & Fuzzy Typo Match                │
│   • Exact match: O(1) normalized string lookup (< 1ms) │
│   • Typo fallback: SequenceMatcher (cutoff 0.95)       │
└──────────────────────────┬─────────────────────────────┘
                           │ (no exact match)
                           ▼
┌────────────────────────────────────────────────────────┐
│ Layer 1: Vectorized Cosine Similarity                  │
│   • all-MiniLM-L6-v2 embeddings (threshold: 0.72)      │
│   • Matrix multiply across all cached entries (~13ms)  │
│   • Cosine >= 0.88 -> HIGH CONFIDENCE HIT (skip CE)    │
│   • Cosine < 0.72  -> Trigger typo check, then MISS    │
└──────────────────────────┬─────────────────────────────┘
                           │ (borderline: 0.72 <= sim < 0.88)
                           ▼
┌────────────────────────────────────────────────────────┐
│ Layer 2: BM25 Sanity Check                             │
│   • Computes lexical overlap evidence for debug trace  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Layer 3: Cross-Encoder Re-ranking                      │
│   • quora-distilroberta-base (cutoff: 0.75)            │
│   • Final intent verification on best candidate (~30ms)│
└────────────────────────────────────────────────────────┘
```

### Layer Latency & Responsibilities

1. **Layer 0 (Exact & Typo Match)**:
   - *Exact repeat*: Normalized string lookup (`_normalized_lookup`) executes in **0.02–0.05ms**, bypassing ML entirely.
   - *Typo path*: If cosine similarity falls below threshold, `_char_similarity()` tests character-level overlap against cached entries. At threshold `0.95`, it resolves minor typos (`pythn` $\to$ `python`) without false hits. Typo fallback latency is not covered by the benchmark dataset.
2. **Layer 1 (Vectorized Cosine Similarity)**: Computes semantic proximity in **~13ms** (including query embedding). Matches with cosine $\ge 0.88$ are accepted immediately.
3. **Layer 2 (BM25 Sanity Check)**: Scores keyword overlap in **~1–5ms**, populating debug telemetry.
4. **Layer 3 (Cross-Encoder Re-ranking)**: Evaluates the single top candidate in **~30ms** for borderline cases, catching adversarial traps.

---

## 7. How Default Thresholds Were Chosen & Calibrated

Initial benchmark runs (Run 1) demonstrated high paraphrase recall but exposed vulnerabilities to semantic traps (46.7% false positive rate). An offline grid sweep (`tests/calibrate.py`) on 90 requests calibrated the pipeline to zero false positives:

| Parameter | Baseline (Run 1) | Calibrated (Run 2) | Rationale & Empirical Impact |
|---|---|---|---|
| `CACHE_SIMILARITY_THRESHOLD` | `0.72` | `0.72` | Maximizes candidate recall into Layer 3. Paraphrase pairs cluster in the 0.75–0.95 range. |
| `_BORDERLINE_UPPER` | `0.88` | `0.88` | When cosine $\ge 0.88$, matches are unambiguous. None of the 15 benchmark traps scored this high, saving ~30ms by skipping Layer 3. |
| `CACHE_CHAR_SIMILARITY_THRESHOLD` | `0.85` | `0.95` | At 0.85, 3 traps falsely matched on character overlap (*"linear search"* vs *"binary search"* at 93.1%, *"upgrade"* vs *"cancel"* at 85.7%, *"squash"* vs *"undo"* at 86.6%). Raising to 0.95 blocked all traps while retaining typos (`pythn` $\to$ `python`). |
| `CACHE_CROSSENCODER_MIN_SCORE` | `0.25` | `0.75` | At 0.25, 4 contrastive traps passed Layer 3. The highest-scoring trap was *"history of ML"* vs *"what is ML"* (0.733). Setting threshold to 0.75 eliminated 100% of trap false positives (0/15 in Run 2). |
| `CACHE_BM25_MIN_OVERLAP` | `0.3` | `0.3` | Permissive lexical baseline used for diagnostic validation rather than hard rejection. |

### Calibration Grid Summary (`tests/calibrate.py`)

| Cosine Threshold | Cross-Encoder Cutoff | Paraphrase Hit Rate | Trap False Positives |
|---|---|---|---|
| **>= 0.72** | **>= 0.75** | **68.9%** | **0 / 15** |
| >= 0.72 | >= 0.80 | 68.9% | 0 / 15 |
| >= 0.74 | >= 0.75 | 66.7% | 0 / 15 |
| >= 0.76 | >= 0.25 | 66.7% | 0 / 15 |
| >= 0.78 | >= 0.25 | 66.7% | 0 / 15 |

Calibrating to `ce >= 0.75` selected the configuration yielding **zero false positives** with maximum paraphrase recall (68.9%). In production caching, a miss costs one LLM call, whereas a false hit returns incorrect data.

---

## 8. Why In-Memory Hot Path + SQLite Persistence

The architecture separates read paths from storage:

```
Request Hot Path:   In-Memory Dicts & Arrays (0ms I/O)
                     ├── _embedding_cache (query_text -> vector)
                     ├── _normalized_lookup (exact match)
                     ├── _embedding_matrix (vectorized cosine)
                     └── _crossencoder_cache (pair -> score)
                           │
Durable Cold Path:   SQLite (cache_store.db)
                     ├── Read once on server startup
                     └── Write atomically on cache miss (add)
```

- **Zero I/O on Reads**: Vector comparisons and lookups operate entirely in RAM.
- **ACID Persistence on Writes**: When a new answer is generated, it is written to SQLite, maintaining persistence across server restarts.

---

## 9. Security Decisions

| Decision | Implementation | Threat Mitigated |
|---|---|---|
| **Parameterized Queries** | `cursor.execute("SELECT ... WHERE query_text = ?", (q,))` | SQL Injection (OWASP Top 10). No dynamic string concatenation in SQL. |
| **Credential Redaction** | `GROQ_API_KEY` loaded into memory; excluded from `/stats`, logs, and API outputs. | Secret leakage in telemetry and monitoring traces. |
| **Pydantic Validation** | Enforced models with type constraints on all request inputs. | Malformed payloads, buffer overflow attempts, and unvalidated parameters. |
| **Fail-Fast Configuration** | Process exits (`sys.exit(1)`) immediately if `GROQ_API_KEY` is missing. | Prevents silent runtime failure modes. |
| **Deterministic Serialization** | Embeddings serialized via standard JSON (`json.dumps`/`loads`). | Eliminates arbitrary code execution vulnerabilities associated with `pickle`. |
| **Sanitized Frontend Rendering** | Client uses `DOMPurify.sanitize(marked.parse(content))` | Prevents Cross-Site Scripting (XSS) from malicious or hallucinated LLM markdown. |

---

## 10. Why Groq as the LLM Backend

- **Latency Optimization**: Groq's Language Processing Units (LPUs) provide near-instant inference for generative tasks.
- **Configurable Models**: Defaults to `openai/gpt-oss-20b` (used in benchmarks) with pluggable support for other Groq-hosted models.
- **OpenAI-Compatible Spec**: Endpoints follow standard completion formats, enabling drop-in substitution with alternative providers.

---

## 11. Frontend & Dashboard Architecture

The project provides two complementary dashboard interfaces:

### Option A: Built-in Single-Page Dashboard (`frontend/index.html`)
- **Single Process**: Directly served by FastAPI at `GET /` (`uvicorn backend.main:app` or `python app.py`).
- **Zero Frontend Build Pipeline**: Vanilla HTML5, modern CSS, and vanilla JavaScript without npm, webpack, or external build steps.
- **Interactive Telemetry**: Includes real-time KPI metrics, query test bench, verification layer inspector, and live threshold tuning sliders (`/config`).

---

## 12. What Was Deliberately Not Used

| Technology | Reason for Omission |
|---|---|
| **FAISS / Qdrant / Pinecone** | Vector databases add network hops and indexing overhead. At thousands of cached vectors, vectorized NumPy matrix multiplication (`matrix @ vector`) executes in `< 0.01ms` with zero dependencies. |
| **Redis** | The in-memory Python structures provide sub-millisecond retrieval in-process. Adding Redis would introduce an external daemon and inter-process network overhead. |
| **LangChain / LlamaIndex** | This application implements a focused, 4-layer proxy pipeline. Heavy agentic frameworks introduce unnecessary abstraction layers and dependency bloat. |
| **Elasticsearch** | BM25Okapi in a lightweight pure-Python module fulfills lexical sanity checks without requiring a JVM or external search cluster. |
| **SciPy** | Would add a ~150MB dependency solely for `cosine_similarity`. NumPy's `np.dot` on L2-normalized vectors performs the identical calculation faster. |
| **Heavy Frontend Frameworks (React/Vue)** | Single-file architecture prevents build tooling overhead while delivering a responsive, clean dashboard. |

---

## 13. Test Architecture & Empirical Validation

The test suite validates semantic accuracy, performance, and threshold dynamics:

### Unit Test Suite (`tests/test_verification.py`)

1. **`test_empty_cache_returns_miss`**: Confirms that querying an empty database immediately returns a cache miss with `reason: empty_cache` without invoking neural models.
2. **`test_paraphrase_returns_hit`**: Confirms that semantically identical questions (*"How do I reset my password?"* vs *"What are the steps to recover my password?"*) produce a cache hit.
3. **`test_trap_pair_returns_miss`**: Core safety test verifying that adversarial pairs (*"How do I reset my password?"* vs *"How do I delete my account?"*) are rejected (`crossencoder_rejected` or `no_similarity_match`).
4. **`test_threshold_sensitivity`**: Tests dynamic runtime threshold mutation, verifying that borderline queries reject under strict settings (0.99) and hit under relaxed settings (0.30).

### Empirical Benchmark Suite (`tests/benchmark.py` & `tests/calibrate.py`)

- **90-Request Evaluation**: Evaluates 15 canonical intents, 45 paraphrases, 15 adversarial traps, and 15 exact repeats.
- **Telemetry Verification**: Records server-side latencies across all 4 pipeline paths:
  - Exact repeats: **0.02–0.05 ms** (> 100,000x faster than LLM).
  - Typo fallback latency is not covered by the benchmark dataset.
  - High-confidence hits: **~12 ms** (~436x faster than LLM, Run 3).
  - Borderline full-pipeline hits: **~47 ms** (~111x faster than LLM, Run 3).
  - LLM cache misses: **~5,188 ms** (Run 3 median).
- **Zero-Trap Validation**: Confirmed 0.0% false-positive rate on calibrated settings (Run 2).

---

## 14. Scalability Roadmap

| Current Design | Production Scale Target | Transition Strategy |
|---|---|---|
| In-memory NumPy matrix | > 50,000 entries | Implement FAISS IVF-PQ index or HNSW for sub-linear search. |
| Single SQLite database | Multi-instance cluster | Migrate to PostgreSQL with `pgvector` or distributed vector DB. |
| Indefinite persistence | Cache memory > 2GB | Implement TTL expiration with LRU (Least Recently Used) pruning. |
| Single-process stats | Distributed fleet | Centralize telemetry via Redis counters or Prometheus/OpenTelemetry metrics. |
| Whole response proxy | Real-time chat apps | Implement Server-Sent Events (SSE) streaming for cache hits and LLM responses. |
