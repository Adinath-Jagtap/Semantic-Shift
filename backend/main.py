"""
FastAPI application — the Semantic-Shift cache proxy API server.

# KNOWN LIMITATIONS (v1)
# ─────────────────────
# 1. No streaming response support — only whole-response replies.
# 2. Only the single most recent user message is used for cache matching —
#    multi-turn conversation context is not embedded.
# 3. No cache eviction / TTL — entries persist indefinitely.
#    Fine for a hackathon demo, not production.
# 4. Single-process only — cache and stats are not shared across
#    multiple server instances.
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from backend.cache_store import CacheStore
from backend.config import settings
from backend.llm_client import GroqAPIError, call_groq
from backend.verification import decide_cache_hit, embed_text

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("semantic_cache")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

# ---------------------------------------------------------------------------
# Application state (module-level singletons)
# ---------------------------------------------------------------------------

store: CacheStore | None = None


class _Stats:
    """In-memory running totals for the /stats endpoint."""

    def __init__(self) -> None:
        self.total_requests: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.total_hit_latency_ms: float = 0.0
        self.total_miss_latency_ms: float = 0.0
        # Rolling log of the last 20 queries for the dashboard.
        self.recent_queries: list[dict[str, Any]] = []

    @property
    def estimated_dollars_saved(self) -> float:
        return self.cache_hits * settings.estimated_cost_per_call

    @property
    def average_miss_latency_ms(self) -> float:
        if self.cache_misses == 0:
            return 0.0
        return self.total_miss_latency_ms / self.cache_misses

    @property
    def estimated_time_saved_ms(self) -> float:
        return self.cache_hits * self.average_miss_latency_ms

    def record(self, entry: dict[str, Any]) -> None:
        """Append a query log entry, keeping only the last 20."""
        self.recent_queries.append(entry)
        if len(self.recent_queries) > 20:
            self.recent_queries = self.recent_queries[-20:]


stats = _Stats()

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str
    cached: bool
    latency_ms: float
    debug: dict[str, Any]


class StatsResponse(BaseModel):
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    estimated_dollars_saved: float
    estimated_time_saved_ms: float
    average_miss_latency_ms: float
    recent_queries: list[dict[str, Any]]


class ConfigUpdate(BaseModel):
    similarity_threshold: float | None = Field(None, ge=0.0, le=1.0)
    bm25_min_overlap: float | None = Field(None, ge=0.0, le=1.0)
    crossencoder_min_score: float | None = Field(None, ge=-10.0, le=10.0)


class ConfigResponse(BaseModel):
    similarity_threshold: float
    bm25_min_overlap: float
    crossencoder_min_score: float


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    error: str
    detail: str


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: validate config, load persisted cache. Shutdown: close DB."""
    global store

    # Config is already validated at import time (config.py exits on missing key).
    logger.info("GROQ_MODEL=%s", settings.groq_model)
    logger.info("CACHE_PERSIST_PATH=%s", settings.cache_persist_path)
    logger.info("SIMILARITY_THRESHOLD=%.2f", settings.similarity_threshold)
    logger.info("BM25_MIN_OVERLAP=%.2f", settings.bm25_min_overlap)
    logger.info("CROSSENCODER_MIN_SCORE=%.2f", settings.crossencoder_min_score)

    store = CacheStore(settings.cache_persist_path)
    logger.info(
        "Cache loaded: %d existing entries from %s",
        len(store.entries),
        settings.cache_persist_path,
    )
    logger.info("Semantic-Shift is ready.")

    yield  # Application runs here.

    store.close()
    logger.info("Cache store closed.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Semantic-Shift",
    description="Semantic cache proxy with 3-layer verification for LLM APIs.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow requests from any origin (dashboard is served from same origin,
# but keeping this open for external tools / testing).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Dashboard — serve the single-page HTML app
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard() -> HTMLResponse:
    """Serve the dashboard SPA from the root URL."""
    return HTMLResponse(_DASHBOARD_HTML.read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest) -> ChatResponse:
    """
    Main proxy endpoint — OpenAI-compatible shape.

    Flow: extract last user message → decide_cache_hit() →
      on hit: return cached answer
      on miss: call Groq, store the new entry, return fresh answer
    """
    # Extract the last user message for cache lookup (spec §7 — v1 limitation).
    last_user_msg = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            last_user_msg = msg.content
            break

    if last_user_msg is None:
        return ChatResponse(
            answer="",
            cached=False,
            latency_ms=0.0,
            debug={"reason": "no_user_message"},
        )

    start = time.perf_counter()

    # --- Cache check ---
    hit_entry, debug = decide_cache_hit(last_user_msg, store)

    if hit_entry is not None:
        # Cache HIT
        latency_ms = (time.perf_counter() - start) * 1000
        stats.total_requests += 1
        stats.cache_hits += 1
        stats.total_hit_latency_ms += latency_ms
        stats.record(
            {
                "query": last_user_msg,
                "cached": True,
                "latency_ms": round(latency_ms, 2),
                "reason": debug.get("reason"),
                "similarity": debug.get("similarity"),
                "crossencoder_score": debug.get("crossencoder_score"),
            }
        )
        return ChatResponse(
            answer=hit_entry.answer,
            cached=True,
            latency_ms=round(latency_ms, 2),
            debug=debug,
        )

    # --- Cache MISS — call Groq ---
    try:
        answer = call_groq(last_user_msg)
    except GroqAPIError as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        stats.total_requests += 1
        stats.cache_misses += 1
        stats.total_miss_latency_ms += latency_ms
        stats.record(
            {
                "query": last_user_msg,
                "cached": False,
                "latency_ms": round(latency_ms, 2),
                "reason": "groq_error",
                "error": exc.body,
            }
        )
        # Return a proper 502 with JSON error body — don't crash the server.
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=502,
            content={
                "error": "llm_upstream_error",
                "detail": exc.body,
            },
        )

    # Store the new entry (writes to SQLite + appends to in-memory list).
    query_embedding = embed_text(last_user_msg)
    store.add(last_user_msg, query_embedding, answer)

    latency_ms = (time.perf_counter() - start) * 1000

    stats.total_requests += 1
    stats.cache_misses += 1
    stats.total_miss_latency_ms += latency_ms
    stats.record(
        {
            "query": last_user_msg,
            "cached": False,
            "latency_ms": round(latency_ms, 2),
            "reason": debug.get("reason"),
        }
    )

    return ChatResponse(
        answer=answer,
        cached=False,
        latency_ms=round(latency_ms, 2),
        debug=debug,
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    total = stats.total_requests
    return StatsResponse(
        total_requests=total,
        cache_hits=stats.cache_hits,
        cache_misses=stats.cache_misses,
        hit_rate=round(stats.cache_hits / total, 4) if total > 0 else 0.0,
        estimated_dollars_saved=round(stats.estimated_dollars_saved, 4),
        estimated_time_saved_ms=round(stats.estimated_time_saved_ms, 2),
        average_miss_latency_ms=round(stats.average_miss_latency_ms, 2),
        recent_queries=stats.recent_queries,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/config", response_model=ConfigResponse)
async def update_config(update: ConfigUpdate) -> ConfigResponse:
    if update.similarity_threshold is not None:
        settings.similarity_threshold = update.similarity_threshold
        logger.info("similarity_threshold updated to %.4f", update.similarity_threshold)

    if update.bm25_min_overlap is not None:
        settings.bm25_min_overlap = update.bm25_min_overlap
        logger.info("bm25_min_overlap updated to %.4f", update.bm25_min_overlap)

    if update.crossencoder_min_score is not None:
        settings.crossencoder_min_score = update.crossencoder_min_score
        logger.info(
            "crossencoder_min_score updated to %.4f", update.crossencoder_min_score
        )

    return ConfigResponse(
        similarity_threshold=settings.similarity_threshold,
        bm25_min_overlap=settings.bm25_min_overlap,
        crossencoder_min_score=settings.crossencoder_min_score,
    )
