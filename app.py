"""
Semantic-Shift — Main Application Runner
Runs the FastAPI backend and serves the frontend dashboard.

Usage:
    python app.py
"""

import os
import sys
from pathlib import Path
import uvicorn
from fastapi.responses import HTMLResponse

# Ensure root directory is in Python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import the existing FastAPI application from backend.main
from backend.main import app

# Ensure frontend/index.html is served at root "/"
FRONTEND_HTML = ROOT_DIR / "frontend" / "index.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """Serve the frontend single-page dashboard."""
    if not FRONTEND_HTML.exists():
        return HTMLResponse(
            f"<h3>Error: Frontend file not found at {FRONTEND_HTML}</h3>",
            status_code=404,
        )
    return HTMLResponse(FRONTEND_HTML.read_text(encoding="utf-8"))


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🚀 Semantic-Shift Cache Proxy is Starting...")
    print("  🌐 Dashboard UI:   http://127.0.0.1:8000")
    print("  📖 API Docs:       http://127.0.0.1:8000/docs")
    print("  💓 Health Check:   http://127.0.0.1:8000/health")
    print("=" * 60 + "\n")

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
