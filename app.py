"""
GenericOrderingService — FastAPI application entrypoint.

Bootstraps a minimal HTTP surface around the existing Python AI logic
(``updated_universal_service_bot.py`` and friends), per Apply pass 3
(Group B). The actual AI endpoints live in ``ai_router.py``; this module
just wires the app, CORS, static FE, and health.

Run with::

    uvicorn app:app --reload --port 8013

A convenience ``start.sh`` is also provided.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ai_router import router as ai_router


app = FastAPI(
    title="GenericOrderingService API",
    description=(
        "HTTP surface for the GenericOrderingService Python AI logic "
        "(Universal Service Bot across 19 sectors)."
    ),
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# CORS (localhost dev)
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8013",
        "http://127.0.0.1:8013",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(ai_router)
from routers import voice_ordering as _vo, semantic_menu as _sm, sector_onboarding as _so, order_copilot as _oc  # noqa: E402
app.include_router(_vo.router); app.include_router(_sm.router); app.include_router(_so.router); app.include_router(_oc.router)


# ---------------------------------------------------------------------------
# Health / root
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "GenericOrderingService",
        "ai_configured": bool(os.getenv("OPENROUTER_API_KEY")),
    }


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(_STATIC_DIR / "index.html"))

else:

    @app.get("/")
    def index_fallback() -> JSONResponse:
        return JSONResponse(
            {
                "service": "GenericOrderingService",
                "docs": "/docs",
                "ai": "/api/ai/status",
                "note": "static/ frontend not found",
            }
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8013, reload=True)
