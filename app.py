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

app = FastAPI(
    title="GenericOrderingService API",
    description=(
        "HTTP surface for the GenericOrderingService Python AI logic "
        "(Universal Service Bot across 19 sectors)."
    ),
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# This legacy AI experiment is never part of the production service boundary.
_APP_ENV = os.getenv("APP_ENV", "development")
_AI_ENABLED = os.getenv("ENABLE_EXPERIMENTAL_AI", "false").lower() == "true"
if _APP_ENV == "production" and _AI_ENABLED:
    raise RuntimeError("ENABLE_EXPERIMENTAL_AI is forbidden in production")

# CORS (explicit origins)
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[value.strip() for value in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if value.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

if _AI_ENABLED:
    from ai_router import router as ai_router
    from routers import order_copilot, sector_onboarding, semantic_menu, voice_ordering

    app.include_router(ai_router)
    app.include_router(voice_ordering.router)
    app.include_router(semantic_menu.router)
    app.include_router(sector_onboarding.router)
    app.include_router(order_copilot.router)


# ---------------------------------------------------------------------------
# Health / root
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "GenericOrderingService",
        "experimental_ai_enabled": _AI_ENABLED,
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
                "note": "Legacy experimental surface; use admin_app.py for service-order operations",
            }
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8013, reload=False)
