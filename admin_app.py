import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from admin.config import ADMIN_PORT, CORS_ORIGINS, validate_settings
from admin.database import SessionLocal, migration_status
from admin.migrations.runner import LATEST_VERSION
from admin.routers import auth, orders, users, sectors, dashboard, export, service_orders
from admin.middleware.error_handler import register_error_handlers
from admin.middleware.rate_limiter import setup_rate_limiter
from admin.middleware.security_headers import SecurityHeadersMiddleware
from admin.middleware.input_sanitizer import InputSanitizerMiddleware

app = FastAPI(
    title="Admin Dashboard API",
    description="Admin dashboard for GenericOrderingService",
    version="2.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(InputSanitizerMiddleware)

# Rate limiter
setup_rate_limiter(app)

# Error handlers
register_error_handlers(app)

# Routers
app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
app.include_router(orders.router, prefix="/api/orders", tags=["Orders"])
app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(sectors.router, prefix="/api/sectors", tags=["Sectors"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(export.router, prefix="/api/export", tags=["Export"])
app.include_router(service_orders.router, prefix="/api/service-orders", tags=["Service orders"])


@app.on_event("startup")
def on_startup():
    validate_settings()


@app.get("/")
def root():
    return {"message": "Generic Service Order API", "docs": "/docs"}


@app.get("/health/live", include_in_schema=False)
def health_live():
    return {"status": "ok"}


@app.get("/health/ready", include_in_schema=False)
def health_ready():
    try:
        migrated, version = migration_status()
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        if not migrated or version != LATEST_VERSION:
            raise HTTPException(status_code=503, detail={"status": "not_ready", "migration": version})
        return {"status": "ready", "migration": version}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=503, detail={"status": "not_ready"})


if __name__ == "__main__":
    uvicorn.run("admin_app:app", host="0.0.0.0", port=ADMIN_PORT, reload=False)
