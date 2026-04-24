import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from admin.config import ADMIN_PORT, FRONTEND_URL
from admin.database import create_tables
from admin.routers import auth, orders, users, sectors, dashboard, export
from admin.middleware.error_handler import register_error_handlers
from admin.middleware.rate_limiter import setup_rate_limiter
from admin.middleware.security_headers import SecurityHeadersMiddleware
from admin.middleware.input_sanitizer import InputSanitizerMiddleware

app = FastAPI(
    title="Admin Dashboard API",
    description="Admin dashboard for GenericOrderingService",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:5173", "http://localhost:3000"],
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


@app.on_event("startup")
def on_startup():
    create_tables()


@app.get("/")
def root():
    return {"message": "Admin Dashboard API", "docs": "/docs"}


if __name__ == "__main__":
    uvicorn.run("admin_app:app", host="0.0.0.0", port=ADMIN_PORT, reload=True)
