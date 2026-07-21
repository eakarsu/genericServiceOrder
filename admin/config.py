import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

# Runtime
APP_ENV = os.getenv("APP_ENV", "development")

# Database
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_DATABASE = os.getenv("DB_DATABASE", "generic_service_orders")
DB_PORT = os.getenv("DB_PORT", "5432")

DATABASE_URL = os.getenv("DATABASE_URL") or (
    f"postgresql://{quote_plus(DB_USER)}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"
)

# JWT
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Email (SMTP)
EMAIL_FROM = os.getenv("EMAIL_FROM", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

# App
ADMIN_PORT = int(os.getenv("ADMIN_PORT", "8001"))
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", FRONTEND_URL).split(",") if origin.strip()]

# Provider/webhook configuration
PROVIDER_MODE = os.getenv("PROVIDER_MODE", "local")
PROVIDER_WEBHOOK_SECRET = os.getenv("PROVIDER_WEBHOOK_SECRET", "")
PROVIDER_API_TOKEN = os.getenv("PROVIDER_API_TOKEN", "")
PROVIDER_URLS = {
    name: os.getenv(f"{name.upper()}_PROVIDER_URL", "").rstrip("/")
    for name in ("maps", "calendar", "messaging", "payment", "tax", "accounting")
}

# Rate limiting
RATE_LIMIT = os.getenv("RATE_LIMIT", "60/minute")


def validate_settings() -> None:
    failures: list[str] = []
    if not DATABASE_URL:
        failures.append("DATABASE_URL is required")
    if len(JWT_SECRET_KEY) < 32:
        failures.append("JWT_SECRET_KEY must contain at least 32 characters")
    if not CORS_ORIGINS or "*" in CORS_ORIGINS:
        failures.append("CORS_ORIGINS must contain explicit origins and cannot include '*'")
    if APP_ENV == "production":
        if PROVIDER_MODE == "local":
            failures.append("PROVIDER_MODE=local is forbidden in production")
        if len(PROVIDER_WEBHOOK_SECRET) < 32:
            failures.append("PROVIDER_WEBHOOK_SECRET must contain at least 32 characters in production")
        missing_providers = [name for name, url in PROVIDER_URLS.items() if not url.startswith("https://")]
        if missing_providers:
            failures.append("HTTPS provider URLs are required for: " + ", ".join(missing_providers))
        if len(PROVIDER_API_TOKEN) < 16:
            failures.append("PROVIDER_API_TOKEN must contain at least 16 characters in production")
        if any(not origin.startswith("https://") for origin in CORS_ORIGINS):
            failures.append("Production CORS origins must use HTTPS")
    if failures:
        raise RuntimeError("Invalid configuration:\n- " + "\n- ".join(failures))
