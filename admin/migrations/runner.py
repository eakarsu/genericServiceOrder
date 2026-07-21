from __future__ import annotations

from sqlalchemy import text

from admin.database import engine
from admin.migrations.versions import v20260720_service_order_lifecycle

MIGRATIONS = [v20260720_service_order_lifecycle]
LATEST_VERSION = MIGRATIONS[-1].VERSION


def migrate() -> None:
    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(100) PRIMARY KEY,
                applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """))
        applied = {row[0] for row in connection.execute(text("SELECT version FROM schema_migrations"))}
        for migration in MIGRATIONS:
            if migration.VERSION in applied:
                continue
            migration.upgrade(connection)
            connection.execute(
                text("INSERT INTO schema_migrations(version) VALUES (:version)"),
                {"version": migration.VERSION},
            )
            print(f"Applied migration {migration.VERSION}")


if __name__ == "__main__":
    migrate()
