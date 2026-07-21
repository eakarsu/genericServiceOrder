from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from admin.config import DATABASE_URL

engine_options = {"echo": False, "pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    engine_options["connect_args"] = {"check_same_thread": False}
engine = create_engine(DATABASE_URL, **engine_options)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def migration_status() -> tuple[bool, str | None]:
    inspector = inspect(engine)
    if "schema_migrations" not in inspector.get_table_names():
        return False, None
    with engine.connect() as connection:
        row = connection.exec_driver_sql(
            "SELECT version FROM schema_migrations ORDER BY applied_at DESC LIMIT 1"
        ).first()
    return bool(row), row[0] if row else None
