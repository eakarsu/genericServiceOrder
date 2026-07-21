from sqlalchemy import text

from admin.database import Base

VERSION = "20260720_service_order_lifecycle"


def upgrade(connection) -> None:
    # Importing registers every table on Base.metadata. This migration is safe on a
    # fresh database and idempotently adds the new lifecycle tables to an existing one.
    import admin.models  # noqa: F401

    Base.metadata.create_all(bind=connection)
    if connection.dialect.name == "postgresql":
        connection.execute(text("""
            CREATE OR REPLACE FUNCTION reject_service_event_mutation() RETURNS trigger AS $$
            BEGIN
              RAISE EXCEPTION 'service_order_events are append-only';
            END;
            $$ LANGUAGE plpgsql
        """))
        connection.execute(text("DROP TRIGGER IF EXISTS service_order_events_append_only ON service_order_events"))
        connection.execute(text("""
            CREATE TRIGGER service_order_events_append_only
            BEFORE UPDATE OR DELETE ON service_order_events
            FOR EACH ROW EXECUTE FUNCTION reject_service_event_mutation()
        """))
