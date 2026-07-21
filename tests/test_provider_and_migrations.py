import hashlib
import hmac
import os
import subprocess
import sys

from sqlalchemy import create_engine, inspect

from admin.services.provider_contracts import decode_payment_event, verify_webhook_signature


def test_webhook_signature_and_contract():
    body = b'{"event_id":"e1","event_type":"CHARGE_SUCCEEDED","order_id":"o1","amount_cents":100}'
    secret = "test-secret-with-more-than-32-characters"
    signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert verify_webhook_signature(body, signature, secret)
    assert not verify_webhook_signature(body + b" ", signature, secret)
    assert decode_payment_event(body)["amount_cents"] == 100


def test_migration_runner_is_repeatable(tmp_path):
    database = tmp_path / "migration.sqlite"
    env = {
        **os.environ,
        "DATABASE_URL": f"sqlite:///{database}",
        "JWT_SECRET_KEY": "test-key-that-is-at-least-32-characters",
        "CORS_ORIGINS": "http://localhost:5173",
    }
    command = [sys.executable, "-m", "admin.migrations.runner"]
    subprocess.run(command, check=True, env=env, capture_output=True, text=True)
    subprocess.run(command, check=True, env=env, capture_output=True, text=True)
    tables = inspect(create_engine(f"sqlite:///{database}")).get_table_names()
    assert {"schema_migrations", "service_orders", "service_order_events"}.issubset(tables)


def test_startup_and_container_do_not_mutate_schema():
    app_source = open("admin_app.py", encoding="utf-8").read()
    docker_source = open("Dockerfile", encoding="utf-8").read()
    assert "create_all" not in app_source
    assert "python:latest" not in docker_source
    assert "sel33man" not in docker_source
