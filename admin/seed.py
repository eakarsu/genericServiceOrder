"""Create the minimum local administrator: python -m admin.seed.

This command is intentionally additive. Schema ownership belongs to migrations and
the command never deletes application data.
"""
import os

from sqlalchemy import inspect

from admin.config import APP_ENV
from admin.database import SessionLocal, engine
from admin.models.role import Role
from admin.models.user import User
from admin.services.auth_service import hash_password
from admin.services.password_service import validate_password_strength


def seed() -> None:
    if APP_ENV not in {"development", "test"}:
        raise RuntimeError("Seeding is permitted only in development or test")
    if "schema_migrations" not in inspect(engine).get_table_names():
        raise RuntimeError("Run `python -m admin.migrations.runner` before seeding")
    email = os.getenv("SEED_ADMIN_EMAIL", "").strip().lower()
    password = os.getenv("SEED_ADMIN_PASSWORD", "")
    valid, message = validate_password_strength(password)
    if not email or not valid:
        raise RuntimeError(f"SEED_ADMIN_EMAIL and a strong SEED_ADMIN_PASSWORD are required: {message}")

    with SessionLocal() as db:
        roles = {
            "admin": ("Administrator", {"service_orders": ["create", "read", "update", "approve"]}),
            "manager": ("Manager", {"service_orders": ["create", "read", "update"]}),
            "viewer": ("Viewer", {"service_orders": ["read"]}),
        }
        for name, (display_name, permissions) in roles.items():
            if not db.query(Role).filter(Role.name == name).first():
                db.add(Role(name=name, display_name=display_name, permissions=permissions))
        db.flush()
        admin_role = db.query(Role).filter(Role.name == "admin").one()
        user = db.query(User).filter(User.email == email).first()
        if user:
            user.password_hash = hash_password(password)
            user.name = "Local Administrator"
            user.role_id = admin_role.id
            user.is_active = True
            user.is_email_verified = True
            db.commit()
            print("Seed administrator credentials refreshed")
            return
        db.add(User(
            email=email, password_hash=hash_password(password), name="Local Administrator",
            role_id=admin_role.id, is_active=True, is_email_verified=True,
        ))
        db.commit()
    print(f"Created local administrator {email}")


if __name__ == "__main__":
    seed()
