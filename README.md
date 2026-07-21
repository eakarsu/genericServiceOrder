# Generic Service Order

Generic Service Order is a transactional API and operator dashboard for field-service work. The supported journey is quote → acceptance → availability → booking → dispatch → field status → change approval → invoice → payment/refund or cancellation.

The production application is `admin_app.py`. The older `app.py` AI/voice experiment is disabled by default and cannot be enabled when `APP_ENV=production`; it is not a source of truth for pricing, availability, booking, payment, or job state.

## What is implemented

- deterministic quote totals and tax inputs expressed in integer cents;
- technician skills, service areas, travel buffers, overbooking protection, rescheduling, and reassignment;
- inventory reservation, release on cancellation/no-show, and consumption on completion;
- constrained job-state transitions, partial work, independently approved change orders, invoicing, partial/full payment, and refund events;
- HMAC-authenticated, idempotent payment webhooks that reject event-ID payload substitution;
- queued customer communication records, typed seams for map/calendar/message/payment/tax/accounting providers, and conflict-aware offline field updates;
- append-only PostgreSQL lifecycle events with a verifiable SHA-256 hash chain;
- explicit, versioned migrations and separate live/ready health endpoints.

Provider contracts are integration boundaries, not claims that a third-party account is configured. `PROVIDER_MODE=local` is accepted for development/tests and rejected in production.

## Local setup

Requirements: Python 3.12+, PostgreSQL 16+, and Node 22+ for the dashboard.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-admin.txt
cp .env.example .env
# Replace every example credential in .env.
python -m admin.migrations.runner
SEED_ADMIN_EMAIL=admin@example.test SEED_ADMIN_PASSWORD='a-strong-local-password' python -m admin.seed
./start.sh
```

Migrations are never run during server startup. Run them as a distinct deployment step before switching traffic. The seed command is additive, development/test-only, requires an explicit strong password, and never clears data.

The API listens on port 8001 by default. `/health/live` reports process liveness; `/health/ready` returns 503 unless the database is reachable and the latest migration is recorded. OpenAPI is at `/docs`.

For the dashboard:

```bash
cd frontend
npm ci
npm run dev
```

## Payment webhook contract

Send JSON to `POST /api/service-orders/webhooks/payment/{provider}` with an `X-Webhook-Signature: sha256=<hex hmac>` header. The HMAC is SHA-256 over the exact request bytes using `PROVIDER_WEBHOOK_SECRET`.

```json
{
  "event_id": "evt_123",
  "event_type": "CHARGE_SUCCEEDED",
  "order_id": "order UUID",
  "amount_cents": 10800
}
```

Supported event types are `CHARGE_SUCCEEDED` and `REFUND_SUCCEEDED`. Replays return the original result; a reused event ID with different bytes is rejected.

## Security and operations

- Set a random `JWT_SECRET_KEY` of at least 32 characters and explicit `CORS_ORIGINS`. Production origins must be HTTPS.
- Refresh tokens are rotated and stored only as SHA-256 digests; password changes revoke outstanding refresh tokens.
- Registered users must verify email before login. Roles protect administration endpoints.
- The application container runs as an unprivileged user and contains neither PostgreSQL nor Redis. Use managed/external services and secret injection.
- Back up PostgreSQL before migration. Lifecycle event rows must not be edited; PostgreSQL installs a trigger that rejects updates/deletes.
- Customer messages form an outbox. A production deployment must supply a worker/provider adapter with retry/dead-letter monitoring before promising delivery.

## Verification

```bash
pytest -q
python -m compileall -q admin admin_app.py app.py
cd frontend && npm ci && npm run lint && npm run build
```

CI runs migrations against PostgreSQL 16, backend lifecycle and failure-path tests, compilation, lint, and the frontend production build. The tests cover successful fulfillment, inventory accounting, overbooking/travel buffers, no-shows, partial work, payments/refunds, webhook replays, rescheduling, reassignment, change-order separation of duties, offline conflicts, migration repeatability, and event-chain tampering.

The separate `requirements.txt` belongs only to the disabled legacy AI/voice experiment. Production images install `requirements-admin.txt` exclusively.
