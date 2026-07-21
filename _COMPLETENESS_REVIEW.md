# Completeness Review: genericServiceOrder

**Review date:** 2026-07-18

## Assessment basis

Static inspection of project-owned source and configuration only; no dependency installation, build, database migration, external-service call, or runtime launch was performed. The scan considered 276 project files (101 source files), 2 manifest(s), 0 test-like file(s), and 0 CI workflow(s), excluding dependency/generated directories.

## Classification

**Functional but incomplete**

This is a substantive but unfinished field/local services application, not just an empty scaffold. Inspection found 101 source files across `sectors/`, `frontend/`, `prompts/`, `admin/` using Next.js, React, Express, Python; however, the checked-in workflow and delivery controls do not yet demonstrate a complete, production-operable product.

## Why it is not complete

- Generated gap/visualization routes describe missing capabilities or simulate recommendations; they do not implement the underlying domain operation.
- Generic LLM calls are used as product behavior without enough typed tools, grounded evidence, deterministic rules, or output evaluation.
- Mock, demo, sample, fixture, or placeholder behavior remains in executable/product paths.
- No recognizable project-owned automated tests were found for the main workflow.
- No checked-in CI workflow proves builds, tests, migrations, and security checks on every change.

## Needed features

1. Implement quote, availability, booking, dispatch, job status, change-order, invoice, payment, and cancellation lifecycles.
2. Add technician/resource skills, travel/service-area constraints, inventory, customer communications, and offline recovery.
3. Integrate maps, calendar, messaging, payment, tax, and accounting providers with idempotent webhooks.
4. Test overbooking, no-shows, partial work, refunds, rescheduling, and technician reassignment end to end.
5. Add risk-based unit, integration, and end-to-end tests in CI, including migration and failure-path coverage.

## Risks or launch blockers

- Weak/fallback secret patterns can permit forged sessions or accidental insecure deployments.
- Automation contains destructive process, filesystem, or database operations; do not run it on a shared machine without review.
- Startup appears coupled to seed/migration behavior, risking data mutation or non-repeatable launches.
- AI-provider availability, cost, privacy, prompt injection, and unvalidated output are launch risks until bounded and evaluated.

## Evidence inspected

- `README.md`
- `frontend/src/App.tsx:19`
- `prompts/p4.txt:68`
- `admin_app.py`
- `requirements.txt`
- `start.sh`

## Recommended next action

Choose one real field/local services journey, define acceptance criteria and external contracts, then close its persistence, permission, integration, failure, and test gaps before expanding features.

## Implementation progress (2026-07-20)

Implemented the field-service journey as the production `admin_app.py` boundary: deterministic quote creation/acceptance, skill/service-area/travel-buffer availability, concurrency-serialized booking, rescheduling, technician reassignment, dispatch and constrained job states, no-show/cancellation inventory release, completion consumption, independently approved change orders, invoicing, idempotent charge/refund events, customer-message outbox records, and version-conflict-aware offline replay. Lifecycle events form a verifiable SHA-256 chain and PostgreSQL rejects event updates/deletes.

Added portable SQLAlchemy models, a separately invoked versioned migration runner, migration-aware readiness, typed and bounded HTTP contracts for map/calendar/messaging/payment/tax/accounting providers, HMAC payment-webhook verification, explicit production provider requirements, and integer-cent accounting. Startup no longer creates schema or seeds data; the local seed is additive, environment-gated, and requires explicit credentials. Removed the embedded database/Redis and hard-coded password from the container, moved production to an unprivileged Python 3.12 image, enforced explicit CORS and strong secrets, hashed stored bearer tokens, required verified email for login, revoked sessions on password changes, and disabled the legacy unbounded AI routes by default and entirely in production.

Added backend lifecycle/failure tests, migration repeatability and provider-contract tests, a PostgreSQL 16 CI migration job, Python compilation, frontend lint/build/security-audit gates, deployment/environment documentation, and explicit operational/provider limitations. Local verification completed with 9 passing backend tests, successful Python compilation, successful frontend lint and production build, zero production npm audit findings, and a clean diff check.

## Runtime acceptance (2026-07-20)

The non-suite runtime validator passed on the fresh assigned PostgreSQL/API/UI ports `55657/6122/6123`: the validator invoked the root-level operator migration and additive seed commands against its disposable database, `start.sh` selected the project virtual environment and bound only the assigned loopback API port, the verified administrator authenticated, the hashed refresh token was persisted, and `/api/auth/me` reloaded the active user and role. The smoke test recorded `API_VERIFIED — startup_login_session_api`. Login accepts syntactically valid provisioned `.test` identities used by isolated acceptance environments, while public registration retains `EmailStr` deliverability validation. All 9 backend tests, Python compilation, frontend lint and production build, shell syntax, and `git diff --check` passed. All acceptance ports were released.
