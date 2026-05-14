#!/usr/bin/env bash
# Launch the GenericOrderingService FastAPI app.
# Prerequisites:
#   pip install -r requirements.txt
#   export OPENROUTER_API_KEY=...   (else /api/ai/* return 503)
set -euo pipefail
cd "$(dirname "$0")"
exec uvicorn app:app --reload --host 0.0.0.0 --port 8013
