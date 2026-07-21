#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
app_dir="${RUNTIME_PROJECT_SOURCE:-$project_dir}"
runtime_port="${ADMIN_PORT:-${PORT:-${BACKEND_PORT:-}}}"
frontend_port="${FRONTEND_PORT:-${CLIENT_PORT:-}}"
[[ "$runtime_port" =~ ^[0-9]+$ ]] || { echo "ADMIN_PORT, PORT, or BACKEND_PORT must be an assigned numeric port" >&2; exit 2; }
[[ "$frontend_port" =~ ^[0-9]+$ ]] || { echo "FRONTEND_PORT or CLIENT_PORT must be an assigned numeric port" >&2; exit 2; }
if lsof -tiTCP:"$runtime_port" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Assigned port $runtime_port is already in use; no process was stopped" >&2
  exit 1
fi
export ADMIN_PORT="$runtime_port"
export JWT_SECRET_KEY="${JWT_SECRET_KEY:-${JWT_SECRET:-}}"
export CORS_ORIGINS="${CORS_ORIGINS:-http://127.0.0.1:$frontend_port}"
export FRONTEND_URL="${FRONTEND_URL:-http://127.0.0.1:$frontend_port}"
export APP_ENV="${APP_ENV:-${NODE_ENV:-development}}"
cd "$app_dir"
if [[ -x "$app_dir/.venv/bin/uvicorn" ]]; then
  exec "$app_dir/.venv/bin/uvicorn" admin_app:app --host 127.0.0.1 --port "$runtime_port"
fi
exec uvicorn admin_app:app --host 127.0.0.1 --port "$runtime_port"
