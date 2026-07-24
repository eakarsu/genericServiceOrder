#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
app_dir="${RUNTIME_PROJECT_SOURCE:-$project_dir}"
cd "$app_dir"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source ./.env
  set +a
fi
runtime_port="${ADMIN_PORT:-${PORT:-${BACKEND_PORT:-}}}"
frontend_port="${FRONTEND_PORT:-${CLIENT_PORT:-}}"
[[ "$runtime_port" =~ ^[0-9]+$ ]] || { echo "ADMIN_PORT, PORT, or BACKEND_PORT must be an assigned numeric port" >&2; exit 2; }
[[ "$frontend_port" =~ ^[0-9]+$ ]] || { echo "FRONTEND_PORT or CLIENT_PORT must be an assigned numeric port" >&2; exit 2; }
[[ "$runtime_port" != "$frontend_port" ]] || { echo "API and frontend ports must be different" >&2; exit 2; }
for assigned_port in "$runtime_port" "$frontend_port"; do
  if lsof -tiTCP:"$assigned_port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Assigned port $assigned_port is already in use; no process was stopped" >&2
    exit 1
  fi
done
export ADMIN_PORT="$runtime_port"
export JWT_SECRET_KEY="${JWT_SECRET_KEY:-${JWT_SECRET:-}}"
export CORS_ORIGINS="${CORS_ORIGINS:-http://127.0.0.1:$frontend_port}"
export FRONTEND_URL="${FRONTEND_URL:-http://127.0.0.1:$frontend_port}"
export API_PROXY_TARGET="http://127.0.0.1:$runtime_port"
export APP_ENV="${APP_ENV:-${NODE_ENV:-development}}"

child_pids=""
cleanup() {
  trap - EXIT INT TERM
  for child_pid in $child_pids; do kill "$child_pid" >/dev/null 2>&1 || true; done
  for child_pid in $child_pids; do wait "$child_pid" >/dev/null 2>&1 || true; done
}
trap cleanup EXIT INT TERM

if [[ -x "$app_dir/.venv/bin/uvicorn" ]]; then
  "$app_dir/.venv/bin/uvicorn" admin_app:app --host 127.0.0.1 --port "$runtime_port" &
else
  uvicorn admin_app:app --host 127.0.0.1 --port "$runtime_port" &
fi
api_pid=$!
child_pids="$api_pid"

npm --prefix frontend run dev -- --host 127.0.0.1 --port "$frontend_port" --strictPort &
ui_pid=$!
child_pids="$child_pids $ui_pid"

echo "Generic Service Order API listening on http://127.0.0.1:$runtime_port"
echo "Generic Service Order UI listening on http://127.0.0.1:$frontend_port"

while kill -0 "$api_pid" >/dev/null 2>&1 && kill -0 "$ui_pid" >/dev/null 2>&1; do sleep 1; done
runtime_result=1
if ! kill -0 "$api_pid" >/dev/null 2>&1; then
  wait "$api_pid" || runtime_result=$?
else
  wait "$ui_pid" || runtime_result=$?
fi
exit "$runtime_result"
