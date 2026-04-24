import json
import bleach
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class InputSanitizerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                body = await request.body()
                if body:
                    try:
                        data = json.loads(body)
                        sanitized = self._sanitize(data)
                        request._body = json.dumps(sanitized).encode()
                    except (json.JSONDecodeError, Exception):
                        pass

        response: Response = await call_next(request)
        return response

    def _sanitize(self, data):
        if isinstance(data, str):
            return bleach.clean(data, tags=[], strip=True)
        elif isinstance(data, dict):
            return {k: self._sanitize(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize(item) for item in data]
        return data
