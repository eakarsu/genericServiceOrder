"""Bounded native provider transport. Never infer cancellation from request-body completion."""
import asyncio
import time
from fastapi import HTTPException

_requests = {}

def claim(owner, limit=20, window=3600):
    now = time.monotonic()
    for key in [key for key, row in _requests.items() if row[1] <= now]:
        del _requests[key]
    count, expires = _requests.get(str(owner), (0, now + window))
    if count >= limit:
        raise HTTPException(429, "Hourly AI request limit reached", headers={"Retry-After": str(max(1, int(expires-now)))})
    if str(owner) not in _requests and len(_requests) >= 10000:
        raise HTTPException(503, "AI request capacity is temporarily unavailable")
    _requests[str(owner)] = (count + 1, expires)

async def connected(request):
    # Some ASGI receive implementations wait for the next event after the body
    # has been consumed. Bound the probe so it cannot stall provider execution.
    try:
        disconnected = await asyncio.wait_for(request.is_disconnected(), timeout=0.05)
    except TimeoutError:
        disconnected = False
    if disconnected:
        raise HTTPException(499, "AI request cancelled")

async def run_native(request, owner, operation, timeout=60):
    claim(owner)
    await connected(request)
    provider = asyncio.create_task(operation())
    async def watch_disconnect():
        while True:
            message = await request.receive()
            if message['type'] == 'http.disconnect':
                raise HTTPException(499, 'AI request cancelled')
    watcher = asyncio.create_task(watch_disconnect())
    try:
        done, _ = await asyncio.wait({provider, watcher}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if watcher in done:
            await watcher
        if provider not in done:
            raise HTTPException(504, 'OpenRouter request timed out')
        result = await provider
        await connected(request)
        return result
    finally:
        for pending in (provider, watcher):
            pending.cancel()
        await asyncio.gather(provider, watcher, return_exceptions=True)
