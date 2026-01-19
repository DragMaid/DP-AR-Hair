from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request
from cachetools import TTLCache
from manager.core.exceptions import AppError
from manager.core.jwt_manager import decode_access_token


class RateLimiter:
    """Fixed-window rate limiter using in-memory TTLCache."""

    def __init__(self, limit: int = 40, window: int = 60, maxsize: int = 10_000):
        self.limit = limit
        self.window = window
        self.cache = TTLCache(maxsize=maxsize, ttl=window)

    def _extract_identity(self, request: Request) -> str:
        """Resolve the best possible rate-limit key."""
        auth = request.headers.get("authorization")

        if auth and auth.startswith("Bearer "):
            token = auth[7:]
            try:
                payload = decode_access_token(token)
                return f"user:{payload}"
            except Exception:
                pass

        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"

        if request.client:
            return f"ip:{request.client.host}"

        return "ip:unknown"

    def check(self, request: Request) -> None:
        key = self._extract_identity(request)

        count = self.cache.get(key, 0)
        if count >= self.limit:
            raise AppError("TOO_MANY_REQUESTS")

        self.cache[key] = count + 1


class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limiter: RateLimiter):
        super().__init__(app)
        self.limiter = limiter

    async def dispatch(self, request: Request, call_next):
        try:
            self.limiter.check(request)
            return await call_next(request)
        except AppError as exc:
            response = JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "code": exc.code,
                        "message": exc.message,
                    }
                },
            )
            if exc.headers:
                response.headers.update(exc.headers)
            return response
