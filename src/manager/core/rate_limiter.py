from time import time
from fastapi import Request, Depends
from fastapi.responses import JSONResponse
from cachetools import TTLCache
from core.exceptions import AppError
from internal.auth import extract_bearer_token
from core.jwt_manager import decode_access_token
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimiter:
    """
    Simple sliding-window rate limiter using in-memory TTLCache.
    """

    def __init__(self, limit: int = 10, window: int = 60, maxsize: int = 10000):
        """
        :param limit: max requests per window
        :param window: window duration in seconds
        :param maxsize: max number of users to track
        """
        self.limit = limit
        self.window = window
        self.cache = TTLCache(maxsize=maxsize, ttl=window)

    def check(self, request: Request, token: str = Depends(extract_bearer_token)):
        """
        Check if a user/key is within rate limit. Raises HTTPException if exceeded.
        :param key: user ID, IP, or token
        """

        try:
            user_id = decode_access_token(token)
            key = user_id
        except Exception:
            key = request.client.host

        now = time()
        timestamps = self.cache.get(key, [])

        # Remove timestamps outside the window
        timestamps = [t for t in timestamps if t > now - self.window]

        if len(timestamps) >= self.limit:
            raise AppError("TOO_MANY_REQUESTS")

        # Add current timestamp
        timestamps.append(now)
        self.cache[key] = timestamps


class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limiter):
        super().__init__(app)
        self.limiter = limiter

    async def dispatch(self, request: Request, call_next):
        try:
            self.limiter.check(request)
            response = await call_next(request)
            return response
        except AppError as exc:
            # Manually create JSON response since starlette does not re-route
            response = JSONResponse(
                status_code=exc.status_code,
                content={"error": {"code": exc.code, "message": exc.message}}
            )
            if exc.headers:
                response.headers.update(exc.headers)
            return response
