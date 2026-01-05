from fastapi import FastAPI
from routers import (
    assignment,
    worker,
    task,
    auth
)
from core.exceptions import register_app_error_handler
from core.rate_limiter import RateLimiter, RateLimiterMiddleware
from core.config import settings

app = FastAPI()
rate_limiter = RateLimiter(
    limit=settings.RATE_LIMITER_LIMIT,
    window=settings.RATE_LIMITER_WINDOW_SEC,
    maxsize=settings.RATE_LIMITER_CAPACITY
)

app.include_router(worker.router)
app.include_router(task.router)
app.include_router(assignment.router)
app.include_router(auth.router)

app.add_middleware(RateLimiterMiddleware, limiter=rate_limiter)

# TODO: what about other errors, they do not fit the schema
register_app_error_handler(app)


@app.get("/health")
async def health():
    return {"message": "Server is up and running!"}
