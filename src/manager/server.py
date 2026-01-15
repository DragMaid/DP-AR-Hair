from fastapi import FastAPI
from manager.routers import (
    assignment,
    worker,
    task,
    auth,
    admin,
    image
)
from manager.core.exceptions import (
    register_app_error_handler,
    register_http_error_handler,
    register_request_error_handler,
    register_response_error_handler,
    register_fallback_error_handler
)
from manager.core.rate_limiter import RateLimiter, RateLimiterMiddleware
from manager.core.config import settings

app = FastAPI(root_path="/api")
rate_limiter = RateLimiter(
    limit=settings.RATE_LIMITER_LIMIT,
    window=settings.RATE_LIMITER_WINDOW_SEC,
    maxsize=settings.RATE_LIMITER_CAPACITY
)

# Include all the routes
app.include_router(worker.router)
app.include_router(task.router)
app.include_router(assignment.router)
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(image.router)

# Middleware for rate limiting
app.add_middleware(RateLimiterMiddleware, limiter=rate_limiter)

# Error handlers for all error types
register_app_error_handler(app)
register_http_error_handler(app)
register_request_error_handler(app)
register_response_error_handler(app)
register_fallback_error_handler(app)

# TODO: I can make nginx only accept X-accel redirect from fastapi if image is included in assignment table


@app.get("/health")
async def health():
    return {"message": "Server is up and running!"}

# TODO: make it so everything can be ran from root (or probably just the client)
