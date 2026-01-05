from .logger import get_logger
from functools import wraps
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi import FastAPI

logger = get_logger(__name__)

ERRORS = {
    # Client / Request errors
    "INVALID_REQUEST": {
        "message": "Invalid request parameters",
        "status_code": 400
    },
    "UNAUTHORIZED": {
        "message": "Unauthorized access",
        "status_code": 401
    },
    "FORBIDDEN": {
        "message": "Not authorized to perform this action",
        "status_code": 403
    },
    "TOO_MANY_REQUESTS": {
        "message": "Too many requests",
        "status_code": 429
    },

    # Assignment errors
    "ASSIGNMENT_REPORT_FAILED": {
        "message": "Assignment failed",
        "status_code": 500
    },
    "ASSIGNMENT_NOT_FOUND": {
        "message": "Assignment not found",
        "status_code": 404
    },
    "ASSIGNMENT_CREATION_FAILED": {
        "message": "Assignment creation failed",
        "status_code": 500
    },

    # Database / Storage errors
    "DB_CONNECTION_FAILED": {
        "message": "Database connection failed",
        "status_code": 500
    },
    "DB_QUERY_FAILED": {
        "message": "Database query failed",
        "status_code": 500
    },
    "DB_CONFLICT": {
        "message": "Database conflict occurred",
        "status_code": 409
    },

    # Worker / Node errors
    "WORKER_TIMEOUT": {
        "message": "Worker timed out",
        "status_code": 504
    },
    "WORKER_NOT_FOUND": {
        "message": "Worker not found",
        "status_code": 404
    },
    "WORKER_CREATION_FAILED": {
        "message": "Failed to create worker",
        "status_code": 500
    },
    "WORKER_INTERNAL_ERROR": {
        "message": "Worker operation failed",
        "status_code": 500
    },

    # Task errors
    "TASK_NOT_FOUND": {
        "message": "Task not found",
        "status_code": 404
    },
    "TASK_INTERNAL_ERROR": {
        "message": "Task operation failed",
        "status_code": 500
    },
    "TASK_CREATION_FAILED": {
        "message": "Failed to create task",
        "status_code": 500
    },

    # Queue errors
    "QUEUE_OVERFLOW": {
        "message": "Queue is full",
        "status_code": 503
    },
    "QUEUE_EMPTY": {
        "message": "Queue is empty",
        "status_code": 404
    },

    # Token errors
    "TOKEN_CREATION_FAILED": {
        "message": "Token creation failed",
        "status_code": 500
    },
    "INVALID_TOKEN": {
        "message": "Could not validate token",
        "status_code": 401,
        "headers": {"WWW-Authenticate": "Bearer"}
    },
    "TOKEN_EXPIRED": {
        "message": "Issued token expired",
        "status_code": 401,
        "headers": {"WWW-Authenticate": "Bearer"}
    },
    "TOKEN_INTERNAL_ERROR": {
        "message": "Token operation failed",
        "status_code": 500,
    },

    # Authentication errors
    "USER_NOT_FOUND": {
        "message": "User not found",
        "status_code": 404
    },
    "AUTH_INTERNAL_ERROR": {
        "message": "Authentication process failed",
        "status_code": 500
    },
    "INVALID_CREDENTIALS": {
        "message": "Could not validate credentials",
        "status_code": 401,
        "headers": {"WWW-Authenticate": "Bearer"}
    },
    "MISSING_AUTH_HEADER": {
        "message": "Missing authentication header",
        "status_code": 400,
        "headers": {"WWW-Authenticate": "Bearer"}
    },

    # External Service errors
    "EXTERNAL_API_FAIL": {
        "message": "External service failure",
        "status_code": 502
    },

    # Fallback / Unknown errors
    "UNKNOWN_ERROR": {
        "message": "An unexpected error occurred",
        "status_code": 500
    },
}

# TODO: why are we using message instead of details like InternalServerError ?


class AppError(Exception):
    def __init__(self, code: str):
        if code not in ERRORS:
            code = "UNKNOWN_ERROR"  # fallback
        self.code = code
        self.message = ERRORS[code]["message"]
        self.status_code = ERRORS[code]["status_code"]
        self.headers = ERRORS[code].get("headers")


def register_app_error_handler(app: FastAPI):
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        logger.error(f"[{exc.code}] {exc.message}", exc_info=True)
        response = JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}}
        )

        if exc.headers:
            response.headers.update(exc.headers)

        return response


def wrap_errors(default_code="UNKNOWN_ERROR"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AppError:
                raise
            except Exception as e:
                logger.error(
                    f"Exception in {func.__name__}: {e}", exc_info=True)
                raise AppError(default_code)
        return wrapper
    return decorator
