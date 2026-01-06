from pydantic import BaseModel
from enum import Enum
from typing import Optional


class ErrorCategories(str, Enum):
    UNAVAILABLE = "UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMITED = "RATE_LIMITED"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    SERVER_ERROR = "SERVER_ERROR"
    CLIENT_ERROR = "CLIENT_ERROR"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


class ErrorSources(str, Enum):
    NETWORK = "NETWORK"
    TRANSPORT = "TRANSPORT"
    BACKEND = "BACKEND"
    PROTOCOL = "PROTOCOL"
    CLIENT = "CLIENT"


class FrontErrorSchema(BaseModel):
    source: ErrorSources
    category: ErrorCategories
    retryable: bool
    message: str


class BackendError(BaseModel):
    code: str
    message: str
    status_code: int


class FrontError(Exception):
    # TODO: fix this way of getting params
    def __init__(self,
                 code: Optional[str] = None,
                 error: Optional[BackendError] = None):
        mapping = FRONT_ERRORS[code] if code \
            else BACK_TO_FRONT_ERRORS[error["code"]]
        self.source = mapping["source"]
        self.category = mapping["category"]
        self.retryable = mapping["retryable"]
        self.message = mapping["message"] if code else error["message"]


FRONT_ERRORS = {
    "SERVICE_UNAVAILABLE": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.UNAVAILABLE,
        "retryable": True,
        "message": "Requested service is unavailable",
    },
    "REQUEST_TIMEOUT": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.TIMEOUT,
        "retryable": True,
        "message": "Request timed out",
    },
    "INVALID_RESPONSE": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.INVALID_RESPONSE,
        "retryable": False,
        "message": "Invalid JSON response",
    },
    "INVALID_ERROR_RESPONSE": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.INVALID_RESPONSE,
        "retryable": False,
        "message": "Malformed error response",
    },
    "MISSING_ERROR_RESPONSE": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.INVALID_RESPONSE,
        "retryable": False,
        "message": "Missing error payload",
    },
}


BACK_TO_FRONT_ERRORS = {
    # Client / Request errors
    "INVALID_REQUEST": {
        "source": ErrorSources.CLIENT,
        "category": ErrorCategories.CLIENT_ERROR,
        "retryable": False,
    },
    "UNAUTHORIZED": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.UNAUTHORIZED,
        "retryable": False,
    },
    "FORBIDDEN": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.FORBIDDEN,
        "retryable": False,
    },
    "TOO_MANY_REQUESTS": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.RATE_LIMITED,
        "retryable": True,
    },

    # Assignment errors
    "ASSIGNMENT_REPORT_FAILED": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.SERVER_ERROR,
        "retryable": True,
    },
    "ASSIGNMENT_NOT_FOUND": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.CLIENT_ERROR,
        "retryable": False,
    },
    "ASSIGNMENT_CREATION_FAILED": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.SERVER_ERROR,
        "retryable": True,
    },

    # Database / Storage errors
    "DB_CONNECTION_FAILED": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.UNAVAILABLE,
        "retryable": True,
    },
    "DB_QUERY_FAILED": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.SERVER_ERROR,
        "retryable": True,
    },
    "DB_CONFLICT": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.CLIENT_ERROR,
        "retryable": False,
    },

    # Worker / Node Errors
    "WORKER_TIMEOUT": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.TIMEOUT,
        "retryable": True,
    },
    "WORKER_NOT_FOUND": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.CLIENT_ERROR,
        "retryable": False,
    },
    "WORKER_CREATION_FAILED": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.SERVER_ERROR,
        "retryable": True,
    },
    "WORKER_INTERNAL_ERROR": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.SERVER_ERROR,
        "retryable": True,
    },

    # Task errors
    "TASK_NOT_FOUND": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.CLIENT_ERROR,
        "retryable": False,
    },
    "TASK_INTERNAL_ERROR": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.SERVER_ERROR,
        "retryable": True,
    },
    "TASK_CREATION_FAILED": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.SERVER_ERROR,
        "retryable": True,
    },

    # Queue errors
    "QUEUE_OVERFLOW": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.UNAVAILABLE,
        "retryable": True,
    },
    "QUEUE_EMPTY": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.CLIENT_ERROR,
        "retryable": False,
    },

    # Token errors
    "TOKEN_CREATION_FAILED": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.SERVER_ERROR,
        "retryable": True,
    },
    "INVALID_TOKEN": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.UNAUTHORIZED,
        "retryable": False,
    },
    "TOKEN_EXPIRED": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.UNAUTHORIZED,
        "retryable": False,
    },
    "TOKEN_INTERNAL_ERROR": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.SERVER_ERROR,
        "retryable": True,
    },

    # Authentication errors
    "USER_NOT_FOUND": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.CLIENT_ERROR,
        "retryable": False,
    },
    "AUTH_INTERNAL_ERROR": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.SERVER_ERROR,
        "retryable": True,
    },
    "INVALID_CREDENTIALS": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.UNAUTHORIZED,
        "retryable": False,
    },
    "MISSING_AUTH_HEADER": {
        "source": ErrorSources.CLIENT,
        "category": ErrorCategories.CLIENT_ERROR,
        "retryable": False,
    },

    # External Service errors
    "EXTERNAL_API_FAIL": {
        "source": ErrorSources.NETWORK,
        "category": ErrorCategories.UNAVAILABLE,
        "retryable": True,
    },

    # Fallback / Unknown errors
    "UNKNOWN_ERROR": {
        "source": ErrorSources.BACKEND,
        "category": ErrorCategories.UNKNOWN,
        "retryable": False,
    },
}
