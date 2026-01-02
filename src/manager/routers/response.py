from enum import Enum
from typing import Generic, Optional, TypeVar
from pydantic import BaseModel
from pydantic.generics import GenericModel

T = TypeVar("T")


class ErrorCode(str, Enum):
    TASK_NOT_AVAILABLE = "TASK_NOT_AVAILABLE"
    WORKER_NOT_AUTHORIZED = "WORKER_NOT_AUTHORIZED"
    INVALID_RESULT = "INVALID_RESULT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    ROUTE_NOT_FOUND = "ROUTE_NOT_FOUND"


class ResponseModel(GenericModel, Generic[T]):
    success: bool = True
    data: Optional[T] = None
    message: Optional[str] = None
    error_code: Optional[str] = None


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    error_code: str
    data: Optional[dict] = None
