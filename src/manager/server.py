from pydantic.generics import GenericModel
from pydantic import BaseModel
from typing import Generic, Optional, TypeVar, List
from fastapi import FastAPI
from manager.database import (
    list_tasks,
    list_workers,
    list_assignments,
    TaskStatus,
    AssignmentStatus
)
from fastapi import Query
from enum import Enum

app = FastAPI()
T = TypeVar("T")


class ErrorCode(str, Enum):
    TASK_NOT_AVAILABLE = "TASK_NOT_AVAILABLE"
    WORKER_NOT_AUTHORIZED = "WORKER_NOT_AUTHORIZED"
    INVALID_RESULT = "INVALID_RESULT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


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


@app.get("/tasks/", response_model=ResponseModel[list])
def get_tasks(
    status: Optional[List[TaskStatus]] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
):
    tasks = list_tasks(status, limit)
    return ResponseModel(data=tasks)


@app.get("/workers/", response_model=ResponseModel[list])
def get_workers(
    email: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000)
):
    workers = list_workers(email, limit)
    return ResponseModel(data=workers)


@app.get("/assignments/", response_model=ResponseModel[list])
def get_assignments(
    status: Optional[List[AssignmentStatus]] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000)
):
    assignments = list_assignments(status, limit)
    return ResponseModel(data=assignments)


@app.post("/tasks/claim", response_model=ResponseModel[dict])
def claim_task(worker_id: str):
    task = claim_task(worker_id)
    if not task:
        ErrorResponse(
            message="No available tasks to claim",
            error_code=ErrorCode.TASK_NOT_AVAILABLE
        )
    return ResponseModel(data=task)
