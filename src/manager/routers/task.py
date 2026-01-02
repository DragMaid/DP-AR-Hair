from fastapi import APIRouter, Query
from typing import Optional, List
from .response import ResponseModel, ErrorResponse, ErrorCode
from internal.task import list_tasks, TaskStatus


router = APIRouter(
    prefix="/tasks",
    tags=["tasks"],
    dependencies=[],
)


@router.get("/", response_model=ResponseModel[list])
def get_tasks(
    status: Optional[List[TaskStatus]] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
):
    tasks = list_tasks(status, limit)
    return ResponseModel(data=tasks)


@router.get("/create")
def create_task():
    pass


@router.get("/delete")
def delete_task():
    pass


@router.post("/claim", response_model=ResponseModel[dict])
def claim_task(worker_id: str):
    task = claim_task(worker_id)
    if not task:
        ErrorResponse(
            message="No available tasks to claim",
            error_code=ErrorCode.TASK_NOT_AVAILABLE
        )
    return ResponseModel(data=task)
