from fastapi import APIRouter, Query
from typing import Optional, List
from pydantic import BaseModel
from .response import ResponseModel, ErrorResponse, ErrorCode
from internal import task as tapi


router = APIRouter(
    prefix="/tasks",
    tags=["tasks"],
    dependencies=[],
)


class UpdateTaskBody(BaseModel):
    assignment_id: str
    status: tapi.AssignmentStatus
    log: str


class CreateTaskBody(BaseModel):
    drive_id: str
    ref_id: str
    path: str
    priority: int


@router.get("/", response_model=ResponseModel[list])
def get_tasks(
    status: Optional[List[tapi.TaskStatus]] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
):
    tasks = tapi.list_tasks(status, limit)
    return ResponseModel(data=tasks)


# TODO: this way of error handling is very bad, will fix after training begins
@router.post("/update")
def update_task(body: UpdateTaskBody):
    tapi.update_task(
        body.assignment_id,
        body.status,
        body.log
    )


@router.get("/create")
def create_task(body: CreateTaskBody):
    tapi.create_task(
        body.drive_id,
        body.ref_id,
        body.path,
        body.priority
    )


@router.get("/delete")
def delete_task(task_id: str):
    tapi.delete_task(task_id)


@router.post("/claim", response_model=ResponseModel[dict])
def claim_task(worker_id: str):
    task = tapi.claim_task(worker_id)
    if not task:
        ErrorResponse(
            message="No available tasks to claim",
            error_code=ErrorCode.TASK_NOT_AVAILABLE
        )
    return ResponseModel(data=task)
