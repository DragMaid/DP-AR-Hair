from fastapi import APIRouter, Query, Depends
from typing import Optional, List, Annotated
from pydantic import BaseModel
from internal import task as tapi
from internal.auth import require_admin, require_worker
from schemas.task import Task


router = APIRouter(
    prefix="/tasks",
    tags=["tasks"],
    dependencies=[],
)


class CreateTaskBody(BaseModel):
    driving_id: str
    reference_id: str
    path: str
    priority: int


class CreateTaskResponse(BaseModel):
    task_id: str


class ClaimTaskResponse(BaseModel):
    assignment_id: str


@router.get("", response_model=List[Task])
def get_tasks(
    status: Optional[List[tapi.TaskStatus]] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
):
    tasks = tapi.list_tasks(status, limit)
    return [Task(**t) for t in tasks]


@router.post("/create", response_model=CreateTaskResponse)
def create_task(
    body: CreateTaskBody,
    _: Annotated[None, Depends(require_admin)]
):
    task_id = tapi.create_task(
        body.drive_id,
        body.ref_id,
        body.path,
        body.priority
    )
    return CreateTaskResponse(task_id=task_id)


@router.post("/delete", status_code=204)
def delete_task(
    task_id: str,
    _: Annotated[None, Depends(require_admin)]
):
    tapi.delete_task(task_id)


@router.post("/claim", response_model=ClaimTaskResponse)
def claim_task(
    worker_id: str,
    _: Annotated[None, Depends(require_worker)]
):
    assignment_id = tapi.claim_task(worker_id)
    return ClaimTaskResponse(assignment_id=assignment_id)
