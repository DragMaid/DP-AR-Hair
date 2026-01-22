from fastapi import APIRouter, Query, Depends
from typing import Optional, List, Annotated
from manager.internal import task as tapi
from manager.internal.auth import require_admin, require_worker
from manager.schemas.task import Task
from manager.schemas.user import User
from manager.typings.backend import (
    CreateTaskResponse,
    CreateTaskBody,
    ClaimTaskResponse,
    ProgressResponse
)


router = APIRouter(
    prefix="/tasks",
    tags=["tasks"],
    dependencies=[],
)


@router.get("", response_model=List[Task])
def get_tasks(
    status: Optional[List[tapi.TaskStatus]] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
):
    tasks = tapi.list_tasks(status, limit)
    return [Task(**t) for t in tasks]


@router.get("/progress", response_model=ProgressResponse)
def get_progress():
    progress = tapi.get_progress()
    return ProgressResponse(
        done=progress["completed_count"],
        total=progress["total_count"]
    )


@router.post("/create", response_model=CreateTaskResponse)
def create_task(
    body: CreateTaskBody,
    _: Annotated[User, Depends(require_admin)]
):
    task_id = tapi.create_task(
        body.driving_id,
        body.reference_id,
        body.path,
        body.priority
    )
    return CreateTaskResponse(task_id=task_id)


@router.post("/delete", status_code=204)
def delete_task(
    task_id: str,
    _: Annotated[User, Depends(require_admin)]
):
    tapi.delete_task(task_id)


# TODO: add a limit to the number of tasks you can receive
@router.post("/claim", response_model=ClaimTaskResponse)
def claim_task(
    worker: Annotated[User, Depends(require_worker)]
):
    response = tapi.claim_task(worker["id"])

    # TODO: return the old assignment to the worker if not finished
    # TODO: maybe returning image id and mapping that id to nginx path would be better
    return ClaimTaskResponse(
        assignment_id=response["assignment_id"],
        driving_path=response["driving_path"],
        reference_path=response["reference_path"],
    )
