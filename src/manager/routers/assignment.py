from fastapi import APIRouter, Query, Depends
from pydantic import BaseModel
from internal import assignment as aapi
from typing import Optional, List, Annotated
from internal.auth import require_worker, require_admin
from internal.image import move_file, get_generated_name
from schemas.user import User
from core.config import settings
from pathlib import Path
from schemas.assignment import (
    AssignmentHistory,
    Assignment,
    AssignmentStatus
)

router = APIRouter(
    prefix="/assignments",
    tags=["assignments"],
    dependencies=[],
)


class TerminateAssignmentBody(BaseModel):
    assignment_id: str
    log: str


class ReportAssignmentBody(TerminateAssignmentBody):
    upload_id: str
    status: AssignmentStatus


@router.get("", response_model=List[Assignment])
def get_assignments(
    owner_id: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000)
):
    assignments = aapi.list_assignments(owner_id, limit)
    return [Assignment(**a) for a in assignments]


@router.get("/history", response_model=List[AssignmentHistory])
def get_assignment_history(
    status: Optional[List[aapi.AssignmentStatus]] = Query(default=None),
    owner_id: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000)
):
    assignment_histories = aapi.list_assignment_history(
        status, owner_id, limit)
    return [AssignmentHistory(**ah) for ah in assignment_histories]


@router.post("/report", status_code=204)
def report_assignment(
    body: ReportAssignmentBody,
    worker: Annotated[User, Depends(require_worker)]
):
    filename = get_generated_name(body.assignment_id)

    file_path = aapi.report_assignment(
        body.assignment_id,
        worker["id"],
        body.upload_id,
        body.status,
        body.log
    )

    extension = file_path.split('.')[-1]
    filename = f"{filename}.{extension}"

    move_file(
        source=Path(file_path),
        destination=Path(settings.GENERATED_IMAGE_DIR, filename)
    )


@router.post("/terminate", status_code=204)
def terminate_assignment(
    body: TerminateAssignmentBody,
    admin: Annotated[User, Depends(require_admin)]
):
    aapi.terminate_assignment(
        body.assignment_id,
        admin["id"],
        body.log
    )
