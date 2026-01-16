from pathlib import Path
from uuid import uuid4
from fastapi import APIRouter, Query, Depends
from pydantic import BaseModel
from typing import Optional, List, Annotated
from manager.internal import assignment as aapi
from manager.internal.auth import require_worker, require_admin
from manager.internal.image import move_file
from manager.schemas.user import User
from manager.core.config import settings
from manager.schemas.assignment import (
    AssignmentHistory,
    Assignment,
    AssignmentStatus,
)
from manager.internal.assignment import UploadPathMap

router = APIRouter(
    prefix="/assignments",
    tags=["assignments"],
    dependencies=[],
)


class TerminateAssignmentBody(BaseModel):
    assignment_id: str
    log: str


class ReportAssignmentBody(TerminateAssignmentBody):
    generated_upload_id: Optional[str]
    driving_upload_id: Optional[str]
    reference_upload_id: Optional[str]
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
    upload_map: UploadPathMap | None = aapi.report_assignment(
        assignment_id=body.assignment_id,
        worker_id=worker["id"],
        driving_upload_id=body.driving_upload_id,
        reference_upload_id=body.reference_upload_id,
        generated_upload_id=body.generated_upload_id,
        status=body.status,
        log=body.log
    )

    if not upload_map:
        return

    file_id = uuid4()
    for key, path in upload_map.model_dump().items():
        extension = path.split('.')[-1]
        filename = f"{file_id}_{key}.{extension}"

        settings.GENERATED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        move_file(
            source=Path(path),
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
