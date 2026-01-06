from fastapi import APIRouter, Query, Depends
from pydantic import BaseModel
from internal import assignment as aapi
from typing import Optional, List, Annotated
from internal.auth import require_worker
from schemas.user import User
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


class ReportAssignmentBody(BaseModel):
    assignment_id: str
    status: AssignmentStatus
    log: str


@router.get("", response_model=List[Assignment])
def get_assignments(
    limit: int = Query(default=100, ge=1, le=1000)
):
    assignments = aapi.list_assignments(limit)
    return [Assignment(**a) for a in assignments]


@router.get("/history", response_model=List[AssignmentHistory])
def get_assignment_history(
    status: Optional[List[aapi.AssignmentStatus]] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000)
):
    assignment_histories = aapi.list_assignment_history(status, limit)
    return [AssignmentHistory(**ah) for ah in assignment_histories]


@router.post("/report", status_code=204)
def report_assignment(
    body: ReportAssignmentBody,
    worker: Annotated[User, Depends(require_worker)]
 ):
    aapi.report_assignment(
        body.assignment_id,
        worker["id"],
        body.status,
        body.log
    )
