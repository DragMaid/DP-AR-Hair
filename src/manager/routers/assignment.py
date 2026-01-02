from fastapi import APIRouter, Query
from pydantic import BaseModel
from .response import ResponseModel
from internal import assignment as aapi
from typing import Optional, List

router = APIRouter(
    prefix="/assignments",
    tags=["assignments"],
    dependencies=[],
)


class ReportAsignmentBody(BaseModel):
    assignment_id: str
    status: aapi.AssignmentStatus
    log: str


@router.get("/", response_model=ResponseModel[list])
def get_assignments(
    limit: int = Query(default=100, ge=1, le=1000)
):
    assignments = aapi.list_assignments(limit)
    return ResponseModel(data=assignments)


@router.get("/history", response_model=ResponseModel[list])
def get_assignment_history(
    status: Optional[List[aapi.AssignmentStatus]] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000)
):
    assignment_history = aapi.list_assignment_history(status, limit)
    return ResponseModel(data=assignment_history)


@router.post("/report")
def report_assignment(body: ReportAsignmentBody):
    aapi.report_assignment(
        body.assignment_id,
        body.status,
        body.log
    )
