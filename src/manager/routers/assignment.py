from fastapi import APIRouter, Query
from .response import ResponseModel
from internal.assignment import (
    list_assignments,
    list_assignment_history,
    AssignmentStatus
)
from typing import Optional, List

router = APIRouter(
    prefix="/assignments",
    tags=["assignments"],
    dependencies=[],
)


@router.get("/", response_model=ResponseModel[list])
def get_assignments(
    limit: int = Query(default=100, ge=1, le=1000)
):
    assignments = list_assignments(limit)
    return ResponseModel(data=assignments)


@router.get("/history", response_model=ResponseModel[list])
def get_assignment_history(
    status: Optional[List[AssignmentStatus]] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000)
):
    assignment_history = list_assignment_history(status, limit)
    return ResponseModel(data=assignment_history)
