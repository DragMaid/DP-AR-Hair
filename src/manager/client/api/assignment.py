from .fetcher import APIFetcher
from typing import List, Optional
from schemas.assignment import Assignment, AssignmentHistory, AssignmentStatus
from routers.assignment import ReportAssignmentBody, TerminateAssignmentBody


async def get_assignments(
    fetcher: APIFetcher,
    owner_id: Optional[str] = None,
    limit: int = 100
) -> List[Assignment]:
    params = {}

    if owner_id:
        params["owner_id"] = owner_id

    if limit:
        params["limit"] = limit

    assignments = await fetcher.fetch(
        method="GET",
        path="/assignments",
        params=params,
        require_auth=False,
        response_model=List[Assignment]
    )

    return assignments


async def get_assignment_history(
    fetcher: APIFetcher,
    owner_id: Optional[str] = None,
    status: Optional[List[AssignmentStatus]] = None,
    limit: int = 100
) -> List[AssignmentHistory]:
    params = {}

    if status:
        params["status"] = status

    if limit:
        params["limit"] = limit

    if owner_id:
        params["owner_id"] = owner_id

    histories = await fetcher.fetch(
        method="GET",
        path="/assignments/history",
        params=params,
        require_auth=False,
        response_model=List[AssignmentHistory]
    )

    return histories


async def report_assignment(
    fetcher: APIFetcher,
    assignment_id: str,
    upload_id: str,
    status: AssignmentStatus,
    log: str
) -> None:
    payload = ReportAssignmentBody(
        upload_id=upload_id,
        assignment_id=assignment_id,
        status=status,
        log=log
    )

    await fetcher.fetch(
        method="POST",
        path="/assignments/report",
        json=payload,
        require_auth=True,
    )


async def terminate_assignment(
    fetcher: APIFetcher,
    assignment_id: str,
    log: str
) -> None:
    payload = TerminateAssignmentBody(
        assignment_id=assignment_id,
        log=log
    )

    await fetcher.fetch(
        method="POST",
        path="/assignments/terminate",
        json=payload,
        require_auth=True,
    )
