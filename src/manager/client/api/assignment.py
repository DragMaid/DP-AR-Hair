from .fetcher import APIFetcher
from typing import List, Optional
from schemas.assignment import Assignment, AssignmentHistory, AssignmentStatus
from routers.assignment import ReportAssignmentBody


async def get_assignments(
    fetcher: APIFetcher,
    limit: int = 100
) -> List[Assignment]:
    assignments = await fetcher.fetch(
        method="GET",
        path="/assignments",
        params={"limit": limit},
        require_auth=False,
        response_model=List[Assignment]
    )

    return assignments


async def get_assignment_history(
    fetcher: APIFetcher,
    status: Optional[List[AssignmentStatus]] = None,
    limit: int = 100
) -> List[AssignmentHistory]:
    params = {}

    if status:
        params["status"] = status

    if limit:
        params["limit"] = limit

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
    status: AssignmentStatus,
    log: str
) -> None:
    payload = ReportAssignmentBody(
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
