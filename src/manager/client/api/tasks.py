from .fetcher import APIFetcher
from typing import List, Optional
from manager.schemas.task import Task, TaskStatus
from manager.routers.task import CreateTaskBody, CreateTaskResponse, ClaimTaskResponse


async def get_tasks(
    fetcher: APIFetcher,
    status: Optional[List[TaskStatus]] = None,
    limit: int = 100
) -> List[Task]:
    params = {}

    if status:
        params["status"] = status

    if limit:
        params["limit"] = limit

    tasks = await fetcher.fetch(
        method="GET",
        path="/tasks",
        params=params,
        require_auth=False,
        response_model=List[Task]
    )
    return tasks


async def create_task(
    fetcher: APIFetcher,
    driving_id: str,
    reference_id: str,
    path: str,
    priority: int
) -> str:
    payload = CreateTaskBody(
        driving_id=driving_id,
        reference_id=reference_id,
        path=path,
        priority=priority
    )

    res = await fetcher.fetch(
        method="POST",
        path="/tasks/create",
        json=payload,
        require_auth=True,
        response_model=CreateTaskResponse
    )

    return res["task_id"]


async def delete_task(
    fetcher: APIFetcher,
    task_id: str
) -> None:
    await fetcher.fetch(
        method="POST",
        path="/tasks/delete",
        params={"task_id": task_id},
        require_auth=True
    )


async def claim_task(
    fetcher: APIFetcher,
) -> ClaimTaskResponse:
    res = await fetcher.fetch(
        method="POST",
        path="/tasks/claim",
        require_auth=True,
        response_model=ClaimTaskResponse,
    )
    return res
