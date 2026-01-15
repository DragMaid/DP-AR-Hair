from .fetcher import APIFetcher
from typing import List, Optional
from manager.schemas.user import User
from manager.routers.worker import ResetWorkerResponse, CreateWorkerResponse


async def get_workers(
    fetcher: APIFetcher,
    email: Optional[str] = None,
    owner_id: Optional[str] = None,
    limit: int = 100
) -> List[User]:
    params = {}

    if email:
        params["email"] = email

    if limit:
        params["limit"] = limit

    if owner_id:
        params["owner_id"] = owner_id

    workers = await fetcher.fetch(
        method="GET",
        path="/workers",
        params=params,
        require_auth=False,
        response_model=List[User]
    )

    return workers


async def create_worker(
    fetcher: APIFetcher,
    email: str
) -> CreateWorkerResponse:
    res = await fetcher.fetch(
        method="POST",
        path="/workers/create",
        params={"email": email},
        require_auth=True,
        response_model=CreateWorkerResponse
    )

    return res["password"]


async def delete_worker(
    fetcher: APIFetcher,
    worker_id: str
) -> None:
    await fetcher.fetch(
        method="POST",
        path="/workers/delete",
        params={"worker_id": worker_id},
        require_auth=True,
    )


async def reset_worker_password(
    fetcher: APIFetcher,
    worker_id: str
) -> str:
    res = fetcher.fetch(
        method="POST",
        path="/workers/reset",
        params={"worker_id": worker_id},
        require_auth=True,
        response_model=ResetWorkerResponse
    )
    return res["password"]
