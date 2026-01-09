from .fetcher import APIFetcher
from typing import List, Optional
from schemas.user import User
from routers.admin import ResetAdminResponse, CreateAdminResponse


async def get_admins(
    fetcher: APIFetcher,
    username: Optional[str] = None,
    limit: int = 100
) -> List[User]:
    params = {}

    if username:
        params["username"] = username

    if limit:
        params["limit"] = limit

    admins = await fetcher.fetch(
        method="GET",
        path="/admin",
        params=params,
        require_auth=True,
        response_model=List[User]
    )

    return admins


async def create_admin(
    fetcher: APIFetcher,
    username: str
) -> CreateAdminResponse:
    res = await fetcher.fetch(
        method="POST",
        path="/admin/create",
        params={"username": username},
        require_auth=True,
        response_model=CreateAdminResponse
    )

    return res["password"]


async def delete_admin(
    fetcher: APIFetcher,
    admin_id: str
) -> None:
    await fetcher.fetch(
        method="POST",
        path="/admin/delete",
        params={"admin_id": admin_id},
        require_auth=True,
    )


async def reset_admin_password(
    fetcher: APIFetcher,
    admin_id: str
) -> str:
    res = fetcher.fetch(
        method="POST",
        path="/admin/reset",
        params={"admin_id": admin_id},
        require_auth=True,
        response_model=ResetAdminResponse
    )
    return res["password"]
