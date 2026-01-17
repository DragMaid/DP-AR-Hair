from fastapi import APIRouter, Query, Depends
from typing import Optional, List, Annotated
from manager.internal.auth import require_god, require_admin
from manager.internal import user as uapi
from manager.schemas.user import User, UserRoles
from manger.typings.backend import CreateAdminResponse, ResetAdminResponse

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[],
)


@router.get("", response_model=List[User])
def get_admins(
    _: Annotated[User, Depends(require_admin)],
    username: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000)
):
    # There will only be one god account so finding owner is not necessary
    admins = uapi.list_users(username, UserRoles.ADMIN, None, limit)
    return [User(**a) for a in admins]


@router.post("/create", response_model=CreateAdminResponse)
def create_admin(
    god: Annotated[User, Depends(require_god)],
    username: str
):
    password = uapi.create_user(username, UserRoles.ADMIN, god["id"])
    return CreateAdminResponse(password=password)


@router.post("/delete", status_code=204)
def delete_admin(
    _: Annotated[User, Depends(require_god)],
    admin_id: str
):
    uapi.remove_user(admin_id, UserRoles.ADMIN)


@router.post("/reset", response_model=ResetAdminResponse)
def reset_admin_password(
    _: Annotated[User, Depends(require_god)],
    admin_id: str
):
    password = uapi.reset_worker_password(admin_id, UserRoles.ADMIN)
    return ResetAdminResponse(password=password)
