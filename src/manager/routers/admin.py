from fastapi import APIRouter, Query, Depends
from internal.auth import require_god, require_admin
from internal import user as uapi
from typing import Optional, List, Annotated
from pydantic import BaseModel
from schemas.user import User, UserRoles

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[],
)


class CreateAdminResponse(BaseModel):
    password: str


class ResetAdminResponse(BaseModel):
    password: str


@router.get("", response_model=List[User])
def get_admins(
    _: Annotated[None, Depends(require_admin)],
    username: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000)
):
    workers = uapi.list_users(username, limit, UserRoles.ADMIN)
    return [User(**w) for w in workers]


@router.post("/create", response_model=CreateAdminResponse)
def create_admin(
    _: Annotated[None, Depends(require_god)],
    username: str
):
    password = uapi.create_user(username, UserRoles.ADMIN)
    return CreateAdminResponse(password=password)


@router.post("/delete", status_code=204)
def delete_admin(
    _: Annotated[None, Depends(require_god)],
    admin_id: str
):
    uapi.remove_user(admin_id, UserRoles.ADMIN)


@router.post("/reset", response_model=ResetAdminResponse)
def reset_admin_password(
    _: Annotated[None, Depends(require_god)],
    admin_id: str
):
    password = uapi.reset_worker_password(admin_id, UserRoles.ADMIN)
    return ResetAdminResponse(password=password)
