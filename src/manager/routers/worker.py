from fastapi import APIRouter, Query, Depends
from typing import Optional, List, Annotated
from manager.internal import user as uapi
from manager.internal.auth import require_admin, require_worker_ownership
from manager.schemas.user import User, UserRoles
from manager.typings.backend import CreateWorkerResponse, ResetWorkerResponse

router = APIRouter(
    prefix="/workers",
    tags=["workers"],
    dependencies=[],
)


@router.get("", response_model=List[User])
def get_workers(
    email: Optional[str] = Query(default=None),
    owner_id: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000)
):
    workers = uapi.list_users(email, UserRoles.WORKER, owner_id, limit)
    return [User(**w) for w in workers]


@router.post("/create", response_model=CreateWorkerResponse)
def create_worker(
    email: str,
    admin: Annotated[User, Depends(require_admin)]
):
    password = uapi.create_user(email, UserRoles.WORKER, admin["id"])
    return CreateWorkerResponse(password=password)


@router.post("/delete", status_code=204)
def delete_worker(
    worker_id: str,
    _: Annotated[None, Depends(require_worker_ownership)]
):
    uapi.remove_user(worker_id, UserRoles.WORKER)


@router.post("/reset", response_model=ResetWorkerResponse)
def reset_worker_password(
    worker_id: str,
    _: Annotated[None, Depends(require_worker_ownership)]
):
    password = uapi.reset_worker_password(worker_id, UserRoles.WORKER)
    return ResetWorkerResponse(password=password)
