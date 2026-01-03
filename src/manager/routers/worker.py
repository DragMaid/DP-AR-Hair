from fastapi import APIRouter, Query, Depends
from internal import worker as wapi
from typing import Optional, List, Annotated
from internal.auth import require_admin
from schemas.user import User
from pydantic import BaseModel

router = APIRouter(
    prefix="/workers",
    tags=["workers"],
    dependencies=[],
)


class CreateWorkerResponse(BaseModel):
    password: str


class ResetWorkerResponse(BaseModel):
    password: str


@router.get("/", response_model=List[User])
def get_workers(
    email: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000)
):
    workers = wapi.list_workers(email, limit)
    return [User(**w) for w in workers]


@router.post("/create", response_model=CreateWorkerResponse)
def create_worker(
    email: str,
    _: Annotated[None, Depends(require_admin)]
):
    password = wapi.create_worker(email)
    return CreateWorkerResponse(password=password)


@router.post("/delete", status_code=204)
def delete_worker(
    worker_id: str,
    _: Annotated[None, Depends(require_admin)]
):
    wapi.remove_worker(worker_id)


@router.post("/reset", response_model=ResetWorkerResponse)
def reset_worker_password(
    worker_id: str,
    _: Annotated[None, Depends(require_admin)]
):
    password = wapi.reset_worker_password(worker_id)
    return ResetWorkerResponse(password=password)
