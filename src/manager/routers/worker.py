from fastapi import APIRouter, Query
from .response import ResponseModel
from internal import worker as wapi
from typing import Optional

router = APIRouter(
    prefix="/workers",
    tags=["workers"],
    dependencies=[],
)


@router.get("/", response_model=ResponseModel[list])
def get_workers(
    email: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000)
):
    workers = wapi.list_workers(email, limit)
    return ResponseModel(data=workers)


@router.post("/create", response_model=ResponseModel[str])
def create_worker(email: str):
    password = wapi.create_worker(email)
    return ResponseModel(data=password)


@router.post("/delete")
def delete_worker(worker_id: str):
    wapi.remove_worker(worker_id)


@router.post("/reset")
def reset_worker_password(worker_id: str):
    password = wapi.reset_worker_password(worker_id)
    return ResponseModel(data=password)


@router.post("/authenticate")
def authenticate_worker(email: str, password: str):
    pass
