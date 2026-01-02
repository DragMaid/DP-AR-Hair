from fastapi import APIRouter, Query
from .response import ResponseModel
from internal.worker import list_workers, create_worker
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
    workers = list_workers(email, limit)
    return ResponseModel(data=workers)


@router.post("/create", response_model=ResponseModel[str])
def register_worker(email: str):
    password = create_worker(email)
    return ResponseModel(data=password)


@router.post("/delete")
def delete_worker(worker_id: str):
    pass


@router.post("/reset")
def reset_worker_account(worker_id: str):
    pass


@router.post("/authenticate")
def authenticate_worker(email: str, password: str):
    pass
