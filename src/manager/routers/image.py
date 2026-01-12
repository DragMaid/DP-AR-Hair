from fastapi import APIRouter, UploadFile, Depends
from pydantic import BaseModel
from typing import Annotated
from schemas.user import User
from schemas.image import UploadStatus
from internal.auth import require_worker
from internal.image import validate_image, insert_upload, save_upload_file
from internal.assignment import require_assignment_ownership
from uuid import uuid4


router = APIRouter(
    prefix="/images",
    tags=["images"],
    dependencies=[],
)


class UploadResponse(BaseModel):
    upload_id: str


@router.post("/upload", response_model=UploadResponse)
def upload_image(
    worker: Annotated[User, Depends(require_worker)],
    assignment_id: str,
    file: UploadFile
):
    require_assignment_ownership(
        assignment_id=assignment_id,
        worker_id=worker["id"],
    )

    validate_image(file)

    uuid = str(uuid4())
    destination = save_upload_file(
        upload_file=file,
        name=uuid
    )

    insert_upload(
        id=uuid,
        worker_id=worker["id"],
        assignment_id=assignment_id,
        file_path=destination,
        status=UploadStatus.PENDING
    )

    return UploadResponse(upload_id=uuid)
