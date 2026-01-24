from fastapi import APIRouter, UploadFile, Depends
from typing import Annotated
from uuid import uuid4
from manager.schemas.user import User
from manager.schemas.image import UploadStatus, ImageCategories
from manager.internal.auth import require_worker
from manager.internal.image import validate_image, insert_upload, save_upload_file
from manager.internal.auth import require_assignment_ownership
from manager.typings.backend import UploadResponse


router = APIRouter(
    prefix="/images",
    tags=["images"],
    dependencies=[],
)


@router.post("/upload", response_model=UploadResponse)
def upload_image(
    worker: Annotated[User, Depends(require_worker)],
    assignment_id: str,
    category: ImageCategories,
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
        category=category,
        status=UploadStatus.PENDING
    )

    return UploadResponse(upload_id=uuid)
