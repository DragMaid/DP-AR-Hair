import tempfile
import mimetypes
from fastapi import UploadFile
from typing import Optional, List
from PIL import Image as PILImage, UnidentifiedImageError
from shutil import copyfileobj, move
from pathlib import Path
from .connect import get_cursor
from manager.schemas.image import Image, ImageCategories, UploadStatus
from manager.core.exceptions import AppError, wrap_errors
from manager.core.config import settings

# TODO: map image errors to frontend
UPLOAD_DIR = Path(settings.IMAGE_TMP_FOLDER)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@wrap_errors(default_code="IMAGE_INTERNAL_ERROR")
def move_file(source: Path, destination: Path):
    if not source.exists():
        raise AppError("UPLOAD_NOT_FOUND")
    move(source, destination)


@wrap_errors(default_code="IMAGE_INTERNAL_ERROR")
def save_upload_file(
    upload_file: UploadFile,
    name: str
) -> str:
    upload_dir = Path(UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    if upload_file.content_type not in ["image/jpg", "image/png"]:
        raise ValueError("Unsupported or unknown file type")

    ext = str(upload_file.content_type).split('/')[-1]
    final_path = upload_dir / f"{name}.{ext}"

    # Start from start of file
    upload_file.file.seek(0)

    # Write atomically via temp file
    with tempfile.NamedTemporaryFile(
        dir=upload_dir,
        delete=False
    ) as tmp:
        copyfileobj(upload_file.file, tmp)
        tmp_path = Path(tmp.name)

    # TODO: maybe map these to errors
    if tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        raise ValueError("Uploaded file is empty")

    tmp_path.replace(final_path)
    upload_file.file.close()

    return str(final_path)


@wrap_errors(default_code="IMAGE_INTERNAL_ERROR")
def list_images(
    image_type: Optional[List[ImageCategories]],
    limit: int = 100
) -> List[Image]:
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            SELECT
                id,
                file_path,
                type,
                created_at
            FROM images
            Where (
                %s IS NULL
                OR type = ANY(%s::image_categories)
            )
            ORDER BY created_at ASC
            LIMIT %s
        """, (image_type, limit,))
        return cur.fetchall()


@wrap_errors(default_code="IMAGE_INTERNAL_ERROR")
def insert_image(
    file_path: str,
    image_type: ImageCategories,
    host: Optional[str] = None,
) -> str:
    with get_cursor(dict_cursor=True, host=host) as cur:
        cur.execute("""
            INSERT INTO images (file_path, category)
            VALUES (%s, %s)
            ON CONFLICT (file_path)
            DO UPDATE SET file_path = %s
            RETURNING id
        """, (file_path, image_type, file_path))
        image_id = cur.fetchone()
        if not image_id:
            raise AppError("IMAGE_CREATION_FAILED")
        return image_id["id"]


@wrap_errors(default_code="IMAGE_UPLOAD_FAILED")
def insert_upload(
    id: str,
    worker_id: str,
    assignment_id: str,
    file_path: str,
    category: ImageCategories,
    status: UploadStatus,
) -> str:
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            INSERT INTO uploads (
                id,
                worker_id,
                assignment_id,
                file_path,
                category,
                status,
                expires_at
            )
            VALUES (
                %s, %s, %s, %s, %s::image_categories, %s,
                NOW() + %s * INTERVAL '1 minute')
            RETURNING id;
        """, (
            id,
            worker_id,
            assignment_id,
            file_path,
            category,
            status,
            settings.UPLOAD_TIMEOUT_MIN,
        ))
        upload_id = cur.fetchone()
        if not upload_id:
            raise AppError("UPLOAD_REGISTRATION_FAILED")
        return upload_id["id"]


@wrap_errors(default_code="IMAGE_UPLOAD_FAILED")
def validate_image(file: UploadFile) -> None:
    if file.content_type not in ["image/png", "image/jpg"]:
        raise AppError("UNSUPPORTED_IMAGE_TYPE")

    try:
        PILImage.open(file.file)
    except UnidentifiedImageError:
        raise AppError("INVALID_IMAGE_CONTENT")


@wrap_errors(default_code="IMAGE_INTERNAL_ERROR")
def retrieve_upload(
    upload_id: str,
    assignment_id: str,
    worker_id: str,
    category: Optional[List[ImageCategories]] = None
) -> str:

    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            UPDATE uploads
            SET status = 'processed'::upload_status
            WHERE id = %s
                AND worker_id = %s
                AND assignment_id = %s
                AND status = 'pending'::upload_status
                AND (category IS NULL OR
                     category = ANY(%s::image_categories[]))
            RETURNING file_path
        """, (upload_id, worker_id, assignment_id, category,))
        upload = cur.fetchone()
        if not upload:
            raise AppError("UPLOAD_NOT_FOUND")
        return upload["file_path"]
