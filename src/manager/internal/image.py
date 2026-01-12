from fastapi import UploadFile
from typing import Optional, List
from .connect import get_cursor
from schemas.image import Image, ImageTypes, UploadStatus
from core.exceptions import AppError, wrap_errors
from core.config import settings
from PIL import Image as PILImage, UnidentifiedImageError
from shutil import copyfileobj, move
import tempfile
import mimetypes
from pathlib import Path

# TODO: map image errors to frontend
UPLOAD_DIR = Path(settings.IMAGE_TMP_FOLDER)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@wrap_errors(default_code="IMAGE_INTERNAL_ERROR")
def get_generated_name(assignment_id: str) -> str:
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            SELECT i.file_path
            FROM images i
            JOIN tasks t ON t.driving_image_id = i.id
            JOIN assignments a ON a.task_id = t.id
            WHERE a.id = %s
            LIMIT 1
        """, (assignment_id,))

        row = cur.fetchone()
        if not row or not row.get("file_path"):
            raise ValueError(
                f"No source image found for assignment {assignment_id}")

        original_filename = row["file_path"].split('/')[-1]
        file_id_parts = original_filename.split('_')[:-1]

        if not file_id_parts:
            raise ValueError(f"Invalid original filename: {original_filename}")

        file_id = '_'.join(file_id_parts)
        generated_name = f"{file_id}_generated"
        return generated_name


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

    ext = mimetypes.guess_extension(upload_file.content_type or "")
    if not ext:
        raise ValueError("Unsupported or unknown file type")

    final_path = upload_dir / f"{name}{ext}"

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
    image_type: Optional[List[ImageTypes]],
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
                OR type = ANY(%s::image_types)
            )
            ORDER BY created_at ASC
            LIMIT %s
        """, (image_type, limit,))
        return cur.fetchall()


@wrap_errors(default_code="IMAGE_INTERNAL_ERROR")
def insert_image(
    file_path: str,
    image_type: ImageTypes,
) -> str:
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            INSERT INTO images (file_path, type)
            VALUES (%s, %s)
            RETURNING id
        """, (file_path, image_type,))
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
    status: UploadStatus,
) -> str:
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            INSERT INTO uploads (id, worker_id, assignment_id, file_path, status, expires_at)
            VALUES (%s, %s, %s, %s, %s, NOW() + %s * INTERVAL '1 minute')
            RETURNING id;
        """, (id, worker_id, assignment_id, file_path, status, settings.UPLOAD_TIMEOUT_MIN,))
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
