from typing import Optional, List
from .connect import get_cursor
from schemas.image import Image, ImageTypes
from core.exceptions import AppError, wrap_errors

# TODO: map image errors to frontend


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
