from .fetcher import APIFetcher
from manager.routers.image import UploadResponse
from manager.schemas.image import ImageCategories


async def upload(
    fetcher: APIFetcher,
    assignment_id: str,
    category: ImageCategories,
    path: str
) -> str:
    with open(path, 'rb') as f:
        filename = str(path).split('/')[-1]
        extension = filename.split('.')[-1]
        files = {"file": (filename, f, f"image/{extension}")}

        response = await fetcher.fetch(
            method="POST",
            path="/images/upload",
            files=files,
            require_auth=True,
            params={"assignment_id": assignment_id, "category": category},
            response_model=UploadResponse
        )

        return response["upload_id"]
