import asyncio
from httpx import AsyncClient
from client.api.tasks import get_tasks
from client.api.fetcher import APIFetcher
from client.api.config import settings
from client.api.auth import Session

if __name__ == "__main__":
    client = AsyncClient()
    fetcher = APIFetcher(
        base_url=settings.BASE_URL,
        client=client,
        session=Session,
    )

    async def main():
        data = await get_tasks(fetcher, status=["pending", "completed"])
        print(data)

    asyncio.run(main())
