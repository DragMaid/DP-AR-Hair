import random
import os
from internal.connect import get_cursor
from dotenv import load_dotenv
from schemas.assignment import AssignmentStatus
from schemas.image import ImageTypes
from schemas.user import UserRoles
from client.core.session import session
from pathlib import Path
from core.config import settings

load_dotenv()

image_ids = []
task_ids = []
assignment_ids = []
upload_ids = {}


def seed_images(cursor, n=10):
    """Seed `images` table with n random image paths."""
    image_paths = [f"/images/img_{i}.jpg" for i in range(1, n)]
    for path in image_paths:
        image_type = random.choice(list(ImageTypes))
        cursor.execute("""
            INSERT INTO images (file_path, type)
            VALUES (%s, %s)
            ON CONFLICT (file_path) DO NOTHING
            RETURNING id
        """, (path, image_type,))
        image_id = cursor.fetchone()
        image_ids.append(image_id["id"])
    print(f"Seeded {n} images.")


def seed_god(cursor):
    username = os.environ["ADMIN_USERNAME"]
    password = os.environ["ADMIN_PASSWORD"]

    cursor.execute("""
        INSERT INTO users (username, password_hash, role)
        VALUES (%s, crypt(%s, gen_salt('bf', 12)), 'admin'::user_roles)
        ON CONFLICT (username) DO NOTHING
    """, (username, password))

    print("Seeded admin with variables from .env")


async def seed_admins(fetcher, n=1):
    assert n >= 1
    from client.api.auth import authorize
    from client.api.admin import create_admin

    # Authorize for god user first
    username = os.environ["ADMIN_USERNAME"]
    password = os.environ["ADMIN_PASSWORD"]
    await authorize(fetcher, username, password, UserRoles.ADMIN)

    for i in range(n):
        password = await create_admin(fetcher, f"admin{i}")

    # Logout of god
    session.clear()

    # Login to last created admin
    await authorize(fetcher, f"admin{n-1}", password, UserRoles.ADMIN)
    print("Test admin password is: ", password)


async def seed_workers(fetcher, n=1):
    from client.api.auth import authorize
    from client.api.worker import create_worker

    for i in range(n):
        password = await create_worker(fetcher, f"worker{i}")

    session.clear()
    await authorize(fetcher, f"worker{n-1}", password, UserRoles.WORKER)


async def seed_tasks(fetcher, n=1):
    from client.api.tasks import create_task
    from random import randint

    cnt = 0
    for i in range(len(image_ids)-1):
        for j in range(i+1, len(image_ids)):
            if cnt == n:
                break
            cnt += 1
            task_id = await create_task(
                fetcher,
                str(image_ids[i]),
                str(image_ids[j]),
                path=f"generated{cnt}.png",
                priority=randint(0, 10)
            )
            task_ids.append(task_id)


async def seed_assignments(fetcher, n=1):
    from client.api.tasks import claim_task
    for i in range(n):
        assignment_id = await claim_task(fetcher)
        assignment_ids.append(assignment_id)


async def seed_assignment_history(fetcher, n=1):
    from client.api.assignment import report_assignment
    for i in range(min(n, len(assignment_ids))):
        assignment_id = assignment_ids[i]
        upload_id = upload_ids.get(assignment_id)

        if not upload_id:
            continue

        await report_assignment(
            fetcher,
            assignment_id,
            upload_id,
            status=random.choice(list(AssignmentStatus)),
            log=""
        )


async def seed_upload(fetcher, n=1):
    from client.api.image import upload
    path = Path("../../assets/test_images/cropped.png")

    for i in range(min(n, len(assignment_ids))):
        if not os.path.isfile(path):
            return

        assignment_id = assignment_ids[i]
        id = await upload(fetcher, assignment_id, path)
        upload_ids[assignment_id] = id


async def seed_all(fetcher):
    await seed_admins(fetcher, 1)
    await seed_tasks(fetcher, 2)
    await seed_workers(fetcher, 1)
    await seed_assignments(fetcher, 2)
    await seed_upload(fetcher, 1)
    await seed_assignment_history(fetcher, 1)


if __name__ == "__main__":
    from client.api.fetcher import APIFetcher
    from httpx import AsyncClient
    import asyncio
    try:
        with get_cursor(dict_cursor=True) as cur:
            seed_god(cur)
            seed_images(cur, 20)

        client = AsyncClient()
        fetcher = APIFetcher(
            base_url="http://localhost:8000",
            client=client,
            session=session,
            strict=False
        )

        asyncio.run(seed_all(fetcher))
    except Exception as e:
        print(f"Error seeding databases: {e}")
        raise
