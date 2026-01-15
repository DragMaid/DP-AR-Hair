import random
import os
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
from manager.internal.connect import get_cursor
from manager.schemas.assignment import AssignmentStatus
from manager.schemas.image import ImageCategories
from manager.schemas.user import UserRoles
from manager.client.core.session import session

load_dotenv()

image_ids = []
task_ids = []
assignment_ids = []
upload_id_maps = defaultdict(dict)


def seed_images(cursor, n=10):
    """Seed `images` table with n random image paths."""
    image_paths = [f"/images/img_{i}.jpg" for i in range(1, n)]
    for path in image_paths:
        image_type = random.choice(list(ImageCategories))
        cursor.execute("""
            INSERT INTO images (file_path, category)
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
    from manager.client.api.auth import authorize
    from manager.client.api.admin import create_admin

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
    from manager.client.api.auth import authorize
    from manager.client.api.worker import create_worker

    for i in range(n):
        password = await create_worker(fetcher, f"worker{i}")

    session.clear()
    await authorize(fetcher, f"worker{n-1}", password, UserRoles.WORKER)


async def seed_tasks(fetcher, n=1):
    from manager.client.api.tasks import create_task
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
    from manager.client.api.tasks import claim_task
    for i in range(n):
        response = await claim_task(fetcher)
        assignment_ids.append(response["assignment_id"])


async def seed_assignment_history(fetcher, n=1):
    from manager.client.api.assignment import report_assignment
    for i in range(min(n, len(assignment_ids))):
        assignment_id = assignment_ids[i]
        upload_id_map = upload_id_maps.get(assignment_id)

        if not upload_id_map:
            continue

        await report_assignment(
            fetcher=fetcher,
            assignment_id=assignment_id,
            driving_upload_id=upload_id_map["driving"],
            reference_upload_id=upload_id_map["reference"],
            generated_upload_id=upload_id_map["generated"],
            status=random.choice(list(AssignmentStatus)),
            log=""
        )


async def seed_upload(fetcher, n=1):
    from manager.client.api.image import upload
    path = Path("../../assets/test_images/cropped.png")

    for i in range(min(n, len(assignment_ids))):
        if not os.path.isfile(path):
            return

        for category in [e.value for e in ImageCategories]:
            assignment_id = assignment_ids[i]
            id = await upload(
                fetcher=fetcher,
                assignment_id=assignment_id,
                path=path,
                category=category
            )
            upload_id_maps[assignment_id][category] = id


async def seed_all(fetcher):
    await seed_admins(fetcher, 1)
    await seed_tasks(fetcher, 2)
    await seed_workers(fetcher, 1)
    await seed_assignments(fetcher, 2)
    await seed_upload(fetcher, 1)
    await seed_assignment_history(fetcher, 1)


if __name__ == "__main__":
    from manager.client.api.fetcher import APIFetcher
    from httpx import AsyncClient
    import asyncio
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args()

    try:
        with get_cursor(dict_cursor=True) as cur:
            seed_god(cur)

        if args.debug:
            with get_cursor(dict_cursor=True) as cur:
                seed_images(cur, 20)

            client = AsyncClient()
            fetcher = APIFetcher(
                base_url="http://localhost:80/api/",
                client=client,
                session=session,
                strict=False
            )

            asyncio.run(seed_all(fetcher))

    except Exception as e:
        print(f"Error seeding databases: {e}")
        raise
