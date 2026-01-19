import os
import random
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from manager.internal.connect import get_cursor
from manager.schemas.assignment import AssignmentStatus
from manager.schemas.image import ImageCategories
from manager.schemas.user import UserRoles
from manager.client.core.session import session

load_dotenv()


class Seeder:
    def __init__(self):
        self.upload_id_maps = defaultdict(dict)

    def seed_images(self, cursor, n=10):
        """Seed `images` table with `n` random images. Returns list of IDs."""
        image_ids = []
        image_paths = [f"/images/img_{i}.jpg" for i in range(1, n + 1)]
        for path in image_paths:
            image_type = random.choice(list(ImageCategories))
            cursor.execute("""
                INSERT INTO images (file_path, category)
                VALUES (%s, %s)
                ON CONFLICT (file_path) DO NOTHING
                RETURNING id
            """, (path, image_type,))
            result = cursor.fetchone()
            if result:
                image_ids.append(result["id"])
        print(f"Seeded {len(image_ids)} images.")
        return image_ids

    def seed_god(self, cursor):
        """Seed admin user from .env variables."""
        username = os.environ["ADMIN_USERNAME"]
        password = os.environ["ADMIN_PASSWORD"]

        cursor.execute("""
            INSERT INTO users (username, password_hash, role)
            VALUES (%s, crypt(%s, gen_salt('bf', 12)), 'admin'::user_roles)
            ON CONFLICT (username) DO NOTHING
        """, (username, password))

        print("Seeded god from .env")

    async def seed_admins(self, fetcher, n=1):
        from manager.client.api.auth import authorize
        from manager.client.api.admin import create_admin

        username = os.environ["ADMIN_USERNAME"]
        password = os.environ["ADMIN_PASSWORD"]
        await authorize(fetcher, username, password, UserRoles.ADMIN)

        admin_ids = []
        for i in range(n):
            password = await create_admin(fetcher, f"admin{i}")
            admin_ids.append(f"admin{i}")

        session.clear()
        # Login to last created admin
        await authorize(fetcher, f"admin{n-1}", password, UserRoles.ADMIN)
        print("Test admin password is:", password)
        return admin_ids

    async def seed_workers(self, fetcher, n=1):
        from manager.client.api.auth import authorize
        from manager.client.api.worker import create_worker

        worker_ids = []
        for i in range(n):
            password = await create_worker(fetcher, f"worker{i}")
            worker_ids.append(f"worker{i}")

        session.clear()
        await authorize(fetcher, f"worker{n-1}", password, UserRoles.WORKER)
        return worker_ids

    async def seed_tasks(self, fetcher, image_ids, n=1):
        from manager.client.api.tasks import create_task
        from random import randint

        task_ids = []
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
                    path=f"./manager/assets/generated_images/generated{cnt}.png",
                    priority=randint(0, 10)
                )
                task_ids.append(task_id)
        return task_ids

    async def seed_assignments(self, fetcher, n=1):
        from manager.typings.backend import ClaimTaskResponse

        assignment_ids = []
        for _ in range(n):
            response = await fetcher.fetch(
                method="POST",
                path="/tasks/claim",
                require_auth=True,
                response_model=ClaimTaskResponse,
            )
            assignment_ids.append(response["assignment_id"])
        return assignment_ids

    async def seed_uploads(self, fetcher, assignment_ids, image_path="assets/test_images/cropped.png"):
        from manager.client.api.image import upload

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist")

        for assignment_id in assignment_ids:
            for category in ImageCategories:
                upload_id = await upload(fetcher, assignment_id, path, category)
                self.upload_id_maps[assignment_id][category] = upload_id
        return self.upload_id_maps

    async def seed_assignment_history(self, fetcher, assignment_ids):
        from manager.client.api.assignment import report_assignment

        for assignment_id in assignment_ids:
            upload_map = self.upload_id_maps.get(assignment_id)
            if not upload_map:
                continue

            await report_assignment(
                fetcher=fetcher,
                assignment_id=assignment_id,
                driving_upload_id=upload_map["driving"],
                reference_upload_id=upload_map["reference"],
                generated_upload_id=upload_map["generated"],
                status=AssignmentStatus.SUCCEED,
                log=""
            )

    async def seed_all(self, fetcher, cursor, image_count=10, task_count=5):
        # DB seed
        self.seed_god(cursor)
        image_ids = self.seed_images(cursor, image_count)

        # API seed
        await self.seed_admins(fetcher, 1)
        # await self.seed_workers(fetcher, 1)
        # task_ids = await self.seed_tasks(fetcher, image_ids, task_count)
        # assignment_ids = await self.seed_assignments(fetcher, len(task_ids))
        # await self.seed_uploads(fetcher, assignment_ids)
        # await self.seed_assignment_history(fetcher, assignment_ids)
        # return {
            # "images": image_ids,
            # "tasks": task_ids,
            # "assignments": assignment_ids,
            # "uploads": self.upload_id_maps
        # }


if __name__ == "__main__":
    import asyncio
    from httpx import AsyncClient
    from manager.client.api.fetcher import APIFetcher
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()

    try:
        seeder = Seeder()

        with get_cursor(dict_cursor=True, host="localhost") as cur:
            if args.debug:
                client = AsyncClient()
                fetcher = APIFetcher(
                    base_url="http://localhost:80/api/",
                    client=client,
                    session=session,
                    strict=False
                )

                asyncio.run(seeder.seed_all(
                    fetcher, cur, image_count=20, task_count=5))
            else:
                seeder.seed_god(cur)

    except Exception as e:
        print("Error during seeding:", e)
        raise
