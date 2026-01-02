import random
import os
from datetime import datetime, timedelta
from .internal.connect import get_cursor
from dotenv import load_dotenv

load_dotenv()

# Example file paths for images
SAMPLE_IMAGE_PATHS = [f"/images/img_{i}.jpg" for i in range(1, 21)]
SAMPLE_WORKER_EMAILS = [f"worker{i}@example.com" for i in range(1, 6)]

TASK_STATUSES = ["pending", "processing", "completed"]
ASSIGNMENT_STATUSES = ["succeed", "failed"]
IMAGE_TYPES = ["driving", "reference", "generated"]


def seed_images(cursor, n=10):
    """Seed `images` table with n random image paths."""
    for path in SAMPLE_IMAGE_PATHS[:n]:
        image_type = random.choice(IMAGE_TYPES)
        cursor.execute("""
            INSERT INTO images (file_path, type)
            VALUES (%s, %s)
            ON CONFLICT (file_path) DO NOTHING
        """, (path, image_type,))
    print(f"Seeded {n} images.")


def seed_workers(cursor, n=5):
    """Seed `workers` table with n random emails."""
    for email in SAMPLE_WORKER_EMAILS[:n]:
        cursor.execute("""
            INSERT INTO users (username, password_hash, role)
            VALUES (
                %s,
                crypt(%s, gen_salt('bf', 12)),
                'worker'::user_roles
            )
            ON CONFLICT (username) DO NOTHING
        """, (email, "nopassword"))
    print(f"Seeded {n} workers.")


def seed_tasks(cursor, n=10):
    """Seed `tasks` table linking driving and reference images."""
    from uuid import uuid4
    cursor.execute("SELECT id FROM images WHERE type = 'driving'")
    driving_image_ids = [row["id"] for row in cursor.fetchall()]

    cursor.execute("SELECT id FROM images WHERE type = 'reference'")
    reference_image_ids = [row["id"] for row in cursor.fetchall()]

    if len(driving_image_ids) < 2 or len(reference_image_ids) < 2:
        raise ValueError("Need at least 2 images for each type to seed tasks.")

    for _ in range(n):
        driving_id, reference_id = random.sample(
            reference_image_ids + driving_image_ids, 2)
        result_path = f"/results/{uuid4()}.png"
        retry_count = random.randint(0, 3)
        priority = random.randint(0, 5)
        status = random.choice(TASK_STATUSES)
        created_at = datetime.now() - timedelta(days=random.randint(0, 10))
        completed_at = created_at + \
            timedelta(hours=random.randint(1, 72)
                      ) if status == "completed" else None

        cursor.execute("""
            INSERT INTO tasks (
                 driving_image_id, reference_image_id,
                 result_path, retry_count,
                 priority, status,
                 created_at, completed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (driving_id, reference_id, result_path, retry_count,
              priority, status, created_at, completed_at))
    print(f"Seeded {n} tasks.")


def seed_assignments(cursor, n=10):
    """Seed `assignments` table linking tasks and workers with logs."""
    cursor.execute("SELECT id FROM tasks")
    task_ids = [row["id"] for row in cursor.fetchall()]
    cursor.execute("SELECT id FROM users WHERE role = 'worker'")
    worker_ids = [row["id"] for row in cursor.fetchall()]

    if not task_ids or not worker_ids:
        raise ValueError(
            "Need existing tasks and workers to seed assignments.")

    for _ in range(n):
        task_id = random.choice(task_ids)
        worker_id = random.choice(worker_ids)
        cursor.execute(
            """
            INSERT INTO assignments (task_id, worker_id)
            VALUES (%s, %s)
            """,
            (task_id, worker_id,)
        )
    print(f"Seeded {n} assignments.")


def seed_assignment_history(cursor, n=10):
    cursor.execute("SELECT id FROM tasks")
    task_ids = [row["id"] for row in cursor.fetchall()]
    cursor.execute("SELECT id FROM users WHERE role = 'worker'")
    worker_ids = [row["id"] for row in cursor.fetchall()]

    if not task_ids or not worker_ids:
        raise ValueError(
            "Need existing tasks and workers to seed assignment histories.")

    for _ in range(n):
        task_id = random.choice(task_ids)
        worker_id = random.choice(worker_ids)
        status = random.choice(ASSIGNMENT_STATUSES)
        cursor.execute("""
            INSERT INTO assignment_history (task_id, worker_id, status)
            VALUES (%s, %s, %s)
        """, (task_id, worker_id, status))

    print(f"Seeded {n} assignment histories.")


def seed_admin(cursor):
    username = os.environ["ADMIN_USERNAME"]
    password = os.environ["ADMIN_PASSWORD"]

    cursor.execute("""
        INSERT INTO users (username, password_hash, role)
        VALUES (%s, crypt(%s, gen_salt('bf', 12)), 'admin'::user_roles)
        ON CONFLICT (username) DO NOTHING
    """, (username, password))

    print("Seeded admin with variables from .env")


def seed_all(cursor):
    seed_images(cursor)
    seed_workers(cursor)
    seed_tasks(cursor)
    seed_assignments(cursor)
    seed_admin(cursor)
    seed_assignment_history(cur)


if __name__ == "__main__":
    try:
        with get_cursor(dict_cursor=True) as cur:
            # seed_all(cur)
            seed_assignment_history(cur)
    except Exception as e:
        print(f"Error seeding databases: {e}")
        raise
