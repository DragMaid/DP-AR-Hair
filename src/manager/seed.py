import random
from datetime import datetime, timedelta

# Example file paths for images
SAMPLE_IMAGE_PATHS = [f"/images/img_{i}.jpg" for i in range(1, 21)]
SAMPLE_WORKER_EMAILS = [f"worker{i}@example.com" for i in range(1, 6)]
TASK_STATUSES = ["pending", "processing", "completed"]
ASSIGNMENT_STATUSES = ["succeed", "processing", "failed"]


def seed_images(cursor, n=10):
    """Seed `images` table with n random image paths."""
    for path in SAMPLE_IMAGE_PATHS[:n]:
        cursor.execute(
            "INSERT INTO images (file_path) VALUES (%s) RETURNING id",
            (path,)
        )
    print(f"Seeded {n} images.")


def seed_workers(cursor, n=5):
    """Seed `workers` table with n random emails."""
    for email in SAMPLE_WORKER_EMAILS[:n]:
        cursor.execute(
            "INSERT INTO workers (email) VALUES (%s) RETURNING id",
            (email,)
        )
    print(f"Seeded {n} workers.")


def seed_tasks(cursor, n=10):
    """Seed `tasks` table linking driving and reference images."""
    cursor.execute("SELECT id FROM images")
    image_ids = [row["id"] for row in cursor.fetchall()]

    if len(image_ids) < 2:
        raise ValueError("Need at least 2 images to seed tasks.")

    for _ in range(n):
        driving_id, reference_id = random.sample(image_ids, 2)
        result_path = f"/results/task_{driving_id}_{reference_id}.png"
        retry_count = random.randint(0, 3)
        priority = random.randint(0, 5)
        status = random.choice(TASK_STATUSES)
        created_at = datetime.now() - timedelta(days=random.randint(0, 10))
        completed_at = created_at + \
            timedelta(hours=random.randint(1, 72)
                      ) if status == "completed" else None

        cursor.execute(
            """
            INSERT INTO tasks
                (driving_image_id, reference_image_id, result_path, retry_count, priority, status, created_at, completed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (driving_id, reference_id, result_path, retry_count,
             priority, status, created_at, completed_at)
        )
    print(f"Seeded {n} tasks.")


def seed_assignments(cursor, n=10):
    """Seed `assignments` table linking tasks and workers with logs."""
    cursor.execute("SELECT id FROM tasks")
    task_ids = [row["id"] for row in cursor.fetchall()]
    cursor.execute("SELECT id FROM workers")
    worker_ids = [row["id"] for row in cursor.fetchall()]

    if not task_ids or not worker_ids:
        raise ValueError(
            "Need existing tasks and workers to seed assignments.")

    for _ in range(n):
        task_id = random.choice(task_ids)
        worker_id = random.choice(worker_ids)
        status = random.choice(ASSIGNMENT_STATUSES)
        logs = f"Assigned at {datetime.now()}"

        cursor.execute(
            """
            INSERT INTO assignments (task_id, worker_id, status, logs)
            VALUES (%s, %s, %s, %s)
            """,
            (task_id, worker_id, status, logs)
        )
    print(f"Seeded {n} assignments.")
