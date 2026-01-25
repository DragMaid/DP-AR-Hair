from typing import Optional, List
from .connect import get_cursor
from manager.schemas.task import TaskStatus, Task
from manager.core.exceptions import AppError, wrap_errors
from manager.core.config import settings


@wrap_errors(default_code="TASK_INTERNAL_ERROR")
def list_tasks(
    status: Optional[List[TaskStatus]],
    limit: int = 100
) -> List[Task]:
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            SELECT
                id,
                driving_image_id,
                reference_image_id,
                result_path,
                retry_count,
                priority,
                status,
                created_at
            FROM tasks_ordered t
            WHERE (
                %s IS NULL
                OR status = ANY(%s::task_status[])
            )
            ORDER BY priority DESC, created_at ASC
            LIMIT %s
        """, (status, status, limit,))
        return cur.fetchall()


@wrap_errors(default_code="TASK_INTERNAL_ERROR")
def claim_task(worker_id: str) -> dict:

    # TODO: Remove the hard coded 3 later
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            SELECT id
            FROM tasks
            WHERE status = 'pending' AND retry_count < 3
            ORDER BY retry_count ASC, priority DESC, created_at
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        """)
        task = cur.fetchone()
        if not task:
            raise AppError("QUEUE_EMPTY")

        cur.execute("""
            UPDATE tasks
            SET status = 'processing'
            WHERE id = %s
        """, (task["id"],))

        cur.execute("""
            INSERT INTO assignments(worker_id, task_id, expires_at)
            VALUES (%s, %s, NOW() + %s * INTERVAL '1 minute')
            RETURNING id, task_id
        """, (worker_id, task["id"], settings.ASSIGNMENT_TIMEOUT_MIN,))

        assignment = cur.fetchone()
        if not assignment:
            raise AppError("ASSIGNMENT_CREATION_FAILED")

        cur.execute("""
            WITH task_data AS (
                SELECT
                    t.driving_image_id,
                    t.reference_image_id,
                    t.result_path
                FROM tasks t
                WHERE t.id = %s
                LIMIT 1
            ),
            driving_path AS (
                SELECT file_path
                FROM images i
                JOIN task_data t ON i.id = t.driving_image_id
            ),
            reference_path AS (
                SELECT file_path
                FROM images i
                JOIN task_data t ON i.id = t.reference_image_id
            )
            SELECT
                %s as assignment_id,
                d.file_path as driving_path,
                r.file_path as reference_path
            FROM task_data t
            CROSS JOIN driving_path d
            CROSS JOIN reference_path r
        """, (assignment["task_id"], assignment["id"]))

        # TOOD: create an error for this later
        response = cur.fetchone()
        if not response:
            raise

        return dict(response)


@wrap_errors(default_code="TASK_INTERNAL_ERROR")
def create_task(
    driving_id: str,
    reference_id: str,
    path: str,
    priority: int,
    host: Optional[str] = None
) -> str:
    with get_cursor(dict_cursor=True, host=host) as cur:
        cur.execute("""
            INSERT INTO tasks (
                driving_image_id,
                reference_image_id,
                result_path,
                priority
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT ON CONSTRAINT unique_task_image_combination
            DO NOTHING
            RETURNING id
        """, (driving_id, reference_id, path, priority,))
        task_id = cur.fetchone()
        if not task_id:
            raise AppError("TASK_CREATION_FAILED")
        return task_id["id"]


@wrap_errors(default_code="TASK_INTERNAL_ERROR")
def delete_task(task_id: str) -> None:
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            DELETE FROM tasks
            WHERE id = %s;
        """, (task_id,))
        if cur.rowcount == 0:
            raise AppError("TASK_NOT_FOUND")


@wrap_errors(default_code="TASK_INTERNAL_ERROR")
def get_progress() -> dict:
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            SELECT
                COUNT(*) AS total_tasks,
                COUNT(*) FILTER (WHERE status = 'completed'::task_status) AS completed_tasks
            FROM tasks;
        """, ())
        row = cur.fetchone()
        return {
            "total_count": row["total_tasks"],
            "completed_count": row["completed_tasks"]
        }
