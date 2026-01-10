from typing import Optional, List
from .connect import get_cursor
from schemas.task import TaskStatus, Task
from core.exceptions import AppError, wrap_errors
from core.config import settings


# TODO: add return types later
# TODO: add proper select tags later
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


# TODO: add a timeout to clear the assignment
@wrap_errors(default_code="TASK_INTERNAL_ERROR")
def claim_task(worker_id: str) -> str:
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            SELECT id
            FROM tasks
            WHERE status = 'pending'
            ORDER BY priority, created_at
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        """)
        task = cur.fetchone()
        if not task:
            return AppError("QUEUE_EMPTY")

        cur.execute("""
            UPDATE tasks
            SET status = 'processing'
            WHERE id = %s
        """, (task["id"],))

        cur.execute("""
            INSERT INTO assignments(worker_id, task_id, expires_at)
            VALUES (%s, %s, NOW() + %s * INTERVAL '1 minute')
            RETURNING id
        """, (worker_id, task["id"], settings.ASSIGNMENT_TIMEOUT_MIN,))

        assignment_id = cur.fetchone()
        if not assignment_id:
            raise AppError("ASSIGNMENT_CREATION_FAILED")

        return assignment_id["id"]


@wrap_errors(default_code="TASK_INTERNAL_ERROR")
def create_task(
    driving_id: str,
    reference_id: str,
    path: str,
    priority: int
) -> str:
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            INSERT INTO tasks (
                driving_image_id,
                reference_image_id,
                result_path,
                priority
            )
            VALUES (%s, %s, %s, %s)
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
