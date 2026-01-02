import logging
from enum import Enum
from typing import Optional, List
from datetime import datetime
from .connect import get_cursor
from .assignment import AssignmentStatus

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    COMPLETED = "completed"
    PROCESSING = "processing"
    PENDING = "pending"


def list_tasks(
    status: Optional[List[TaskStatus]],
    limit: int = 100
):
    try:
        with get_cursor(dict_cursor=True) as cur:
            cur.execute(
                """
                SELECT *
                FROM tasks_ordered
                WHERE (
                    %s IS NULL
                    OR status = ANY(%s::task_status[])
                )
                ORDER BY priority DESC, created_at ASC
                LIMIT %s
                """, (status, status, limit,),
            )
            return cur.fetchall()

    except Exception as e:
        logger.error(f"Error getting tasks: {e}")
        raise


def claim_task(worker_id: str) -> Optional[str]:
    try:
        with get_cursor(dict_cursor=True) as cur:
            cur.execute("""
                SELECT *
                FROM tasks
                WHERE status = 'pending'
                ORDER BY priority, created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            """)
            task = cur.fetchone()
            if not task:
                return None

            cur.execute("""
                UPDATE tasks
                SET status = 'processing'
                WHERE id = %s
            """, (task["id"],))

            cur.execute("""
                INSERT INTO assignments(worker_id, task_id)
                VALUES (%s, %s)
                RETURNING id
            """, (worker_id, task["id"],))

            assignment_id = cur.fetchone()
            return assignment_id

    except Exception as e:
        logger.error(f"Error getting task: {e}")
        raise


def update_task(
    assignment_id: str,
    status: AssignmentStatus,
    log: str
) -> None:
    try:
        with get_cursor(dict_cursor=True) as cur:
            if status == AssignmentStatus.FAILED:
                cur.execute("""
                    UPDATE tasks
                    SET status = 'pending',  retry_count = retry_count + 1
                    WHERE id = (
                        SELECT task_id
                        FROM assignments
                        WHERE id = %s)
                    """, (assignment_id,))
            else:
                cur.execute("""
                    UPDATE tasks
                    SET status = 'completed', completed_at = %s,
                    WHERE id = (
                        SELECT task_id
                        FROM assignments
                        WHERE id = %s)
                    """, (datetime.now(), assignment_id,))

            cur.execute("""
                INSERT INTO assignment_history (
                    task_id,
                    worker_id,
                    status,
                    log
                )
                SELECT
                    task_id,
                    worker_id,
                    %s,
                    %s
                FROM assignments a
                WHERE a.id = %s;
            """, (status, log, assignment_id,))

            cur.execute("""
                DELETE FROM assignments
                WHERE id = %s;
            """, (assignment_id,))

    except Exception as e:
        logger.error(f"Error updating task: {e}")
        raise


def create_task(
    drive_id: str,
    ref_id: str,
    path: str,
    priority: int
) -> None:
    try:
        with get_cursor(dict_cursor=True) as cur:
            cur.execute("""
                INSERT INTO tasks (
                    driving_image_id,
                    reference_image_id,
                    result_path,
                    priority
                )
                VALUES (%s, %s, %s, %s);
            """, (drive_id, ref_id, path, priority,))
    except Exception as e:
        logger.error(f"Error adding task: {e}")
        raise


def delete_task(
    task_id: str
) -> None:
    try:
        with get_cursor(dict_cursor=True) as cur:
            cur.execute("""
                DELETE FROM tasks
                WHERE id = %s;
            """, (task_id,))
    except Exception as e:
        logger.error(f"Error adding task: {e}")
        raise
