import logging
from datetime import datetime
from enum import Enum
from typing import Optional, List
from .connect import get_cursor

logger = logging.getLogger(__name__)


class AssignmentStatus(str, Enum):
    FAILED = "failed"
    SUCCEED = "succeed"


def list_assignments(
    limit: int = 100
):
    try:
        with get_cursor(dict_cursor=True) as cur:
            cur.execute("""
                SELECT
                    a.id,
                    a.task_id,
                    a.worker_id,
                    a.created_at
                FROM assignments a
                JOIN tasks_ordered t ON a.task_id = t.id
                ORDER BY t.priority DESC, a.created_at ASC
                LIMIT %s
            """, (limit,))
            assignments = cur.fetchall()
            return assignments
    except Exception as e:
        logger.error(f"Error getting assignments: {e}")
        raise


def list_assignment_history(
    status: Optional[List[AssignmentStatus]],
    limit: int = 100
):
    try:
        with get_cursor(dict_cursor=True) as cur:
            cur.execute("""
                SELECT
                    a.id,
                    a.task_id,
                    a.worker_id,
                    a.status,
                    a.log,
                    a.created_at
                FROM assignment_history a
                JOIN tasks_ordered t ON a.task_id = t.id
                WHERE (
                    %s IS NULL
                    OR a.status = ANY(%s::assignment_status[])
                )
                ORDER BY t.priority DESC, a.created_at ASC
                LIMIT %s
            """, (status, status, limit,))
            assignments = cur.fetchall()
            return assignments
    except Exception as e:
        logger.error(f"Error getting assignments: {e}")
        raise


def report_assignment(
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
                    SET status = 'completed', completed_at = %s
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
