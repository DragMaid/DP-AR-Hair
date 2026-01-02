import logging
from enum import Enum
from typing import Optional, List
from .connect import get_cursor

logger = logging.getLogger(__name__)


class AssignmentStatus(str, Enum):
    FAILED = "failed"
    SUCCEED = "succeed"
    PROCESSING = "processing"


def list_assignments(
    limit: int = 100
):
    try:
        with get_cursor(dict_cursor=True) as cur:
            cur.execute("""
                SELECT *
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
                SELECT *
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
