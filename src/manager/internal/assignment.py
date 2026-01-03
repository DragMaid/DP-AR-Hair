from datetime import datetime
from schemas.assignment import AssignmentStatus
from typing import Optional, List
from .connect import get_cursor
from core.exceptions import AppError
from core.exceptions import wrap_errors


def list_assignments(
    limit: int = 100
):
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


def list_assignment_history(
    status: Optional[List[AssignmentStatus]],
    limit: int = 100
):
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


@wrap_errors(default_code="ASSIGNMENT_REPORT_FAILED")
def report_assignment(
    assignment_id: str,
    worker_id: str,
    status: AssignmentStatus,
    log: str
) -> None:
    with get_cursor(dict_cursor=True) as cur:
        # Checks if assignment exists
        cur.execute("""
            SELECT id
            FROM assignments
            WHERE id = %s AND worker_id = %s
        """, (assignment_id, worker_id,))
        a_id = cur.fetchone()
        if not a_id:
            raise AppError("ASSIGNMENT_NOT_FOUND")

        # Update the task status based on report
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

        # Insert into history after assignment is done
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

        # Delete the original assignment
        cur.execute("""
            DELETE FROM assignments
            WHERE id = %s;
        """, (assignment_id,))
