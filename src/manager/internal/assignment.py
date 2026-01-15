from typing import Optional, List
from pydantic import BaseModel
from manager.schemas.assignment import AssignmentStatus, Assignment, AssignmentHistory
from manager.schemas.image import ImageCategories
from manager.core.exceptions import AppError
from manager.core.exceptions import wrap_errors
from .connect import get_cursor
from .auth import require_assignment_ownership
from .image import retrieve_upload


class UploadPathMap(BaseModel):
    driving: str
    reference: str
    generated: str


def list_assignments(
    owner_id: Optional[str],
    limit: int = 100
) -> List[Assignment]:
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            SELECT
                a.id,
                a.task_id,
                a.worker_id,
                a.created_at
            FROM assignments a
            JOIN tasks_ordered t ON a.task_id = t.id
            WHERE %s IS NULL OR EXISTS (
                SELECT 1
                FROM ownership o
                WHERE a.worker_id = o.worker_id
                    AND o.admin_id =  %s
            )
            ORDER BY t.priority DESC, a.created_at ASC
            LIMIT %s
        """, (owner_id, owner_id, limit,))
        assignments = cur.fetchall()
        return assignments


def list_assignment_history(
    status: Optional[List[AssignmentStatus]],
    owner_id: Optional[str],
    limit: int = 100
) -> List[AssignmentHistory]:
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            SELECT
                a.id,
                a.task_id,
                a.worker_id,
                a.status,
                a.log,
                a.created_at
            FROM assignment_history_ordered a
            JOIN tasks_ordered t ON a.task_id = t.id
            WHERE (
                %s IS NULL
                OR a.status = ANY(%s::assignment_status[])
                AND EXISTS (
                    SELECT 1
                    FROM ownership o
                    WHERE a.worker_id = o.worker_id
                        AND o.admin_id = %s
                )
            )
            ORDER BY a.created_at ASC, t.priority DESC
            LIMIT %s
        """, (status, status, owner_id, limit,))
        assignments = cur.fetchall()
        return assignments


@wrap_errors(default_code="ASSIGNMENT_REPORT_FAILED")
def report_assignment(
    assignment_id: str,
    worker_id: str,
    driving_upload_id: str,
    reference_upload_id: str,
    generated_upload_id: str,
    status: AssignmentStatus,
    log: str
) -> UploadPathMap:
    require_assignment_ownership(
        assignment_id=assignment_id,
        worker_id=worker_id
    )

    driving_path = retrieve_upload(
        upload_id=driving_upload_id,
        assignment_id=assignment_id,
        worker_id=worker_id,
        category=[ImageCategories.DRIVING]
    )

    reference_path = retrieve_upload(
        upload_id=reference_upload_id,
        assignment_id=assignment_id,
        worker_id=worker_id,
        category=[ImageCategories.REFERENCE]
    )

    generated_path = retrieve_upload(
        upload_id=generated_upload_id,
        assignment_id=assignment_id,
        worker_id=worker_id,
        category=[ImageCategories.GENERATED]
    )

    update_assignment(assignment_id, status, log)

    return UploadPathMap(
        driving=driving_path,
        reference=reference_path,
        generated=generated_path
    )


@wrap_errors(default_code="ASSIGNMENT_REPORT_FAILED")
def terminate_assignment(
    assignment_id: str,
    admin_id: str,
    log: str
) -> None:
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            SELECT id
            FROM assignments a
            WHERE a.id = %s AND EXISTS (
                SELECT 1
                FROM ownership o
                WHERE o.worker_id = a.worker_id
                    AND o.admin_id = %s
            )
        """, (assignment_id, admin_id,))
        a_id = cur.fetchone()
        if not a_id:
            raise AppError("ASSIGNMENT_NOT_FOUND")

        update_assignment(
            assignment_id,
            AssignmentStatus.TERMINATED,
            log
        )


@wrap_errors(default_code="ASSIGNMENT_REPORT_FAILED")
def update_assignment(
    assignment_id: str,
    status: AssignmentStatus,
    log: str
) -> None:
    with get_cursor(dict_cursor=True) as cur:
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
                SET status = 'completed'
                WHERE id = (
                    SELECT task_id
                    FROM assignments
                    WHERE id = %s)
                """, (assignment_id,))

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
