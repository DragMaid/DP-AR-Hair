import os
import psycopg2
import psycopg2.extras
import logging
from dotenv import load_dotenv
from contextlib import contextmanager
from typing import Optional
from enum import Enum
from datetime import datetime
import manager.seed as seeder

logger = logging.getLogger(__name__)
load_dotenv()


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


def get_connection():
    """
    Create a new PostgreSQL connection.
    Caller is responsible for closing it.
    """
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", 5432),
    )


@contextmanager
def get_cursor(commit: bool = True, dict_cursor: bool = False) -> None:
    """
    Context-managed cursor with automatic commit / rollback.

    Usage:
        with get_cursor() as cur:
            cur.execute(...)
    """
    conn = get_connection()
    try:
        cursor_factory = (
            psycopg2.extras.RealDictCursor if dict_cursor else None
        )
        cur = conn.cursor(cursor_factory=cursor_factory)
        yield cur
        cur.close()
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def seed_all():
    try:
        with get_cursor(dict_cursor=True) as cur:
            seeder.seed_images(cur)
            seeder.seed_workers(cur)
            seeder.seed_tasks(cur)
            seeder.seed_assignments(cur)
    except Exception as e:
        logger.error(f"Error seeding databases: {e}")
        cur.rollback()
        raise


def list_tasks(limit=100):
    try:
        with get_cursor(dict_cursor=True) as cur:
            cur.execute("""
                SELECT *
                FROM tasks_ordered
                ORDER BY priority, created_at
                LIMIT %s
            """, (limit,))
            task = cur.fetchall()
            return task

    except Exception as e:
        logger.error(f"Error getting tasks: {e}")
        raise


def list_workers(limit=100):
    try:
        with get_cursor(dict_cursor=True) as cur:
            cur.execute("""
                SELECT *
                FROM workers
                LIMIT %s
            """, (limit,))
            task = cur.fetchall()
            return task

    except Exception as e:
        logger.error(f"Error getting workers: {e}")
        raise


def list_assignments(limit=100):
    try:
        with get_cursor(dict_cursor=True) as cur:
            cur.execute("""
                SELECT *
                FROM assignments_ordered a JOIN tasks_ordered t
                ON a.task_id = t.id
                ORDER BY a.status_rank, t.priority
                LIMIT %s
            """, (limit,))
            assignments = cur.fetchall()
            return assignments
    except Exception as e:
        logger.error(f"Error getting assignments: {e}")
        raise


def get_task(worker_id: str) -> Optional[str]:
    try:
        with get_cursor(dict_cursor=True) as cur:
            cur.execute("""
                SELECT *
                FROM tasks
                WHERE status='pending'
                ORDER BY priority, created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            """)
            task = cur.fetchone()
            if task:
                cur.execute(
                    "UPDATE tasks SET status = 'processing' WHERE id = %s",
                    (task["id"],)
                )
                cur.execute("""
                    INSERT INTO assignments(worker_id, task_id)
                    VALUES (%s, %s)
                    RETURNING id
                """, (worker_id, task["id"],))
                assignment_id = cur.fetch_bone()
                return assignment_id

    except Exception as e:
        logger.error(f"Error getting task: {e}")
        cur.rollback()
        raise


def update_task(assignment_id: str, status: ProcessingStatus) -> None:
    try:
        with get_cursor(dict_cursor=True) as cur:
            if status == ProcessingStatus.FAILED:
                cur.execute("""
                    UPDATE tasks
                    SET status='pending', completed_at=%s
                    WHERE id=(
                        SELECT task_id
                        FROM assignments
                        WHERE id=%s)
                    """, (datetime.now(), assignment_id,))
            else:
                cur.execute("""
                    UPDATE tasks
                    SET status = %s, retry_count = retry_count + 1
                    WHERE id=(
                        SELECT task_id
                        FROM assignments
                        WHERE id = %s)
                    """, (status, assignment_id,))
    except Exception as e:
        logger.error(f"Error updating task: {e}")
        cur.rollback()
        raise


if __name__ == "__main__":
    seed_all()
