import secrets
import logging
from typing import Optional
from .connect import get_cursor

logger = logging.getLogger(__name__)


def list_workers(
        email: Optional[str],
        limit: int = 100
):
    try:
        with get_cursor(dict_cursor=True) as cur:
            cur.execute("""
                SELECT id, username, role, created_at
                FROM users
                WHERE (%s IS NULL OR username = %s)
                    AND role = 'worker'::user_roles
                LIMIT %s
            """, (email, email, limit,))
            task = cur.fetchall()
            return task

    except Exception as e:
        logger.error(f"Error getting workers: {e}")
        raise


def create_worker(email: str) -> str:
    try:
        password = secrets.token_urlsafe(24)

        with get_cursor(dict_cursor=False) as cur:
            cur.execute("""
                INSERT INTO users (username, password_hash, role)
                VALUES (%s, crypt(%s, gen_salt('bf', 12)), 'worker')
            """, (email, password,))
            return password
    except Exception as e:
        logger.error(f"Error creating worker: {e}")
        raise


def remove_worker(worker_id: str) -> None:
    try:
        with get_cursor(dict_cursor=True) as cur:
            cur.execute("""
                DELETE FROM users
                WHERE role = 'worker'::user_roles
                AND id = %s
            """, (worker_id,))
    except Exception as e:
        logger.error(f"Error removing worker: {e}")
        raise


def reset_worker_password(worker_id: str) -> str:
    try:
        password = secrets.token_urlsafe(24)

        with get_cursor(dict_cursor=True) as cur:
            cur.execute("""
                UPDATE users
                SET password_hash = crypt(%s, gen_salt('bf', 12))
                WHERE id = %s
            """, (password, worker_id,))
            return password
    except Exception as e:
        logger.error(f"Error resetting worker account: {e}")
        raise
