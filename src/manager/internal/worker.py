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
                SELECT *
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
        with get_cursor(dict_cursor=True) as cur:
            cur.execute("""
                WITH pwd AS (
                    SELECT encode(gen_random_bytes(24), 'base64') AS password
                )
                INSERT INTO users (email, password_hash)
                SELECT %s, crypt(pwd.password, gen_salt('bf', 12))
                FROM pwd
                RETURNING pwd.password;
            """, (email,))
            password = cur.fetchone()
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


def authenticate_worker(email: str, password: str):
    try:
        with get_cursor(dict_cursor=True) as cur:
            cur.execute("""
                SELECT id
                FROM users
                WHERE username = %s AND role = 'worker'::user_roles
                    AND password_hash = crypt(%s, password_hash);
            """, (email, password,))
            id = cur.fetchone()
            return id is not None
    except Exception as e:
        logger.error(f"Error authenticating worker: {e}")
        raise
