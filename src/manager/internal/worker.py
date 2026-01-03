import secrets
from psycopg2 import IntegrityError
from typing import Optional
from .connect import get_cursor
from core.exceptions import wrap_errors, AppError


@wrap_errors(default_code="WORKER_INTERNAL_ERROR")
def list_workers(
    email: Optional[str],
    limit: int = 100
):
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


@wrap_errors(default_code="WORKER_INTERNAL_ERROR")
def create_worker(email: str):
    password = secrets.token_urlsafe(24)
    try:
        with get_cursor(dict_cursor=False) as cur:
            cur.execute("""
                INSERT INTO users (username, password_hash, role)
                VALUES (%s, crypt(%s, gen_salt('bf', 12)), 'worker')
            """, (email, password,))
    except IntegrityError as e:
        raise AppError("WORKER_CREATION_FAILED") from e

    return password


@wrap_errors(default_code="WORKER_INTERNAL_ERROR")
def remove_worker(worker_id: str):
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            DELETE FROM users
            WHERE role = 'worker'::user_roles
            AND id = %s
        """, (worker_id,))

        if cur.rowcount == 0:
            raise AppError("WORKER_NOT_FOUND")


@wrap_errors(default_code="WORKER_INTERNAL_ERROR")
def reset_worker_password(worker_id: str):
    password = secrets.token_urlsafe(24)

    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            UPDATE users
            SET password_hash = crypt(%s, gen_salt('bf', 12))
            WHERE id = %s
        """, (password, worker_id,))

        if cur.rowcount == 0:
            raise AppError("WORKER_NOT_FOUND")

    return password
