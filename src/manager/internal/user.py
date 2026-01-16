import secrets
from psycopg2 import IntegrityError
from typing import Optional, List
from .connect import get_cursor
from core.exceptions import wrap_errors, AppError
from schemas.user import UserRoles, User


@wrap_errors(default_code="WORKER_INTERNAL_ERROR")
def list_users(
    email: Optional[str],
    role: UserRoles,
    owner_id: Optional[str],
    limit: int = 100,
) -> List[User]:
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            SELECT u.id, u.username, u.role, u.created_at
            FROM users u
            WHERE
                (%s IS NULL OR u.username = %s)
                AND u.role = %s::user_roles
                AND (
                    %s IS NULL
                    OR EXISTS (
                        SELECT 1
                        FROM ownership o
                        WHERE o.worker_id = u.id
                            AND o.admin_id = %s
                    )
                )
            ORDER BY u.created_at DESC
            LIMIT %s
        """, (email, email, role, owner_id, owner_id, limit,))
        users = cur.fetchall()
        return users


@wrap_errors(default_code="WORKER_INTERNAL_ERROR")
def create_user(email: str, role: UserRoles, admin_id: str) -> str:
    password = secrets.token_urlsafe(24)
    try:
        with get_cursor(dict_cursor=False) as cur:
            cur.execute("""
                INSERT INTO users (username, password_hash, role)
                VALUES (%s, crypt(%s, gen_salt('bf', 12)), %s)
                RETURNING id
            """, (email, password, role,))
            user_id = cur.fetchone()
            cur.execute("""
                INSERT INTO ownership (worker_id, admin_id)
                VALUES (%s, %s)
            """, (user_id, admin_id,))
    except IntegrityError as e:
        raise AppError("WORKER_CREATION_FAILED") from e

    return password


@wrap_errors(default_code="WORKER_INTERNAL_ERROR")
def remove_user(user_id: str, role: UserRoles) -> None:
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            DELETE FROM users
            WHERE role = %s::user_roles
            AND id = %s
        """, (role, user_id,))

        if cur.rowcount == 0:
            raise AppError("WORKER_NOT_FOUND")


@wrap_errors(default_code="WORKER_INTERNAL_ERROR")
def reset_worker_password(user_id: str, role: UserRoles) -> str:
    password = secrets.token_urlsafe(24)

    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            UPDATE users
            SET password_hash = crypt(%s, gen_salt('bf', 12))
            WHERE id = %s AND role = %s::user_roles
        """, (password, user_id, role))

        if cur.rowcount == 0:
            raise AppError("WORKER_NOT_FOUND")

    return password
