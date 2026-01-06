from fastapi import Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from schemas.user import UserRoles
from pydantic import BaseModel
from core.exceptions import wrap_errors, AppError
from .connect import get_cursor
from core.jwt_manager import decode_access_token
from core.config import settings


class TokenData(BaseModel):
    username: str


@wrap_errors(default_code="AUTH_INTERNAL_ERROR")
def extract_bearer_token(
    credentials: HTTPAuthorizationCredentials = Depends(
        HTTPBearer(auto_error=False))
):
    if credentials is None:
        raise AppError("MISSING_AUTH_HEADER")

    if credentials.scheme.lower() != "bearer":
        raise AppError("INVALID_CREDENTIALS")

    return credentials.credentials


@wrap_errors(default_code="AUTH_INTERNAL_ERROR")
def get_user(user_id: str):
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            SELECT id, username, role, created_at
            FROM users
            WHERE id = %s
        """, (user_id,))
        user = cur.fetchone()
        if not user:
            raise AppError("USER_NOT_FOUND")
        return user


@wrap_errors(default_code="AUTH_INTERNAL_ERROR")
def get_current_user(token: str = Depends(extract_bearer_token)):
    user_id = decode_access_token(token)
    try:
        user = get_user(user_id=user_id)
    except AppError as e:
        if e.code == "USER_NOT_FOUND":
            raise AppError("INVALID_CREDENTIALS")
        raise

    return user


@wrap_errors(default_code="AUTH_INTERNAL_ERROR")
def require_god(user=Depends(get_current_user)):
    # TODO: this is considerable
    if user["username"] != settings.ADMIN_USERNAME:
        raise AppError("FORBIDDEN")
    return user


@wrap_errors(default_code="AUTH_INTERNAL_ERROR")
def require_worker(user=Depends(get_current_user)):
    if user["role"] != UserRoles.WORKER:
        raise AppError("FORBIDDEN")
    return user


@wrap_errors(default_code="AUTH_INTERNAL_ERROR")
def require_admin(user=Depends(get_current_user)):
    if user["role"] != UserRoles.ADMIN:
        raise AppError("FORBIDDEN")
    return user


@wrap_errors(default_code="AUTH_INTERNAL_ERROR")
def require_ownership(
    user=Depends(get_current_user),
    owned_id: str = Query(alias="worker_id")
):
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            SELECT admin_id
            FROM ownership
            WHERE worker_id = %s
        """, (owned_id,))
        admin_id = cur.fetchone()

        if not admin_id or admin_id != user["id"]:
            raise AppError("FORBIDDEN")


@wrap_errors(default_code="AUTH_INTERNAL_ERROR")
def authenticate_user(username: str, password: str, role: str):
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            SELECT id, username, role, created_at
            FROM users
            WHERE username = %s AND role = %s::user_roles
                AND password_hash = crypt(%s, password_hash);
        """, (username, role, password,))
        user = cur.fetchone()
        if not user:
            raise AppError("INVALID_CREDENTIALS")
        return user
