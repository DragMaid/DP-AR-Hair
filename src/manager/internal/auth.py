import os
import jwt
import logging
from fastapi import Depends
from jwt import PyJWTError
from datetime import datetime, timezone, timedelta
from schemas.user import UserRoles
from pydantic import BaseModel
from core.exceptions import wrap_errors, AppError
from dotenv import load_dotenv
from .connect import get_cursor

# TODO: move config to a centralized config file
load_dotenv()
logger = logging.getLogger(__name__)
SECRET_KEY = os.environ["SECRET_KEY"]
ALGORITHM = os.environ["ALGORITHM"]


class TokenData(BaseModel):
    username: str


# TODO: I actually wants the tokens to both be long lived and sliding
@wrap_errors(default_code="TOKEN_CREATION_FAILED")
def create_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


@wrap_errors(default_code="AUTH_INTERNAL_ERROR")
def get_user(username: str):
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            SELECT id, username, role, created_at
            FROM users
            WHERE username = %s
        """, (username,))
        user = cur.fetchone()
        if not user:
            raise AppError("USER_NOT_FOUND")
        return user


@wrap_errors(default_code="AUTH_INTERNAL_ERROR")
def get_current_user(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except PyJWTError:
        raise AppError("INVALID_TOKEN")

    username = payload.get("sub")
    if not username:
        raise AppError("INVALID_TOKEN")

    try:
        user = get_user(username=username)
    except AppError as e:
        if e.code == "USER_NOT_FOUND":
            raise AppError("INVALID_CREDENTIALS")
        raise

    return user


@wrap_errors(default_code="AUTH_INTERNAL_ERROR")
def require_worker(user=Depends(get_current_user)):
    if user.role != UserRoles.WORKER:
        raise AppError("FORBIDDEN")
    return user


@wrap_errors(default_code="AUTH_INTERNAL_ERROR")
def require_admin(user=Depends(get_current_user)):
    if user.role != UserRoles.ADMIN:
        raise AppError("FORBIDDEN")
    return user


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
