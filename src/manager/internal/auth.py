import os
import jwt
import logging
from fastapi import HTTPException, status, Depends
from jwt.exceptions import InvalidTokenError
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel
from dotenv import load_dotenv
from .connect import get_cursor

load_dotenv()
logger = logging.getLogger(__name__)
SECRET_KEY = os.environ["SECRET_KEY"]
ALGORITHM = os.environ["ALGORITHM"]


class TokenData(BaseModel):
    username: str


# TODO: I actually wants the tokens to both be long lived and sliding
def create_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_user(username: str):
    try:
        with get_cursor(dict_cursor=True) as cur:
            cur.execute("""
                SELECT id, username, role, created_at
                FROM users
                WHERE username = %s
            """, (username,))
            task = cur.fetchone()
            return task

    except Exception as e:
        logger.error(f"Error getting users: {e}")
        raise


def get_current_user(token: str):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


# TODO: move the type to somewhere more usable
def require_worker(user=Depends(get_current_user)):
    if user.role != "worker":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to perform this action",
        )

    return user


def require_admin(user=Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to perform this action",
        )

    return user


def authenticate_user(username: str, password: str, role: str):
    try:
        with get_cursor(dict_cursor=True) as cur:
            cur.execute("""
                SELECT id, username, role, created_at
                FROM users
                WHERE username = %s AND role = %s::user_roles
                    AND password_hash = crypt(%s, password_hash);
            """, (username, role, password,))
            user = cur.fetchone()
            return user
    except Exception as e:
        logger.error(f"Error authenticating user: {e}")
        raise
