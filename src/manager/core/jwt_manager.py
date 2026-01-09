import jwt
from datetime import datetime, timedelta, timezone
from cachetools import TTLCache
from time import time
from .config import settings
from .exceptions import wrap_errors, AppError


last_activity_cache = TTLCache(
    maxsize=settings.TOKEN_STORAGE_CAPACITY,
    ttl=settings.TOKEN_INACTIVE_EXPIRATION_MIN
)


@wrap_errors(default_code="TOKEN_CREATION_FAILED")
def create_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire})
    token = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    last_activity_cache[to_encode["sub"]] = {
        "token": token, "last_seen": time()}
    return token


@wrap_errors(default_code="TOKEN_INTERNAL_ERROR")
def decode_access_token(token: str) -> str:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY,
                             algorithms=[settings.ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise AppError("INVALID_TOKEN")
        # TODO: this way of handling tokens is not good but sufficient for closed env
        # Checks if token expired due to inactivity
        check_sliding_token(user_id, token)
        return user_id

    except jwt.ExpiredSignatureError:
        raise AppError("TOKEN_EXPIRED")

    except jwt.PyJWTError:
        raise AppError("INVALID_TOKEN")


@wrap_errors(default_code="TOKEN_INTERNAL_ERROR")
def check_sliding_token(user_id: str, token: str):
    now = time()
    user_cache = last_activity_cache.get(user_id)
    if not user_cache:
        raise AppError("TOKEN_EXPIRED")
    if now - user_cache["last_seen"] > settings.TOKEN_INACTIVE_EXPIRATION_MIN:
        raise AppError("TOKEN_EXPIRED")
    if user_cache["token"] != token:
        raise AppError("INVALID_TOKEN")
