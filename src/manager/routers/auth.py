from fastapi import APIRouter
from fastapi import HTTPException, status
from pydantic import BaseModel
from datetime import timedelta
from enum import Enum
from internal.auth import authenticate_user, create_token

router = APIRouter(
    prefix="/",
    tags=["auth"],
    dependencies=[],
)


class Token(BaseModel):
    access_token: str
    token_type: str


class UserRoles(str, Enum):
    WORKER = "worker"
    ADMIN = "admin"


class LoginForm(BaseModel):
    username: str
    password: str
    role: UserRoles


@router.post("/login")
def authorize(form: LoginForm):
    user = authenticate_user(form.username,
                             form.password,
                             form.role)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # TODO: move this config file later
    token_expire_min = timedelta(minutes=60*4)  # 4 hours is colab limit
    access_token = create_token(
        data={"sub": user.username},
        expires_delta=token_expire_min
    )

    return Token(access_token=access_token, token_type="bearer")
