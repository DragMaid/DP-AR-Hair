from fastapi import APIRouter
from pydantic import BaseModel
from datetime import timedelta
from internal.auth import authenticate_user, create_token
from schemas.user import UserRoles, User
from core.config import settings

router = APIRouter(
    prefix="/login",
    tags=["auth"],
    dependencies=[],
)


class Token(BaseModel):
    access_token: str
    token_type: str


class LoginForm(BaseModel):
    username: str
    password: str
    role: UserRoles


@router.post("/", response_model=Token)
def authorize(form: LoginForm):
    user = User(**authenticate_user(form.username, form.password, form.role))
    token_expire_min = timedelta(
        minutes=settings.TOKEN_EXPIRATION_MIN)
    access_token = create_token(
        data={"sub": user.username},
        expires_delta=token_expire_min
    )
    return Token(access_token=access_token, token_type="bearer")
