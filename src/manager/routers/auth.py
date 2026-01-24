from fastapi import APIRouter
from datetime import timedelta
from manager.internal.auth import authenticate_user
from manager.schemas.user import User
from manager.core.config import settings
from manager.core.jwt_manager import create_token
from manager.typings.backend import Token, LoginForm

router = APIRouter(
    prefix="/login",
    tags=["auth"],
    dependencies=[],
)


@router.post("", response_model=Token)
def authorize(form: LoginForm):
    user = User(**authenticate_user(
        form.username,
        form.password,
        form.role
    ))

    token_expire_min = timedelta(
        minutes=settings.TOKEN_EXPIRATION_MIN)

    access_token = create_token(
        data={"sub": user.id},
        expires_delta=token_expire_min
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        user_id=user.id
    )
