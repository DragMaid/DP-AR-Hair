from fastapi import APIRouter
from pydantic import BaseModel
from datetime import timedelta
from internal.auth import authenticate_user, create_token
from schemas.user import UserRoles

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
    user = authenticate_user(form.username, form.password, form.role)

    # TODO: move this config file later
    token_expire_min = timedelta(minutes=60*4)  # 4 hours is colab limit
    access_token = create_token(
        data={"sub": user.username},
        expires_delta=token_expire_min
    )

    return Token(access_token=access_token, token_type="bearer")
