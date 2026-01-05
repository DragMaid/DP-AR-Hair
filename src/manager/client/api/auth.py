from .fetcher import APIFetcher
from .session import Session
from schemas.user import UserRoles
from routers.auth import Token, LoginForm


async def authorize(
    fetcher: APIFetcher,
    username: str,
    password: str,
    role: UserRoles
) -> None:
    payload = LoginForm(
        username=username,
        password=password,
        role=role
    )

    token = await fetcher.fetch(
        method="GET",
        path="/login",
        json=payload,
        require_auth=False,
        response_model=Token
    )

    Session.set_token(token.access_token)
