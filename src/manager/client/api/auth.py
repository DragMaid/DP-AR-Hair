from .fetcher import APIFetcher
from manager.client.core.session import session
from manager.schemas.user import UserRoles
from manager.routers.auth import Token, LoginForm


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
        method="POST",
        path="/login",
        json=payload,
        require_auth=False,
        strict=True,
        response_model=Token
    )

    session.set_token(token.access_token)
    session.set_user_id(token.user_id)
