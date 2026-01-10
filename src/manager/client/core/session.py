from dataclasses import dataclass
from typing import Optional


@dataclass
class Session:
    _access_token: Optional[str] = None
    _user_id: Optional[str] = None

    def is_authenticated(self) -> bool:
        return self._access_token is not None

    def get_token(self) -> str:
        return self._access_token

    def set_token(self, token: str) -> None:
        self._access_token = token

    def set_user_id(self, user_id: str) -> None:
        self._user_id = user_id

    def get_user_id(self) -> None:
        return self._user_id

    def clear(self) -> None:
        self._access_token = None


session = Session()
