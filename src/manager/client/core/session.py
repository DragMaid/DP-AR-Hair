from dataclasses import dataclass
from typing import Optional


@dataclass
class Session:
    _access_token: Optional[str] = None

    def is_authenticated(self) -> bool:
        return self._access_token is not None

    def get_token(self) -> str:
        return self._access_token

    def set_token(self, token: str) -> None:
        self._access_token = token

    def clear(self) -> None:
        self._access_token = None


session = Session()
