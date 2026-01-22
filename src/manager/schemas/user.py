from enum import Enum
from pydantic import BaseModel
from datetime import datetime


class UserRoles(str, Enum):
    WORKER = "worker"
    ADMIN = "admin"


class User(BaseModel):
    id: str
    username: str
    role: UserRoles
    created_at: datetime


class UserDB(User):
    password_hash: str
