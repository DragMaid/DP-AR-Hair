import os
from pydantic import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    POSTGRES_USER: str = os.environ["POSTGRES_USER"]
    POSTGRES_PASSWORD: str = os.environ["POSTGRES_PASSWORD"]
    POSTGRES_PASSWORD_ENCODED: str = os.environ["POSTGRES_PASSWORD_ENCODED"]
    POSTGRES_DB: str = os.environ["POSTGRES_DB"]
    POSTGRES_PORT: str = os.environ["POSTGRES_PORT"]
    POSTGRES_HOST: str = os.environ["POSTGRES_HOST"]
    DATABASE_URL: str = os.environ["DATABASE_URL"]

    ADMIN_USERNAME: str = os.environ["ADMIN_USERNAME"]
    ADMIN_PASSWORD: str = os.environ["ADMIN_PASSWORD"]

    SECRET_KEY: str = os.environ["SECRET_KEY"]
    ALGORITHM: str = os.environ["ALGORITHM"]

    TOKEN_EXPIRATION_MIN: int = 2400  # 4 hours


settings = Settings()
