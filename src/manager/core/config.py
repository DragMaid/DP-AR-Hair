import os
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


class Settings(BaseModel):
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
    TOKEN_INACTIVE_EXPIRATION_SEC: int = 20 * 60  # 20 minutes
    TOKEN_STORAGE_CAPACITY: int = 10000

    RATE_LIMITER_LIMIT: int = 30
    RATE_LIMITER_WINDOW_SEC: int = 60
    RATE_LIMITER_CAPACITY: int = 10000

    ASSIGNMENT_TIMEOUT_MIN: int = 20
    UPLOAD_TIMEOUT_MIN: int = 5

    IMAGE_TMP_FOLDER: Path = Path("assets/tmp/")
    GENERATED_IMAGE_DIR: Path = Path("assets/generated/")

    TMP_FILE_CLEANUP_MIN: int = 10


settings = Settings()
