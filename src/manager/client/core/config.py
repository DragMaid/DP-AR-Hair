from pydantic import BaseModel


class Settings(BaseModel):
    DEFAULT_TIMEOUT: int = 20.0
    DEFAULT_RETRIES: int = 3
    BASE_URL: str = "http://127.0.0.1:80/api"


settings = Settings()
