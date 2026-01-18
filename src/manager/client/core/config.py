from pydantic import BaseModel


class Settings(BaseModel):
    DEFAULT_TIMEOUT: int = 50.0
    DEFAULT_RETRIES: int = 3
    BASE_URL: str = "http://localhost/api"


settings = Settings()
