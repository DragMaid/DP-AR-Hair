from pydantic import BaseModel


class Settings(BaseModel):
    DEFAULT_TIMEOUT: int = 50.0
    DEFAULT_RETRIES: int = 3
    BASE_URL: str = "http://154.26.137.253/api"


settings = Settings()
