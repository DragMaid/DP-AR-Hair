from enum import Enum
from pydantic import BaseModel
from datetime import datetime


class ImageTypes(str, Enum):
    DRIVING = "driving"
    REFERENCE = "reference"
    GENERATED = "generated"


class Image(BaseModel):
    id: str
    file_path: str
    type: ImageTypes
    created_at: datetime


class UploadStatus(str, Enum):
    PENDING = "pending"
    PROCESSED = "processed"


class Upload(BaseModel):
    id: str
    worker_id: str
    assignment_id: str
    file_path: str
    expires_at: datetime
    created_at: datetime
