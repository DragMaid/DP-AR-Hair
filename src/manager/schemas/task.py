from enum import Enum
from pydantic import BaseModel
from datetime import datetime


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"


class Task(BaseModel):
    id: str
    driving_image_id: str
    reference_image_id: str
    result_path: str
    retry_count: int
    priority: int
    status: TaskStatus
    created_at: datetime
