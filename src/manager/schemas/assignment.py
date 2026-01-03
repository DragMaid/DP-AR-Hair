from enum import Enum
from pydantic import BaseModel
from datetime import datetime


class AssignmentStatus(str, Enum):
    SUCCEED = "succeed"
    FAILED = "failed"


class Assignment(BaseModel):
    id: str
    task_id: str
    worker_id: str
    created_at: datetime


class AssignmentHistory(BaseModel):
    id: str
    task_id: str
    worker_id: str
    status: AssignmentStatus
    created_at: datetime
    log: str
