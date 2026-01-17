from pydantic import BaseModel
from typing import Optional, List
from manager.schemas.assignment import AssignmentStatus
from manager.schemas.user import UserRoles


class CreateAdminResponse(BaseModel):
    password: str


class ResetAdminResponse(BaseModel):
    password: str


class TerminateAssignmentBody(BaseModel):
    assignment_id: str
    log: str


class ReportAssignmentBody(TerminateAssignmentBody):
    generated_upload_id: Optional[str]
    driving_upload_id: Optional[str]
    reference_upload_id: Optional[str]
    status: AssignmentStatus


class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str


class LoginForm(BaseModel):
    username: str
    password: str
    role: UserRoles


class UploadResponse(BaseModel):
    upload_id: str


class CreateTaskBody(BaseModel):
    driving_id: str
    reference_id: str
    path: str
    priority: int


class CreateTaskResponse(BaseModel):
    task_id: str


class ClaimTaskResponse(BaseModel):
    assignment_id: str
    driving_path: str
    reference_path: str


class CreateWorkerResponse(BaseModel):
    password: str


class ResetWorkerResponse(BaseModel):
    password: str
