from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime
from enum import Enum


# Project Schemas
class ProjectType(str, Enum):
    ML = "ML"
    LLM = "LLM"
    VISION = "VISION"
    AUDIO = "AUDIO"
    TABULAR = "TABULAR"


class ProjectStatus(str, Enum):
    NOT_STARTED = "NotStarted"
    IN_PROGRESS = "InProgress"
    COMPLETED = "Completed"
    FAILED = "Failed"
    ARCHIVED = "Archived"


class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None
    project_type: ProjectType
    status: ProjectStatus = ProjectStatus.NOT_STARTED


class ProjectCreate(ProjectBase):
    pass


class ProjectUpdate(ProjectBase):
    name: Optional[str] = None
    project_type: Optional[ProjectType] = None
    status: Optional[ProjectStatus] = None


class ProjectStatusUpdate(BaseModel):
    status: ProjectStatus


class ProjectResponse(ProjectBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

