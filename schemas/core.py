from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from enum import Enum

class StatusEnum(str, Enum):
    """Common status enum for all entities"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CREATED = "created"
    RUNNING = "running"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    UNDEPLOYING = "undeploying"

class BaseEntity(BaseModel):
    """Base class for all entities"""
    name: str = Field(..., description="Name of the entity")
    description: Optional[str] = Field(None, description="Description of the entity")
    status: StatusEnum = StatusEnum.CREATED
    parameters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(protected_namespaces=())

class BaseCreate(BaseEntity):
    """Base class for create operations"""
    pass

class BaseUpdate(BaseModel):
    """Base class for update operations"""
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[StatusEnum] = None
    parameters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseResponse(BaseEntity):
    """Base class for response operations"""
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class BaseResult(BaseModel):
    """Base class for result operations"""
    metrics: Dict[str, Any]
    plots_dir: Optional[str] = None
    report_path: Optional[str] = None
    recommendations: Optional[Dict[str, Any]] = None
    findings: Optional[List[Dict[str, Any]]] = None
    raw_output: Optional[Dict[str, Any]] = None

class BaseConfig(BaseModel):
    """Base class for configuration"""
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None 