from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class AuditBase(BaseModel):
    project_id: int
    model_id: Optional[int] = None
    dataset_id: Optional[int] = None
    user_id: int
    audit_type: str = Field(..., description="Type of audit: red_teaming, benchmark, performance, etc.")
    status: str = Field(..., description="Status: pending, running, completed, failed")
    results: Optional[Dict[str, Any]] = None

class AuditCreate(AuditBase):
    pass

class AuditUpdate(BaseModel):
    status: Optional[str] = None
    results: Optional[Dict[str, Any]] = None

class AuditResponse(AuditBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class AuditList(BaseModel):
    id: int
    project_id: int
    audit_type: str
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
