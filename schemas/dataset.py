from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy import MetaData

class DatasetBase(BaseModel):
    name: str
    description: Optional[str] = None
    dataset_type: str
    file_path: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    project_id: int
    user_id: int

class DatasetCreate(DatasetBase):
    pass

class DatasetResponse(DatasetBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
        
    @classmethod
    def from_orm(cls, obj):
        # Ensure metadata is a dictionary
        if hasattr(obj, 'metadata') and obj.metadata is not None:
            if not isinstance(obj.metadata, dict):
                obj.metadata = {}
        return super().from_orm(obj)
