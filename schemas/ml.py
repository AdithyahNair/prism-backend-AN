from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy import MetaData

class ModelBase(BaseModel):
    name: str
    description: Optional[str] = None
    model_type: str
    version: str = "1.0.0"
    file_path: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    project_id: int
    user_id: int

class ModelCreate(ModelBase):
    
    pass

class ModelResponse(ModelBase):
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