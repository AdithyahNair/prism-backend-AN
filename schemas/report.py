from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class ReportBase(BaseModel):
    project_id: int
    model_id: int
    dataset_id: int
    report_type: str
    blockchain_hash: str
    file_path: str
    metadata: Optional[Dict[str, Any]] = None

class ReportCreate(ReportBase):
    pass

class ReportResponse(ReportBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True 