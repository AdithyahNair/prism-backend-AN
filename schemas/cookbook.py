from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class CookbookStatus(str, Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"

class CookbookType(str, Enum):
    BENCHMARK = "benchmark"
    RED_TEAM = "red_team"
    AUDIT = "audit"

class CookbookBase(BaseModel):
    name: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    recipes: Optional[List[str]] = None
    project_id: Optional[int] = None

class CookbookCreate(CookbookBase):
    pass

class CookbookUpdate(CookbookBase):
    name: Optional[str] = None
    status: Optional[CookbookStatus] = None

class CookbookResponse(CookbookBase):
    id: int
    status: CookbookStatus
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class CookbookRunBase(BaseModel):
    cookbook_id: int
    status: Optional[CookbookStatus] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None

class CookbookRunCreate(CookbookRunBase):
    pass

class CookbookRunResponse(CookbookRunBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class CookbookRunRequest(BaseModel):
    parameters: Dict[str, Any]

class RecipeResultResponse(BaseModel):
    recipe_id: str
    run_id: str
    status: str
    metrics: Dict[str, Any]
    predictions: List[Dict[str, Any]]
    evaluation_summary: List[Dict[str, Any]]
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None 