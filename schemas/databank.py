from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# TestDataset Schemas
class TestDatasetBase(BaseModel):
    name: str
    description: Optional[str] = None
    type: str = Field(..., description="Dataset type (text, image, tabular)")
    format: str = Field(default="json", description="File format (json, csv)")
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    dataset_schema: Optional[Dict[str, Any]] = None

class TestDatasetCreate(TestDatasetBase):
    pass

class TestDatasetUpdate(TestDatasetBase):
    name: Optional[str] = None
    type: Optional[str] = None
    format: Optional[str] = None

class TestDataset(TestDatasetBase):
    id: int
    file_path: str
    size: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# Cookbook Schemas
class CookbookBase(BaseModel):
    name: str
    description: Optional[str] = None
    version: str = Field(default="1.0.0")
    status: str = Field(default="active", description="active, deprecated, etc.")
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    datasets: List[str] = Field(default_factory=list, description="List of dataset names used")
    metrics: List[str] = Field(default_factory=list, description="List of metric names used")

class CookbookCreate(CookbookBase):
    pass

class CookbookUpdate(CookbookBase):
    name: Optional[str] = None
    version: Optional[str] = None
    status: Optional[str] = None

class Cookbook(CookbookBase):
    id: int
    file_path: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# Cookbook Run Schemas
class CookbookRunBase(BaseModel):
    cookbook_id: int
    project_id: int
    parameters: Dict[str, Any]

class CookbookRunCreate(CookbookRunBase):
    pass

class CookbookRunUpdate(BaseModel):
    status: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class CookbookRun(CookbookRunBase):
    id: int
    status: str
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# Recipe Result Schemas
class RecipeResultBase(BaseModel):
    run_id: int
    recipe_id: str
    status: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

class RecipeResultCreate(RecipeResultBase):
    pass

class RecipeResultUpdate(BaseModel):
    status: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class RecipeResult(RecipeResultBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

# Attack Module Schemas
class AttackModuleBase(BaseModel):
    name: str
    enabled: bool = True

class AttackModuleCreate(AttackModuleBase):
    pass

class AttackModuleUpdate(AttackModuleBase):
    name: Optional[str] = None
    enabled: Optional[bool] = None

class AttackModule(AttackModuleBase):
    file_path: str
    size: int
    modified: datetime
    config: Dict[str, Any]

    class Config:
        from_attributes = True 