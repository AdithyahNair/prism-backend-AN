from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging

from core.deps import get_db, get_current_user
from models.models import User
from schemas.databank import (
    TestDataset, Cookbook, CookbookRun, CookbookRunCreate,
    RecipeResult, AttackModule
)
from services.databank_service import DatabankService
from utils.error_handlers import ValidationError

router = APIRouter(
    prefix="/databank",
    tags=["databank"],
    responses={404: {"description": "Not found"}}
)

logger = logging.getLogger(__name__)

# Dataset Management
@router.post("/datasets/upload", response_model=TestDataset)
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload a test dataset to the databank"""
    databank_service = DatabankService(db)
    return await databank_service.upload_dataset(file, current_user)

@router.get("/datasets", response_model=List[TestDataset])
async def list_datasets(
    type: Optional[str] = Query(None, description="Filter by dataset type"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    category: Optional[str] = Query(None, description="Filter by category"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all test datasets in the databank with optional filters"""
    databank_service = DatabankService(db)
    datasets = databank_service.list_datasets()
    
    # Apply filters if provided
    if type:
        datasets = [d for d in datasets if d.type == type]
    if tag:
        datasets = [d for d in datasets if tag in d.tags]
    if category:
        datasets = [d for d in datasets if category in d.categories]
        
    return datasets

@router.get("/datasets/{dataset_id}", response_model=TestDataset)
async def get_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get test dataset details"""
    databank_service = DatabankService(db)
    return databank_service.get_dataset(dataset_id)

@router.delete("/datasets/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a test dataset"""
    databank_service = DatabankService(db)
    databank_service.delete_dataset(dataset_id)
    return {"message": "Dataset deleted successfully"}

# Cookbook Management
@router.post("/cookbooks/upload", response_model=Cookbook)
async def upload_cookbook(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload a cookbook to the databank"""
    databank_service = DatabankService(db)
    return await databank_service.upload_cookbook(file)

@router.get("/cookbooks", response_model=List[Cookbook])
async def list_cookbooks(
    tag: Optional[str] = Query(None, description="Filter by tag"),
    category: Optional[str] = Query(None, description="Filter by category"),
    dataset: Optional[str] = Query(None, description="Filter by dataset used"),
    metric: Optional[str] = Query(None, description="Filter by metric used"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all cookbooks in the databank with optional filters"""
    databank_service = DatabankService(db)
    cookbooks = databank_service.list_cookbooks()
    
    # Apply filters if provided
    if tag:
        cookbooks = [c for c in cookbooks if tag in c.tags]
    if category:
        cookbooks = [c for c in cookbooks if category in c.categories]
    if dataset:
        cookbooks = [c for c in cookbooks if dataset in c.datasets]
    if metric:
        cookbooks = [c for c in cookbooks if metric in c.metrics]
        
    return cookbooks

@router.get("/cookbooks/{cookbook_id}", response_model=Cookbook)
async def get_cookbook(
    cookbook_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get cookbook details"""
    databank_service = DatabankService(db)
    return databank_service.get_cookbook(cookbook_id)

@router.delete("/cookbooks/{cookbook_id}")
async def delete_cookbook(
    cookbook_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a cookbook"""
    databank_service = DatabankService(db)
    databank_service.delete_cookbook(cookbook_id)
    return {"message": "Cookbook deleted successfully"}

# Cookbook Run Management
@router.post("/cookbooks/{cookbook_id}/runs", response_model=CookbookRun)
async def create_cookbook_run(
    cookbook_id: str,
    run: CookbookRunCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new cookbook run"""
    databank_service = DatabankService(db)
    return await databank_service.create_cookbook_run(cookbook_id, run.project_id, run.parameters)

@router.post("/cookbooks/{cookbook_id}/runs/{run_id}/execute")
async def execute_cookbook_run(
    cookbook_id: str,
    run_id: int,
    parameters: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Execute a cookbook run"""
    databank_service = DatabankService(db)
    return await databank_service.execute_cookbook_run(run_id, cookbook_id, parameters)

@router.get("/cookbooks/{cookbook_id}/runs", response_model=List[CookbookRun])
async def list_cookbook_runs(
    cookbook_id: str,
    project_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all runs for a specific cookbook"""
    databank_service = DatabankService(db)
    return databank_service.list_cookbook_runs(cookbook_id, project_id, skip, limit)

@router.get("/cookbooks/{cookbook_id}/runs/{run_id}", response_model=CookbookRun)
async def get_cookbook_run(
    cookbook_id: str,
    run_id: int,
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get details of a specific cookbook run"""
    databank_service = DatabankService(db)
    return databank_service.get_cookbook_run(run_id, cookbook_id, project_id)

@router.get("/cookbooks/{cookbook_id}/runs/{run_id}/results", response_model=List[RecipeResult])
async def get_cookbook_results(
    cookbook_id: str,
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get detailed results for a cookbook run"""
    databank_service = DatabankService(db)
    return databank_service.get_cookbook_results(cookbook_id, run_id)

# Attack Module Management
@router.post("/attack-modules/upload", response_model=AttackModule)
async def upload_attack_module(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload an attack module to the databank"""
    databank_service = DatabankService(db)
    return await databank_service.upload_attack_module(file)

@router.get("/attack-modules", response_model=List[AttackModule])
async def list_attack_modules(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all attack modules in the databank"""
    databank_service = DatabankService(db)
    return databank_service.list_attack_modules()

@router.get("/attack-modules/{module_name}", response_model=AttackModule)
async def get_attack_module(
    module_name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get attack module details"""
    databank_service = DatabankService(db)
    return databank_service.get_attack_module(module_name) 