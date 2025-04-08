from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import List, Dict, Any, Optional
import logging

from core.deps import get_db, get_current_user
from services.database import (
    create_project as db_create_project,
    get_projects_by_user_id,
    get_project_by_id,
    update_project as db_update_project,
    delete_project as db_delete_project
)
from pydantic import BaseModel

router = APIRouter(
    prefix="/projects",
    tags=["projects"],
    responses={404: {"description": "Not found"}}
)

logger = logging.getLogger(__name__)

# Define Pydantic models for request/response
class ProjectBase(BaseModel):
    name: str
    description: str
    project_type: str
    status: str

class ProjectCreate(ProjectBase):
    pass

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    project_type: Optional[str] = None
    status: Optional[str] = None

class ProjectResponse(ProjectBase):
    id: int
    user_id: int
    created_at: str
    updated_at: Optional[str] = None

    class Config:
        from_attributes = True

@router.post("/create", response_model=ProjectResponse)
async def create_project(
    project: ProjectCreate,
    current_user: Dict = Depends(get_current_user)
):
    """Create a new project"""
    try:
        logger.info(f"Creating project: {project.name} for user: {current_user['id']}")
        
        project_data = {
            "name": project.name,
            "description": project.description,
            "project_type": project.project_type,
            "status": project.status,
            "user_id": current_user["id"]
        }
        
        new_project = await db_create_project(project_data)
        return new_project
    except Exception as e:
        logger.error(f"Error creating project: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/list", response_model=List[ProjectResponse])
async def list_projects(
    current_user: Dict = Depends(get_current_user)
):
    """List all projects for the current user"""
    try:
        logger.info(f"Listing projects for user: {current_user['id']}")
        projects = await get_projects_by_user_id(current_user["id"])
        return projects
    except Exception as e:
        logger.error(f"Error listing projects: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: int,
    current_user: Dict = Depends(get_current_user)
):
    """Get a specific project"""
    try:
        logger.info(f"Getting project: {project_id} for user: {current_user['id']}")
        project = await get_project_by_id(project_id)
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project with ID {project_id} not found"
            )
            
        # Check if the project belongs to the current user
        if project["user_id"] != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this project"
            )
            
        return project
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error getting project: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: int,
    project: ProjectUpdate,
    current_user: Dict = Depends(get_current_user)
):
    """Update a project"""
    try:
        logger.info(f"Updating project: {project_id}")
        
        # Check if project exists and belongs to user
        existing_project = await get_project_by_id(project_id)
        if not existing_project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project with ID {project_id} not found"
            )
            
        if existing_project["user_id"] != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this project"
            )
        
        # Build update data - only include fields that are provided
        update_data = {}
        if project.name is not None:
            update_data["name"] = project.name
        if project.description is not None:
            update_data["description"] = project.description
        if project.project_type is not None:
            update_data["project_type"] = project.project_type
        if project.status is not None:
            update_data["status"] = project.status
            
        updated_project = await db_update_project(project_id, update_data)
        return updated_project
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error updating project: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.delete("/{project_id}")
async def delete_project(
    project_id: int,
    current_user: Dict = Depends(get_current_user)
):
    """Delete a project"""
    try:
        logger.info(f"Deleting project: {project_id}")
        
        # Check if project exists and belongs to user
        existing_project = await get_project_by_id(project_id)
        if not existing_project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project with ID {project_id} not found"
            )
            
        if existing_project["user_id"] != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this project"
            )
        
        await db_delete_project(project_id)
        return {"message": "Project deleted successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deleting project: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        ) 