from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session

from models.models import User, Project, Model, Dataset, Audit
from schemas.project import (
    ProjectCreate,
    ProjectUpdate,
    ProjectResponse,
    ProjectStatusUpdate
)
import logging

logger = logging.getLogger(__name__)

class ProjectService:
    def __init__(self, db: Session):
        self.db = db

    def create_project(self, project: ProjectCreate, current_user: User) -> ProjectResponse:
        """Create a new project"""
        try:
            db_project = Project(**project.dict(), user_id=current_user.id)
            self.db.add(db_project)
            self.db.commit()
            self.db.refresh(db_project)
            return ProjectResponse.from_orm(db_project)
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating project: {str(e)}")
            raise

    def get_project(self, project_id: int, current_user: User) -> Optional[ProjectResponse]:
        """Get a specific project"""
        try:
            project = self.db.query(Project).filter(
                Project.id == project_id,
                Project.user_id == current_user.id
            ).first()
            return ProjectResponse.from_orm(project) if project else None
        except Exception as e:
            logger.error(f"Error getting project: {str(e)}")
            raise

    def list_projects(self, current_user: User) -> List[ProjectResponse]:
        """List all projects for the current user"""
        try:
            projects = self.db.query(Project).filter(
                Project.user_id == current_user.id
            ).all()
            return [ProjectResponse.from_orm(project) for project in projects]
        except Exception as e:
            logger.error(f"Error listing projects: {str(e)}")
            raise

    def update_project(
        self, 
        project_id: int, 
        project: ProjectUpdate, 
        current_user: User
    ) -> Optional[ProjectResponse]:
        """Update a project"""
        try:
            db_project = self.db.query(Project).filter(
                Project.id == project_id,
                Project.user_id == current_user.id
            ).first()
            
            if not db_project:
                return None
                
            for field, value in project.dict(exclude_unset=True).items():
                setattr(db_project, field, value)
                
            self.db.commit()
            self.db.refresh(db_project)
            return ProjectResponse.from_orm(db_project)
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating project: {str(e)}")
            raise

    def delete_project(self, project_id: int, current_user: User) -> bool:
        """Delete a project"""
        try:
            project = self.db.query(Project).filter(
                Project.id == project_id,
                Project.user_id == current_user.id
            ).first()
            
            if not project:
                return False
                
            self.db.delete(project)
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting project: {str(e)}")
            raise

    def get_project_stats(self, project_id: int, current_user: User) -> Dict[str, Any]:
        """Get project statistics"""
        try:
            project = self.db.query(Project).filter(
                Project.id == project_id,
                Project.user_id == current_user.id
            ).first()
            
            if not project:
                return {}
                
            return {
                "total_models": self.db.query(Model).filter(Model.project_id == project_id).count(),
                "total_datasets": self.db.query(Dataset).filter(Dataset.project_id == project_id).count(),
                "total_audits": self.db.query(Audit).filter(Audit.project_id == project_id).count(),
                "last_updated": project.updated_at
            }
        except Exception as e:
            logger.error(f"Error getting project stats: {str(e)}")
            raise

    def update_project_status(
        self, 
        project_id: int, 
        status: ProjectStatusUpdate, 
        current_user: User
    ) -> Optional[ProjectResponse]:
        """Update project status"""
        try:
            db_project = self.db.query(Project).filter(
                Project.id == project_id,
                Project.user_id == current_user.id
            ).first()
            
            if not db_project:
                return None
                
            db_project.status = status.status
            self.db.commit()
            self.db.refresh(db_project)
            return ProjectResponse.from_orm(db_project)
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating project status: {str(e)}")
            raise 