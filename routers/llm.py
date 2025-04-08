from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging
from enum import Enum

from core.deps import get_db, get_current_user
from models.models import User
from schemas.enums import RedTeamingCategory, AttackMode
from schemas.llm import (
    LLMConnectorCreate,
    LLMConnectorResponse,
    LLMConnectorUpdate,
    LLMAuditCreate,
    LLMAuditResponse,
    ChatMessage,
    ChatResponse
)
from services.llm_service import LLMService
from utils.error_handlers import ProjectNotFoundError, ValidationError

router = APIRouter(
    prefix="/llm",
    tags=["llm"],
    responses={404: {"description": "Not found"}}
)

logger = logging.getLogger(__name__)

# LLM Connector Management
@router.get("/connectors", response_model=List[LLMConnectorResponse])
async def list_connectors(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all available LLM connectors"""
    try:
        llm_service = LLMService(db)
        connectors = llm_service.list_connectors(project_id, current_user)
        return connectors
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error listing connectors: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.post("/connectors", response_model=LLMConnectorResponse)
async def create_connector(
    project_id: int,
    provider: str,
    config: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new LLM connector"""
    try:
        llm_service = LLMService(db)
        connector = await llm_service.create_connector(project_id, current_user, provider, config)
        return connector
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating connector: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.get("/connectors/{connector_id}", response_model=LLMConnectorResponse)
async def get_connector(
    project_id: int,
    connector_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get connector details"""
    try:
        llm_service = LLMService(db)
        connector = llm_service.get_connector(project_id, connector_id, current_user)
        return connector
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting connector: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

# LLM Chat Operations
@router.post("/{project_id}/chat", response_model=ChatResponse)
async def chat_with_llm(
    project_id: int,
    connector_id: int,
    messages: List[ChatMessage],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Chat with LLM"""
    try:
        llm_service = LLMService(db)
        # Convert ChatMessage objects to dict format
        message_dicts = [
            {
                "role": msg.role,
                "content": msg.content,
                "name": msg.name if hasattr(msg, "name") else None
            }
            for msg in messages
        ]
        response = await llm_service.chat_with_llm(project_id, current_user, connector_id, message_dicts)
        return response
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error chatting with LLM: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

# LLM Audit Operations
@router.post("/{project_id}/audit/red-teaming", response_model=LLMAuditResponse)
async def run_red_teaming(
    project_id: int,
    connector_id: int,
    attack_modules: List[str],
    parameters: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Run red teaming audit"""
    try:
        llm_service = LLMService(db)
        audit = await llm_service.run_red_teaming(
            project_id,
            current_user,
            connector_id,
            attack_modules,
            parameters
        )
        return audit
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error running red teaming: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.post("/{project_id}/audit/benchmark", response_model=LLMAuditResponse)
async def run_benchmark(
    project_id: int,
    connector_id: int,
    cookbook_name: str,
    parameters: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Run benchmark tests"""
    try:
        llm_service = LLMService(db)
        audit = await llm_service.run_benchmark(
            project_id,
            current_user,
            connector_id,
            cookbook_name,
            parameters
        )
        return audit
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error running benchmark: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.get("/{project_id}/audit/{audit_id}/status", response_model=LLMAuditResponse)
async def get_audit_status(
    project_id: int,
    audit_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get audit status"""
    try:
        llm_service = LLMService(db)
        status = llm_service.get_audit_status(project_id, audit_id, current_user)
        return status
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting audit status: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.get("/{project_id}/audit/{audit_id}/report", response_model=LLMAuditResponse)
async def get_audit_report(
    project_id: int,
    audit_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get audit report"""
    try:
        llm_service = LLMService(db)
        report = llm_service.get_audit_report(project_id, audit_id, current_user)
        return report
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting audit report: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

# LLM Red Teaming Operations
@router.post("/{project_id}/audit/red-teaming/manual", response_model=LLMAuditResponse)
async def run_manual_red_teaming(
    project_id: int,
    connector_id: int,
    target_categories: List[RedTeamingCategory],
    parameters: Optional[Dict[str, Any]] = None,
    max_attempts: int = 10,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Run manual red teaming attack"""
    try:
        llm_service = LLMService(db)
        audit = await llm_service.run_red_teaming(
            project_id=project_id,
            user=current_user,
            connector_id=connector_id,
            attack_mode=AttackMode.MANUAL,
            target_categories=target_categories,
            parameters=parameters,
            max_attempts=max_attempts
        )
        return audit
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error running manual red teaming: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.post("/{project_id}/audit/red-teaming/targeted", response_model=LLMAuditResponse)
async def run_targeted_red_teaming(
    project_id: int,
    connector_id: int,
    target_categories: List[RedTeamingCategory],
    parameters: Optional[Dict[str, Any]] = None,
    max_attempts: int = 10,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Run targeted red teaming attack"""
    try:
        llm_service = LLMService(db)
        audit = await llm_service.run_red_teaming(
            project_id=project_id,
            user=current_user,
            connector_id=connector_id,
            attack_mode=AttackMode.TARGETED,
            target_categories=target_categories,
            parameters=parameters,
            max_attempts=max_attempts
        )
        return audit
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error running targeted red teaming: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.post("/{project_id}/audit/red-teaming/comprehensive", response_model=LLMAuditResponse)
async def run_comprehensive_red_teaming(
    project_id: int,
    connector_id: int,
    target_categories: List[RedTeamingCategory],
    parameters: Optional[Dict[str, Any]] = None,
    max_attempts: int = 10,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Run comprehensive red teaming attack"""
    try:
        llm_service = LLMService(db)
        audit = await llm_service.run_red_teaming(
            project_id=project_id,
            user=current_user,
            connector_id=connector_id,
            attack_mode=AttackMode.COMPREHENSIVE,
            target_categories=target_categories,
            parameters=parameters,
            max_attempts=max_attempts
        )
        return audit
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error running comprehensive red teaming: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.get("/{project_id}/audit/red-teaming/{audit_id}/progress", response_model=Dict[str, Any])
async def get_red_teaming_progress(
    project_id: int,
    audit_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get red teaming progress and results"""
    try:
        llm_service = LLMService(db)
        progress = await llm_service.get_red_teaming_progress(
            project_id=project_id,
            audit_id=audit_id,
            user=current_user
        )
        return progress
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting red teaming progress: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.get("/{project_id}/audit/red-teaming/{audit_id}/vulnerabilities", response_model=List[Dict[str, Any]])
async def get_red_teaming_vulnerabilities(
    project_id: int,
    audit_id: int,
    category: Optional[RedTeamingCategory] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get vulnerabilities found during red teaming"""
    try:
        llm_service = LLMService(db)
        vulnerabilities = await llm_service.get_red_teaming_vulnerabilities(
            project_id=project_id,
            audit_id=audit_id,
            category=category,
            user=current_user
        )
        return vulnerabilities
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting red teaming vulnerabilities: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.get("/{project_id}/audit/red-teaming/{audit_id}/metrics", response_model=Dict[str, Any])
async def get_red_teaming_metrics(
    project_id: int,
    audit_id: int,
    category: Optional[RedTeamingCategory] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get detailed metrics for red teaming audit"""
    try:
        llm_service = LLMService(db)
        metrics = await llm_service.get_red_teaming_metrics(
            project_id=project_id,
            audit_id=audit_id,
            category=category,
            user=current_user
        )
        return metrics
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting red teaming metrics: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.get("/{project_id}/audit/red-teaming/{audit_id}/metrics/visualization", response_model=Dict[str, Any])
async def get_red_teaming_visualization_data(
    project_id: int,
    audit_id: int,
    visualization_type: str,
    category: Optional[RedTeamingCategory] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get visualization data for red teaming metrics"""
    try:
        llm_service = LLMService(db)
        visualization_data = await llm_service.get_red_teaming_visualization_data(
            project_id=project_id,
            audit_id=audit_id,
            visualization_type=visualization_type,
            category=category,
            user=current_user
        )
        return visualization_data
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting red teaming visualization data: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.get("/{project_id}/audit/red-teaming/{audit_id}/metrics/performance", response_model=Dict[str, Any])
async def get_red_teaming_performance_metrics(
    project_id: int,
    audit_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get performance metrics for red teaming audit"""
    try:
        llm_service = LLMService(db)
        performance_metrics = await llm_service.get_red_teaming_performance_metrics(
            project_id=project_id,
            audit_id=audit_id,
            user=current_user
        )
        return performance_metrics
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting red teaming performance metrics: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.get("/{project_id}/audit/red-teaming/{audit_id}/metrics/category", response_model=Dict[str, Any])
async def get_category_specific_metrics(
    project_id: int,
    audit_id: int,
    category: RedTeamingCategory,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get category-specific metrics for red teaming audit"""
    try:
        llm_service = LLMService(db)
        category_metrics = await llm_service.get_category_specific_metrics(
            project_id=project_id,
            audit_id=audit_id,
            category=category,
            user=current_user
        )
        return category_metrics
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting category-specific metrics: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error") 