from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
import logging
import os
import json
from datetime import datetime
from pathlib import Path

from models.models import Project, LLMConnector, Audit, User
from utils.error_handlers import ProjectNotFoundError, ValidationError
from utils.file_handler import load_json_file
from schemas.llm import (
    LLMConnectorCreate,
    LLMConnectorUpdate,
    LLMConnectorResponse,
    ChatMessage,
    ChatResponse,
    LLMAuditCreate,
    LLMAuditResponse,
    RedTeamingCreate,
    RedTeamingResponse,
    RedTeamingProgress,
    RedTeamingVulnerabilities,
    RedTeamingMetrics,
    RedTeamingVisualization,
    RedTeamingPerformance,
    RedTeamingCategoryMetrics
)
from .red_teaming_service import RedTeamingService
from .visualization_service import VisualizationService

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, db: Session):
        self.db = db
        self.base_dir = Path("moonshot-data")
        self.connectors_dir = self.base_dir / "connectors"
        self.cookbooks_dir = self.base_dir / "cookbooks"
        self.attack_modules_dir = self.base_dir / "attack_modules"
        self.test_cases_dir = self.base_dir / "test_cases"
        
        # Initialize sub-services
        self.red_teaming_service = RedTeamingService(db)
        self.visualization_service = VisualizationService()

    def _get_project(self, project_id: int, user: User) -> Project:
        """Get project and verify access"""
        project = self.db.query(Project).filter(
            Project.id == project_id,
            Project.user_id == user.id
        ).first()
        if not project:
            raise ProjectNotFoundError(f"Project {project_id} not found")
        return project

    def _load_connector_module(self, connector_name: str) -> Dict[str, Any]:
        """Load connector module configuration"""
        try:
            connector_file = self.connectors_dir / f"{connector_name}-connector.py"
            if not connector_file.exists():
                raise ValidationError(f"Connector {connector_name} not found")
            
            # Load connector configuration
            config_file = self.connectors_dir / "connectors_config.json"
            config = load_json_file(config_file)
            
            if connector_name not in config:
                raise ValidationError(f"Connector {connector_name} not configured")
            
            connector_config = config[connector_name]
            
            # Validate required fields
            required_fields = ["model", "max_tokens", "temperature"]
            missing_fields = [field for field in required_fields if field not in connector_config]
            if missing_fields:
                raise ValidationError(f"Missing required fields in connector config: {', '.join(missing_fields)}")
            
            return connector_config
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in connector config: {str(e)}")
            raise ValidationError("Invalid connector configuration format")
        except Exception as e:
            logger.error(f"Error loading connector module {connector_name}: {str(e)}")
            raise ValidationError(f"Failed to load connector {connector_name}: {str(e)}")

    def _load_cookbook(self, cookbook_name: str) -> Dict[str, Any]:
        """Load cookbook configuration"""
        try:
            cookbook_file = self.cookbooks_dir / f"{cookbook_name}.json"
            if not cookbook_file.exists():
                raise ValidationError(f"Cookbook {cookbook_name} not found")
            
            return load_json_file(cookbook_file)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in cookbook {cookbook_name}: {str(e)}")
            raise ValidationError(f"Invalid cookbook format for {cookbook_name}")
        except Exception as e:
            logger.error(f"Error loading cookbook {cookbook_name}: {str(e)}")
            raise ValidationError(f"Failed to load cookbook {cookbook_name}: {str(e)}")

    def _load_attack_module(self, module_name: str) -> Dict[str, Any]:
        """Load attack module configuration"""
        try:
            module_file = self.attack_modules_dir / f"{module_name}.py"
            if not module_file.exists():
                raise ValidationError(f"Attack module {module_name} not found")
            
            config_file = self.attack_modules_dir / "attack_modules_config.json"
            config = load_json_file(config_file)
            
            if module_name not in config:
                raise ValidationError(f"Attack module {module_name} not configured")
            
            return config[module_name]
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in attack module config: {str(e)}")
            raise ValidationError("Invalid attack module configuration format")
        except Exception as e:
            logger.error(f"Error loading attack module {module_name}: {str(e)}")
            raise ValidationError(f"Failed to load attack module {module_name}: {str(e)}")

    async def create_connector(self, connector: LLMConnectorCreate) -> LLMConnectorResponse:
        """Create a new LLM connector"""
        try:
            # Validate connector configuration
            await self._validate_connector_config(connector)
            
            # Create database record
            db_connector = LLMConnector(
                name=connector.name,
                provider=connector.provider,
                api_key=connector.api_key,
                config=connector.config,
                created_at=datetime.utcnow()
            )
            
            self.db.add(db_connector)
            self.db.commit()
            self.db.refresh(db_connector)
            
            return LLMConnectorResponse.from_orm(db_connector)
            
        except Exception as e:
            logger.error(f"Error creating connector: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create connector: {str(e)}"
            )

    async def list_connectors(self) -> List[LLMConnectorResponse]:
        """List all available LLM connectors"""
        try:
            connectors = self.db.query(LLMConnector).all()
            return [LLMConnectorResponse.from_orm(connector) for connector in connectors]
        except Exception as e:
            logger.error(f"Error listing connectors: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list connectors: {str(e)}"
            )

    async def get_connector(self, connector_id: int) -> LLMConnectorResponse:
        """Get a specific LLM connector"""
        try:
            connector = self.db.query(LLMConnector).filter(LLMConnector.id == connector_id).first()
            if not connector:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Connector not found"
                )
            return LLMConnectorResponse.from_orm(connector)
        except Exception as e:
            logger.error(f"Error getting connector: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get connector: {str(e)}"
            )

    def _validate_messages(self, messages: List[ChatMessage]) -> None:
        """Validate chat messages"""
        if not messages:
            raise ValidationError("Messages list cannot be empty")
            
        for msg in messages:
            if not msg.role in ["user", "assistant", "system"]:
                raise ValidationError(f"Invalid message role: {msg.role}")
            if not msg.content:
                raise ValidationError("Message content cannot be empty")
            if msg.role == "system" and not msg.name:
                raise ValidationError("System messages must have a name")

    async def chat_with_llm(
        self,
        connector_id: int,
        messages: List[ChatMessage]
    ) -> ChatResponse:
        """Chat with an LLM using a specific connector"""
        try:
            # Get connector from database
            connector = await self.get_connector(connector_id)
            
            # Format messages for the connector
            formatted_messages = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "name": msg.name if msg.role == "system" else None
                }
                for msg in messages
            ]
            
            # Get response using the connector's configuration
            response = await self._get_llm_response(
                provider=connector.provider,
                api_key=connector.api_key,
                messages=formatted_messages,
                **connector.config
            )
            
            return ChatResponse(
                content=response["content"],
                role=response["role"],
                usage=response.get("usage", {})
            )
            
        except Exception as e:
            logger.error(f"Error in chat with LLM: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to chat with LLM: {str(e)}"
            )

    async def _get_llm_response(
        self,
        provider: str,
        api_key: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Get response from LLM based on provider"""
        try:
            if provider == "OPENAI":
                from connectors.openai_connector import OpenAIConnector
                connector = OpenAIConnector(api_key=api_key)
                return await connector.chat(messages=messages, **kwargs)
            elif provider == "AZURE":
                from connectors.azure_connector import AzureConnector
                connector = AzureConnector(api_key=api_key)
                return await connector.chat(messages=messages, **kwargs)
            elif provider == "GEMINI":
                from connectors.gemini_connector import GeminiConnector
                connector = GeminiConnector(api_key=api_key)
                return await connector.chat(messages=messages, **kwargs)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get LLM response: {str(e)}"
            )

    async def run_red_teaming(
        self,
        project_id: int,
        user: Any,
        connector_id: int,
        attack_mode: str,
        target_categories: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        max_attempts: int = 3
    ) -> RedTeamingResponse:
        """Run red teaming operations"""
        try:
            # Create red teaming audit
            audit = await self.red_teaming_service.create_red_teaming(
                project_id=project_id,
                user=user,
                connector_id=connector_id,
                attack_mode=attack_mode,
                target_categories=target_categories,
                parameters=parameters,
                max_attempts=max_attempts
            )
            
            return RedTeamingResponse.from_orm(audit)
            
        except Exception as e:
            logger.error(f"Error in red teaming: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to run red teaming: {str(e)}"
            )

    async def get_red_teaming_progress(
        self,
        project_id: int,
        audit_id: int
    ) -> RedTeamingProgress:
        """Get progress of red teaming operations"""
        try:
            progress = await self.red_teaming_service.get_progress(
                project_id=project_id,
                audit_id=audit_id
            )
            return RedTeamingProgress(**progress)
        except Exception as e:
            logger.error(f"Error getting red teaming progress: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get red teaming progress: {str(e)}"
            )

    async def get_red_teaming_vulnerabilities(
        self,
        project_id: int,
        audit_id: int
    ) -> RedTeamingVulnerabilities:
        """Get vulnerabilities found during red teaming"""
        try:
            vulnerabilities = await self.red_teaming_service.get_vulnerabilities(
                project_id=project_id,
                audit_id=audit_id
            )
            return RedTeamingVulnerabilities(**vulnerabilities)
        except Exception as e:
            logger.error(f"Error getting red teaming vulnerabilities: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get red teaming vulnerabilities: {str(e)}"
            )

    async def get_red_teaming_metrics(
        self,
        project_id: int,
        audit_id: int
    ) -> RedTeamingMetrics:
        """Get metrics from red teaming operations"""
        try:
            metrics = await self.red_teaming_service.get_metrics(
                project_id=project_id,
                audit_id=audit_id
            )
            return RedTeamingMetrics(**metrics)
        except Exception as e:
            logger.error(f"Error getting red teaming metrics: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get red teaming metrics: {str(e)}"
            )

    async def get_red_teaming_visualization(
        self,
        project_id: int,
        audit_id: int
    ) -> RedTeamingVisualization:
        """Get visualization data for red teaming results"""
        try:
            metrics = await self.get_red_teaming_metrics(project_id, audit_id)
            visualization_data = await self.visualization_service.plot_performance_metrics(
                y_true=metrics.true_labels,
                y_pred=metrics.predicted_labels,
                y_pred_proba=metrics.prediction_probabilities
            )
            return RedTeamingVisualization(**visualization_data)
        except Exception as e:
            logger.error(f"Error getting red teaming visualization: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get red teaming visualization: {str(e)}"
            )

    async def get_red_teaming_performance(
        self,
        project_id: int,
        audit_id: int
    ) -> RedTeamingPerformance:
        """Get performance metrics for red teaming operations"""
        try:
            performance = await self.red_teaming_service.get_performance_metrics(
                project_id=project_id,
                audit_id=audit_id
            )
            return RedTeamingPerformance(**performance)
        except Exception as e:
            logger.error(f"Error getting red teaming performance: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get red teaming performance: {str(e)}"
            )

    async def get_red_teaming_category_metrics(
        self,
        project_id: int,
        audit_id: int
    ) -> RedTeamingCategoryMetrics:
        """Get category-specific metrics for red teaming operations"""
        try:
            category_metrics = await self.red_teaming_service.get_category_metrics(
                project_id=project_id,
                audit_id=audit_id
            )
            return RedTeamingCategoryMetrics(**category_metrics)
        except Exception as e:
            logger.error(f"Error getting red teaming category metrics: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get red teaming category metrics: {str(e)}"
            )

    async def _validate_connector_config(self, connector: LLMConnectorCreate) -> None:
        """Validate connector configuration"""
        try:
            # Validate provider
            if connector.provider not in ["OPENAI", "AZURE", "GEMINI"]:
                raise ValueError(f"Unsupported provider: {connector.provider}")
            
            # Validate API key
            if not connector.api_key or len(connector.api_key) < 32:
                raise ValueError("Invalid API key format")
            
            # Validate config if provided
            if connector.config:
                required_fields = ["model", "max_tokens", "temperature"]
                for field in required_fields:
                    if field not in connector.config:
                        raise ValueError(f"Missing required config field: {field}")
                        
        except Exception as e:
            logger.error(f"Error validating connector config: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid connector configuration: {str(e)}"
            )

    def _validate_connector_name(self, name: str) -> bool:
        """Validate connector name format"""
        return bool(name and len(name) <= 100 and name.isalnum())

    def _validate_provider(self, provider: str) -> bool:
        """Validate provider name"""
        return provider in ["OPENAI", "AZURE", "GEMINI"]

    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key format"""
        return bool(api_key and len(api_key) >= 32)

    async def run_benchmark(
        self,
        project_id: int,
        user: User,
        connector_id: int,
        cookbook_name: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> LLMAuditResponse:
        """Run benchmark tests"""
        project = self._get_project(project_id, user)
        connector = self.get_connector(project_id, connector_id, user)
        
        # Load cookbook
        cookbook = self._load_cookbook(cookbook_name)
        
        # Create audit record
        audit = Audit(
            project_id=project_id,
            audit_type="LLM",
            status="running",
            config={
                "audit_type": "benchmark",
                "cookbook": cookbook_name,
                "custom_parameters": parameters,
                "created_by": user.id,
                "connector_id": connector_id
            },
            created_at=datetime.utcnow(),
            started_at=datetime.utcnow()
        )
        
        self.db.add(audit)
        self.db.commit()
        self.db.refresh(audit)
        
        # TODO: Implement async benchmark logic
        
        return LLMAuditResponse.from_orm(audit)

    def get_audit_status(self, project_id: int, audit_id: int, user: User) -> LLMAuditResponse:
        """Get audit status"""
        project = self._get_project(project_id, user)
        
        audit = self.db.query(Audit).filter(
            Audit.id == audit_id,
            Audit.project_id == project_id
        ).first()
        
        if not audit:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audit {audit_id} not found"
            )
            
        return LLMAuditResponse.from_orm(audit)

    def get_audit_report(self, project_id: int, audit_id: int, user: User) -> LLMAuditResponse:
        """Get audit report"""
        project = self._get_project(project_id, user)
        
        audit = self.db.query(Audit).filter(
            Audit.id == audit_id,
            Audit.project_id == project_id
        ).first()
        
        if not audit:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audit {audit_id} not found"
            )
            
        return LLMAuditResponse.from_orm(audit) 

    async def update_connector(
        self,
        project_id: int,
        connector_id: int,
        user: User,
        connector: LLMConnectorUpdate
    ) -> LLMConnectorResponse:
        """Update an LLM connector"""
        project = self._get_project(project_id, user)
        db_connector = self.get_connector(project_id, connector_id, user)
        
        update_data = connector.dict(exclude_unset=True)
        
        # Validate updates if provided
        if "name" in update_data and not self._validate_connector_name(update_data["name"]):
            raise ValidationError("Invalid connector name format")
            
        if "provider" in update_data:
            if not self._validate_provider(update_data["provider"]):
                raise ValidationError(f"Provider must be one of: OPENAI, AZURE, GEMINI")
            # Validate provider and load module
            self._load_connector_module(update_data["provider"])
            
        if "api_key" in update_data and not self._validate_api_key(update_data["api_key"]):
            raise ValidationError("Invalid API key format")
        
        # Update fields
        for field, value in update_data.items():
            setattr(db_connector, field, value)
        
        db_connector.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(db_connector)
        
        return LLMConnectorResponse.from_orm(db_connector) 
