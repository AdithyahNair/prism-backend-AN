from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from fastapi import UploadFile, HTTPException, status
import logging
from datetime import datetime
from pathlib import Path
import json

from models.models import User, TestDataset, Cookbook
from utils.error_handlers import ValidationError
from utils.file_handler import save_file, load_json_file

logger = logging.getLogger(__name__)

class DatabankService:
    def __init__(self, db: Session):
        self.db = db
        self.moonshot_data = Path("moonshot-data")
        self.datasets_dir = self.moonshot_data / "datasets"
        self.cookbooks_dir = self.moonshot_data / "cookbooks"
        self.attack_modules_dir = self.moonshot_data / "attack-modules"

    async def upload_dataset(self, file: UploadFile, user: User) -> TestDataset:
        """Upload a test dataset to the databank"""
        try:
            if not file.filename.endswith('.json'):
                raise ValidationError("Dataset must be a JSON file")
            
            # Save file to moonshot-data/datasets
            file_path = await save_file(file, str(self.datasets_dir))
            
            # Load and validate dataset format
            dataset_data = load_json_file(file_path)
            self._validate_dataset_format(dataset_data)
            
            # Create dataset record
            dataset = TestDataset(
                name=file.filename[:-5],  # Remove .json extension
                description=dataset_data.get("description", ""),
                file_path=str(file_path),
                type=dataset_data.get("type", "text"),
                format="json",
                size=file_path.stat().st_size,
                tags=dataset_data.get("tags", []),
                categories=dataset_data.get("categories", []),
                dataset_schema=dataset_data.get("schema")
            )
            
            self.db.add(dataset)
            self.db.commit()
            self.db.refresh(dataset)
            
            return dataset
        except Exception as e:
            logger.error(f"Error uploading dataset: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload dataset: {str(e)}"
            )

    def _validate_dataset_format(self, data: Dict[str, Any]) -> None:
        """Validate dataset format"""
        required_fields = ["type", "data"]
        for field in required_fields:
            if field not in data:
                raise ValidationError(f"Missing required field: {field}")
        
        valid_types = ["text", "image", "tabular"]
        if data["type"] not in valid_types:
            raise ValidationError(f"Invalid dataset type. Must be one of: {valid_types}")

    def list_datasets(self) -> List[TestDataset]:
        """List all test datasets in the databank"""
        return self.db.query(TestDataset).all()

    def get_dataset(self, dataset_id: str) -> TestDataset:
        """Get test dataset details"""
        dataset = self.db.query(TestDataset).filter(TestDataset.name == dataset_id).first()
        if not dataset:
            raise ValidationError(f"Dataset {dataset_id} not found")
        return dataset

    def delete_dataset(self, dataset_id: str) -> None:
        """Delete a test dataset"""
        dataset = self.get_dataset(dataset_id)
        try:
            # Delete file
            file_path = Path(dataset.file_path)
            if file_path.exists():
                file_path.unlink()
            
            # Delete database record
            self.db.delete(dataset)
            self.db.commit()
        except Exception as e:
            logger.error(f"Error deleting dataset: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete dataset: {str(e)}"
            )

    async def upload_cookbook(self, file: UploadFile) -> Cookbook:
        """Upload a cookbook to the databank"""
        try:
            if not file.filename.endswith('.json'):
                raise ValidationError("Cookbook must be a JSON file")
            
            # Save file to moonshot-data/cookbooks
            file_path = await save_file(file, str(self.cookbooks_dir))
            
            # Load and validate cookbook format
            cookbook_data = load_json_file(file_path)
            self._validate_cookbook_format(cookbook_data)
            
            # Create cookbook record
            cookbook = Cookbook(
                name=file.filename[:-5],  # Remove .json extension
                description=cookbook_data.get("description", ""),
                file_path=str(file_path),
                version=cookbook_data.get("version", "1.0.0"),
                status="active",
                tags=cookbook_data.get("tags", []),
                categories=cookbook_data.get("categories", []),
                datasets=cookbook_data.get("datasets", []),
                metrics=cookbook_data.get("metrics", [])
            )
            
            self.db.add(cookbook)
            self.db.commit()
            self.db.refresh(cookbook)
            
            return cookbook
        except Exception as e:
            logger.error(f"Error uploading cookbook: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload cookbook: {str(e)}"
            )

    def _validate_cookbook_format(self, data: Dict[str, Any]) -> None:
        """Validate cookbook format"""
        required_fields = ["version", "recipes"]
        for field in required_fields:
            if field not in data:
                raise ValidationError(f"Missing required field: {field}")
        
        if not isinstance(data["recipes"], list):
            raise ValidationError("Recipes must be a list")

    def list_cookbooks(self) -> List[Cookbook]:
        """List all cookbooks in the databank"""
        return self.db.query(Cookbook).all()

    def get_cookbook(self, cookbook_id: str) -> Cookbook:
        """Get cookbook details"""
        cookbook = self.db.query(Cookbook).filter(Cookbook.name == cookbook_id).first()
        if not cookbook:
            raise ValidationError(f"Cookbook {cookbook_id} not found")
        return cookbook

    def delete_cookbook(self, cookbook_id: str) -> None:
        """Delete a cookbook"""
        cookbook = self.get_cookbook(cookbook_id)
        try:
            # Delete file
            file_path = Path(cookbook.file_path)
            if file_path.exists():
                file_path.unlink()
            
            # Delete database record
            self.db.delete(cookbook)
            self.db.commit()
        except Exception as e:
            logger.error(f"Error deleting cookbook: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete cookbook: {str(e)}"
            )

    async def upload_attack_module(self, file: UploadFile) -> Dict[str, Any]:
        """Upload an attack module to the databank"""
        try:
            if not file.filename.endswith('.py'):
                raise ValidationError("Attack module must be a Python file")
            
            # Save file to moonshot-data/attack-modules
            file_path = await save_file(file, str(self.attack_modules_dir))
            
            # Update config file
            config_file = self.attack_modules_dir / "attack_modules_config.json"
            config = load_json_file(config_file) if config_file.exists() else {}
            
            module_name = file.filename[:-3]  # Remove .py extension
            config[module_name] = {
                "name": module_name,
                "file": file.filename,
                "enabled": True
            }
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            return {
                "name": file.filename,
                "path": file_path,
                "config": config[module_name]
            }
        except ValidationError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            logger.error(f"Error uploading attack module: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload attack module: {str(e)}"
            )

    def list_attack_modules(self) -> List[Dict[str, Any]]:
        """List all attack modules in the databank"""
        try:
            config_file = self.attack_modules_dir / "attack_modules_config.json"
            config = load_json_file(config_file) if config_file.exists() else {}
            
            modules = []
            for file in self.attack_modules_dir.glob("*.py"):
                if file.name == "__init__.py":
                    continue
                    
                module_name = file.stem
                module_config = config.get(module_name, {
                    "name": module_name,
                    "file": file.name,
                    "enabled": True
                })
                
                modules.append({
                    "name": file.name,
                    "path": str(file),
                    "size": file.stat().st_size,
                    "modified": datetime.fromtimestamp(file.stat().st_mtime),
                    "config": module_config
                })
            return modules
        except Exception as e:
            logger.error(f"Error listing attack modules: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list attack modules: {str(e)}"
            )

    def get_attack_module(self, module_name: str) -> Dict[str, Any]:
        """Get attack module details"""
        try:
            file = self.attack_modules_dir / f"{module_name}.py"
            if not file.exists():
                raise ValidationError(f"Attack module {module_name} not found")
            
            config_file = self.attack_modules_dir / "attack_modules_config.json"
            config = load_json_file(config_file) if config_file.exists() else {}
            module_config = config.get(module_name, {
                "name": module_name,
                "file": file.name,
                "enabled": True
            })
            
            return {
                "name": file.name,
                "path": str(file),
                "size": file.stat().st_size,
                "modified": datetime.fromtimestamp(file.stat().st_mtime),
                "config": module_config
            }
        except ValidationError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
        except Exception as e:
            logger.error(f"Error getting attack module: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get attack module: {str(e)}"
            ) 