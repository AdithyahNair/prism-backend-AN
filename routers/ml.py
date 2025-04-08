import os
os.environ["USE_GCS"] = "true"
os.environ["GCS_BUCKET_NAME"] = "chatbotstorages"

from fastapi import APIRouter, Depends, HTTPException, status, File, Form, UploadFile
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging

from core.deps import get_db, get_current_user
from models.models import User
from schemas.ml import ModelCreate, ModelResponse
from schemas.dataset import DatasetCreate, DatasetResponse
from schemas.audit import AuditCreate, AuditResponse
from schemas.report import ReportResponse
from services.ml_service import MLService
from utils.error_handlers import ProjectNotFoundError, ValidationError
from fastapi.responses import FileResponse

router = APIRouter(
    prefix="/ml",
    tags=["ml"],
    responses={404: {"description": "Not found"}}
)

logger = logging.getLogger(__name__)

# Model Management
@router.post("/{project_id}/models/upload", response_model=ModelResponse)
async def upload_model(
    project_id: int,
    name: str = Form(...),
    model_type: str = Form(...),
    version: str = Form(...),
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_user)
):
    """Upload a model file"""
    try:
        logger.info(f"Received upload request for project {project_id}")
        logger.info(f"File details: {file.filename if file else 'No file'}")
        logger.info(f"File type: {type(file)}")
        logger.info(f"Current user: {current_user}")
        
        if not file:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided in the request"
            )
            
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File has no name"
            )

        if not file.content_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File has no content type"
            )

        # Log form data
        logger.info(f"Form data: name={name}, model_type={model_type}, version={version}, description={description}")

        # Ensure file is properly initialized
        if not hasattr(file, 'file'):
            logger.error("File object is not properly initialized")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File object is not properly initialized"
            )

        ml_service = MLService(db)
        model = await ml_service.upload_model(
            name=name,
            model_type=model_type,
            version=version,
            file=file,
            project_id=project_id,
            description=description,
            current_user=current_user
        )

        # After successful upload
        if os.getenv("USE_GCS") == "true" and model.file_path.startswith("gs://"):
            # Extract path parts
            logger.info(f"Verifying uploaded file: {model.file_path}")
            
            # No verification needed as it was just uploaded
            # The error was trying to verify a path that doesn't match where the file was uploaded
            # Just return the model without additional verification
            
        return model
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error uploading model: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/{project_id}/models/list", response_model=List[ModelResponse])
async def list_models(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_user)
):
    """List all models in a project"""
    try:
        ml_service = MLService(db)
        models = ml_service.list_models(project_id, current_user)
        return models
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

# Dataset Management
@router.post("/{project_id}/datasets/upload", response_model=DatasetResponse)
async def upload_dataset(
    project_id: int,
    file: UploadFile = File(...),
    dataset_type: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload a dataset file"""
    try:
        logger.info(f"Starting dataset upload for project {project_id}")
        ml_service = MLService(db)
        
        # Upload dataset
        dataset = await ml_service.upload_dataset(
            project_id=project_id,
            file=file,
            user=current_user,
            dataset_type=dataset_type
        )
        
        # No verification needed as we've already handled this in the service
        logger.info(f"Dataset uploaded successfully: {dataset.name}")
        return dataset
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        logger.error(f"HTTP exception in dataset upload: {e.detail}")
        raise
    except ValidationError as e:
        # Convert validation errors to HTTP exceptions
        logger.error(f"Validation error in dataset upload: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error uploading dataset: {e.message}"
        )
    except Exception as e:
        # Log and convert any other exceptions
        logger.error(f"Error uploading dataset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading dataset: {str(e)}"
        )

# Dataset Management
@router.post("/{project_id}/datasets/generate", response_model=DatasetResponse)
async def generate_dataset(
    project_id: int,
    model_type: str,
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 2,
    imbalance_ratio: float = 1.0,
    noise: float = 0.1,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_user)
):
    """Upload a dataset file"""
    try:
        ml_service = MLService(db)
        dataset = await ml_service.generate_synthetic_data(
            project_id=project_id,
            model_type=model_type,
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            imbalance_ratio=imbalance_ratio,
            noise=noise,
            user=current_user
        )
        return dataset
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error uploading dataset: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.get("/{project_id}/datasets/list", response_model=List[DatasetResponse])
async def list_datasets(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_user)
):
    """List all datasets in a project"""
    try:
        ml_service = MLService(db)
        datasets = ml_service.list_datasets(project_id, current_user)
        return datasets
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

# Audit Operations
@router.post("/{project_id}/audit/performance", response_model=AuditResponse)
async def run_performance_audit(
    project_id: int,
    model_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_user)
):
    """Run performance audit"""
    try:
        ml_service = MLService(db)
        audit = await ml_service.run_performance_audit(
            project_id, 
            current_user,
            model_id=model_id,
            dataset_id=dataset_id
        )
        return audit
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error running performance audit: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.post("/{project_id}/audit/fairness", response_model=AuditResponse)
async def run_fairness_audit(
    project_id: int,
    model_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_user)
):
    """Run fairness and bias audit"""
    try:
        ml_service = MLService(db)
        audit = await ml_service.run_fairness_audit(
            project_id, 
            current_user,
            model_id=model_id,
            dataset_id=dataset_id
        )
        return audit
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error running fairness audit: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.post("/{project_id}/audit/explainability", response_model=AuditResponse)
async def run_explainability_audit(
    project_id: int,
    model_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_user)
):
    """Run explainability audit"""
    try:
        ml_service = MLService(db)
        audit = await ml_service.run_explainability_audit(
            project_id, 
            current_user,
            model_id=model_id,
            dataset_id=dataset_id
        )
        return audit
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error running explainability audit: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.post("/{project_id}/audit/drift", response_model=AuditResponse)
async def run_drift_analysis(
    project_id: int,
    model_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_user)
):
    """Run drift analysis"""
    try:
        ml_service = MLService(db)
        audit = await ml_service.run_drift_audit(
            project_id, 
            current_user,
            model_id=model_id,
            dataset_id=dataset_id
        )
        return audit
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error running drift analysis: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.get("/{project_id}/audit/{audit_id}/report", response_model=Dict[str, Any])
async def get_audit_report(
    project_id: int,
    audit_id: int,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_user)
):
    """Get audit report"""
    try:
        ml_service = MLService(db)
        report = ml_service.get_audit_report(project_id, audit_id, current_user)
        return report
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting audit report: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.post("/{project_id}/reports/generate", response_model=ReportResponse)
async def generate_consolidated_report(
    project_id: int,
    model_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_user)
):
    """Generate a consolidated report for all audits"""
    try:
        ml_service = MLService(db)
        report = await ml_service.generate_consolidated_report(
            project_id,
            current_user,
            model_id=model_id,
            dataset_id=dataset_id
        )
        return report
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating consolidated report: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error") 

@router.get("/performance/{project_id}/{model_id}/{model_version}")
async def get_performance_metrics(
    project_id: int,
    model_id: int,
    model_version: str,
    current_user: Any = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get performance metrics for a model"""
    ml_service = MLService(db)
    return ml_service.get_performance_metrics(project_id, model_id, model_version)

@router.get("/fairness/{project_id}/{model_id}/{model_version}")
async def get_fairness_metrics(
    project_id: int,
    model_id: int,
    model_version: str,
    current_user: Any = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get fairness metrics for a model"""
    ml_service = MLService(db)
    return ml_service.get_fairness_metrics(project_id, model_id, model_version)

@router.get("/explainability/{project_id}/{model_id}/{model_version}")
async def get_explainability_metrics(
    project_id: int,
    model_id: int,
    model_version: str,
    current_user: Any = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get explainability metrics for a model"""
    ml_service = MLService(db)
    return ml_service.get_explainability_metrics(project_id, model_id, model_version)

@router.get("/drift/{project_id}/{model_id}/{model_version}")
async def get_drift_metrics(
    project_id: int,
    model_id: int,
    model_version: str,
    current_user: Any = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get drift metrics for a model"""
    ml_service = MLService(db)
    return ml_service.get_drift_metrics(project_id, model_id, model_version)


@router.get("/download/{project_id}/{model_id}/{model_version}", response_class=FileResponse)
async def download_pdf(
    project_id: int, 
    model_id: int, 
    model_version: str,
    db: Session = Depends(get_db)
):
    ml_service = MLService(db)
    file_path = ml_service.get_pdf_file_path(project_id, model_id, model_version)
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found")

    filename = os.path.basename(file_path)
    return FileResponse(path=file_path, filename=filename, media_type='application/pdf')

@router.get("/debug/gcs")
async def debug_gcs():
    """Debug GCS setup"""
    import os
    from google.cloud import storage
    
    result = {
        "env_vars": {
            "USE_GCS": os.getenv("USE_GCS"),
            "GCS_BUCKET_NAME": os.getenv("GCS_BUCKET_NAME")
        },
        "settings": {
            "USE_GCS": settings.USE_GCS,
            "GCS_BUCKET_NAME": settings.GCS_BUCKET_NAME
        }
    }
    
    try:
        # Test direct GCS access
        client = storage.Client()
        bucket = client.bucket("chatbotstorages")
        test_blob = bucket.blob("debug_test.txt")
        test_blob.upload_from_string("Test upload " + datetime.utcnow().isoformat())
        result["gcs_test"] = "success"
    except Exception as e:
        result["gcs_test"] = str(e)
    
    return result

@router.get("/debug/gcs-upload")
async def debug_gcs_upload():
    """Test direct GCS upload"""
    from google.cloud import storage
    
    try:
        client = storage.Client()
        bucket = client.bucket("chatbotstorages")
        
        # Try both path structures to see which one works
        blob1 = bucket.blob("15/test.txt")
        blob2 = bucket.blob("models/15/test.txt")
        
        blob1.upload_from_string("Test content for path 15/test.txt")
        blob2.upload_from_string("Test content for path models/15/test.txt")
        
        # Check if the blobs exist
        exists1 = blob1.exists()
        exists2 = blob2.exists()
        
        return {
            "path1": "15/test.txt",
            "exists1": exists1,
            "path2": "models/15/test.txt",
            "exists2": exists2
        }
    except Exception as e:
        return {"error": str(e)}