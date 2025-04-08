from fastapi import HTTPException, status
from typing import Any, Dict, Optional
from pydantic import BaseModel

class ErrorResponse(BaseModel):
    """Standard error response model"""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str
    path: str

class BaseAPIError(HTTPException):
    """Base class for all API errors"""
    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        super().__init__(status_code=status_code, detail=message, headers=headers)
        self.error_code = error_code
        self.details = details or {}

class ProjectError(BaseAPIError):
    """Project-related errors"""
    def __init__(
        self,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code=error_code,
            message=message,
            details=details
        )

class ProjectNotFoundError(ProjectError):
    """Project not found error"""
    def __init__(self, project_id: str):
        super().__init__(
            error_code="PROJECT_NOT_FOUND",
            message=f"Project with ID {project_id} not found",
            details={"project_id": project_id}
        )

class ProjectTypeMismatchError(ProjectError):
    """Project type mismatch error"""
    def __init__(self, project_id: str, expected_type: str, actual_type: str):
        super().__init__(
            error_code="PROJECT_TYPE_MISMATCH",
            message=f"Project {project_id} is of type {actual_type}, but {expected_type} was expected",
            details={
                "project_id": project_id,
                "expected_type": expected_type,
                "actual_type": actual_type
            }
        )

class FileUploadError(BaseAPIError):
    """File upload related errors"""
    def __init__(
        self,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code=error_code,
            message=message,
            details=details
        )

class InvalidFileTypeError(FileUploadError):
    """Invalid file type error"""
    def __init__(self, file_name: str, allowed_types: list):
        super().__init__(
            error_code="INVALID_FILE_TYPE",
            message=f"File {file_name} has an invalid type. Allowed types: {', '.join(allowed_types)}",
            details={
                "file_name": file_name,
                "allowed_types": allowed_types
            }
        )

class FileSizeLimitError(FileUploadError):
    """File size limit exceeded error"""
    def __init__(self, file_name: str, max_size: int):
        super().__init__(
            error_code="FILE_SIZE_LIMIT",
            message=f"File {file_name} exceeds the maximum size limit of {max_size} bytes",
            details={
                "file_name": file_name,
                "max_size": max_size
            }
        )

class AuditError(BaseAPIError):
    """Audit-related errors"""
    def __init__(
        self,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code=error_code,
            message=message,
            details=details
        )

class AuditConfigurationError(AuditError):
    """Invalid audit configuration error"""
    def __init__(self, error_code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=error_code,
            message=message,
            details=details
        )

class AuditExecutionError(AuditError):
    """Error during audit execution"""
    def __init__(
        self,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=error_code,
            message=message,
            details=details
        )

class ResourceNotFoundError(BaseAPIError):
    """Resource not found error"""
    def __init__(
        self,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=error_code,
            message=message,
            details=details
        )

class ValidationError(BaseAPIError):
    """Validation error"""
    def __init__(
        self,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code=error_code,
            message=message,
            details=details
        )

class DatabaseError(BaseAPIError):
    """Database-related errors"""
    def __init__(
        self,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=error_code,
            message=message,
            details=details
        )

class AuthenticationError(BaseAPIError):
    """Authentication-related errors"""
    def __init__(
        self,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code=error_code,
            message=message,
            details=details
        )

class AuthorizationError(BaseAPIError):
    """Authorization-related errors"""
    def __init__(
        self,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            error_code=error_code,
            message=message,
            details=details
        )

class RateLimitError(BaseAPIError):
    """Rate limit exceeded error"""
    def __init__(
        self,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code=error_code,
            message=message,
            details=details
        )

class ServiceUnavailableError(BaseAPIError):
    """Service unavailable error"""
    def __init__(
        self,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code=error_code,
            message=message,
            details=details
        ) 