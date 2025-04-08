from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ErrorResponse:
    def __init__(
        self,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        path: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.path = path
        self.timestamp = timestamp or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "path": self.path,
            "timestamp": self.timestamp.isoformat()
        }

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            # Log the error
            logger.error(f"Error processing request: {str(exc)}", exc_info=True)
            
            # Determine error code based on exception type
            error_code = "INTERNAL_SERVER_ERROR"
            status_code = 500
            
            if hasattr(exc, 'status_code'):
                status_code = exc.status_code
                
            if "Not found" in str(exc):
                error_code = "NOT_FOUND"
                status_code = 404
            elif "Unauthorized" in str(exc) or "authentication" in str(exc).lower():
                error_code = "UNAUTHORIZED"
                status_code = 401
            elif "Permission" in str(exc) or "Forbidden" in str(exc):
                error_code = "FORBIDDEN"
                status_code = 403
            elif "already exists" in str(exc).lower() or "Duplicate" in str(exc):
                error_code = "DUPLICATE_RESOURCE"
                status_code = 409
            elif "Invalid" in str(exc) or "Validation" in str(exc):
                error_code = "VALIDATION_ERROR"
                status_code = 400
                
            # Create error response
            error_response = ErrorResponse(
                error_code=error_code,
                message=str(exc),
                details={"error_type": type(exc).__name__},
                path=request.url.path,
                timestamp=datetime.utcnow()
            )
            
            return JSONResponse(
                status_code=status_code,
                content=error_response.to_dict()
            ) 