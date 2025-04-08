from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from services.database import get_supabase_client, get_user_by_id
import logging
from typing import Dict, Any, Callable
import jwt
from core.config import settings
from starlette.types import ASGIApp, Scope, Receive, Send
from starlette.middleware.base import BaseHTTPMiddleware
from core.deps import UserWrapper

logger = logging.getLogger(__name__)
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
    """
    Verify the JWT token from the Authorization header.
    This can be used as a dependency in FastAPI routes.
    """
    try:
        token = credentials.credentials
        
        # Verify the JWT token
        try:
            # First try to decode the token directly
            payload = jwt.decode(
                token, 
                settings.JWT_SECRET, 
                algorithms=[settings.JWT_ALGORITHM]
            )
            user_id = int(payload.get("sub"))
        except jwt.PyJWTError as e:
            logger.error(f"JWT decode error: {str(e)}")
            raise HTTPException(
                status_code=401,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get the user from our users table
        user = await get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(
                status_code=401,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return UserWrapper(user)
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

class SupabaseAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        """Process the request and add authentication."""
        # Skip auth for public endpoints
        public_paths = [
            "/docs", 
            "/redoc", 
            "/openapi.json", 
            "/", 
            "/auth/signup", 
            "/auth/signin",
            "/auth/token",
            "/static"
        ]
        
        # Skip auth for tokens endpoint
        if any(request.url.path.startswith(path) for path in public_paths):
            return await call_next(request)
            
        try:
            # Get the authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                raise HTTPException(
                    status_code=401,
                    detail="Missing authorization header",
                    headers={"WWW-Authenticate": "Bearer"},
                )
                
            # Verify the token
            token = auth_header.split(" ")[1]
            
            try:
                # Decode the JWT token
                payload = jwt.decode(
                    token,
                    settings.JWT_SECRET,
                    algorithms=[settings.JWT_ALGORITHM]
                )
                user_id = int(payload.get("sub"))
            except jwt.PyJWTError as e:
                logger.error(f"JWT decode error: {str(e)}")
                raise HTTPException(
                    status_code=401,
                    detail="Invalid token",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Get the user from our users table
            user = await get_user_by_id(user_id)
            
            if not user:
                raise HTTPException(
                    status_code=401,
                    detail="User not found",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Add the user to the request state
            request.state.user = user
            
            return await call_next(request)
            
        except HTTPException as e:
            from starlette.responses import JSONResponse
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail},
                headers=e.headers or {}
            )
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            from starlette.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid authentication credentials"},
                headers={"WWW-Authenticate": "Bearer"}
            ) 