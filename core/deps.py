from typing import Callable, Dict, Any, Optional
from supabase import Client
from services.database import get_supabase_client, get_user_by_id
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
import jwt
from core.config import settings
import logging

logger = logging.getLogger(__name__)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)

def get_db() -> Client:
    """
    Get Supabase client with added compatibility methods for SQLAlchemy transition.
    This allows existing code that expects SQLAlchemy session to work with Supabase.
    """
    client = get_supabase_client()
    
    # Add compatibility methods for SQLAlchemy transition
    # This lets old code that calls SQLAlchemy session methods continue to work
    def noop(*args, **kwargs):
        """Does nothing, used for compatibility with SQLAlchemy methods"""
        logger.info("Supabase client: rollback called (no-op during transition)")
        return None
        
    # Add SQLAlchemy session compatibility methods
    client.rollback = noop
    client.commit = noop
    client.close = noop
    client.flush = noop
    client.refresh = noop
    
    return client

class UserWrapper:
    """
    Wrapper for user data that allows both dictionary-style and attribute access.
    This helps with compatibility during migration from SQLAlchemy models to Supabase.
    """
    def __init__(self, user_data: Dict[str, Any]):
        self._data = user_data
        
    def __getattr__(self, name):
        """Allow attribute-style access: user.id"""
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'User' object has no attribute '{name}'")
        
    def __getitem__(self, key):
        """Allow dict-style access: user['id']"""
        return self._data[key]
        
    def __contains__(self, key):
        """Allow 'in' operator: 'id' in user"""
        return key in self._data
        
    def get(self, key, default=None):
        """Dict-like get method"""
        return self._data.get(key, default)

async def get_current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme)
) -> UserWrapper:
    """
    Get current user from either:
    1. Request state (set by middleware)
    2. JWT token (for API endpoints)
    """
    # First try to get the user from request state (middleware-set)
    if hasattr(request.state, "user"):
        return UserWrapper(request.state.user)
        
    # If not in state, try from token
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    try:
        # Decode the JWT token
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Convert user_id to int
        try:
            user_id = int(user_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user ID in token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get the user from our users table
        user = await get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        # Wrap the user dict for compatibility
        return UserWrapper(user)
    except jwt.PyJWTError as e:
        logger.error(f"JWT error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication error",
            headers={"WWW-Authenticate": "Bearer"},
        ) 