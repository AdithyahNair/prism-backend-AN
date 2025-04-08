from fastapi import APIRouter, Depends, HTTPException, status, Request, Form
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from datetime import timedelta, datetime
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any
import logging
import bcrypt

from core.auth import (
    verify_password,
    create_access_token,
    get_password_hash
)
from core.deps import get_current_user
from core.config import settings
from services.database import get_supabase_client, get_user_by_email

router = APIRouter(
    prefix="/auth",
    tags=["authentication"]
)

logger = logging.getLogger(__name__)

# Pydantic models for request/response
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: str

class Token(BaseModel):
    access_token: str
    token_type: str

@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user: UserCreate):
    """Register a new user"""
    try:
        # Check if email already exists
        logger.info(f"Checking if email {user.email} already exists")
        existing_user = await get_user_by_email(user.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        logger.info(f"Creating new user with email {user.email}")
        hashed_password = get_password_hash(user.password)
        
        # Insert user into Supabase
        supabase = get_supabase_client()
        user_data = {
            "username": user.username,
            "email": user.email,
            "hashed_password": hashed_password,
            "is_active": True,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Log the user data structure (without the password)
        log_data = user_data.copy()
        log_data.pop("hashed_password", None)
        logger.info(f"Inserting user data: {log_data}")
        
        response = supabase.table('users').insert(user_data).execute()
        if not response.data:
            logger.error("Failed to create user in Supabase")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error creating user"
            )
            
        new_user = response.data[0]
        logger.info(f"User {user.email} created successfully")
        return new_user
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {str(e)}"
        )

@router.post("/signin", response_model=Token)
async def login_for_access_token(user_data: UserLogin):
    """Login and get access token"""
    logger.info(f"Logging in for access token with email: {user_data.email}")
    
    try:
        # Verify user credentials
        user = await get_user_by_email(user_data.email)
        if not user or not verify_password(user_data.password, user['hashed_password']):
            logger.error(f"Invalid credentials for user: {user_data.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(user["id"])}, 
            expires_delta=access_token_expires
        )
        
        logger.info(f"Access token created for user: {user_data.email}")
        return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during login: {str(e)}"
        )

# This endpoint is used by Swagger UI for OAuth2 password flow
@router.post("/token", response_model=Token)
async def login_oauth2(form_data: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 compatible token login, get an access token for future requests"""
    logger.info(f"OAuth2 login for: {form_data.username}")
    
    try:
        # Form data uses username field, but we store as email
        user = await get_user_by_email(form_data.username)
        if not user or not verify_password(form_data.password, user['hashed_password']):
            logger.error(f"Invalid credentials for user: {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(user["id"])}, 
            expires_delta=access_token_expires
        )
        
        logger.info(f"OAuth2 token created for user: {form_data.username}")
        return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error during OAuth2 login: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during login: {str(e)}"
        )

@router.post("/signout")
async def logout(current_user: Dict = Depends(get_current_user)):
    """Logout current user"""
    return {"message": "Successfully signed out"}

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: Dict = Depends(get_current_user)):
    """Get current user information"""
    return current_user 