"""
Routers package for the Prism backend application.
"""

from fastapi import APIRouter

from .project import router as project_router
from .ml import router as ml_router
from .llm import router as llm_router
from .databank import router as databank_router
from .auth import router as auth_router

api_router = APIRouter()

api_router.include_router(auth_router)
api_router.include_router(project_router)
api_router.include_router(ml_router)
api_router.include_router(llm_router)
api_router.include_router(databank_router)

__all__ = [
    "auth_router",
    "project_router",
    "ml_router",
    "llm_router",
    "databank_router"
] 