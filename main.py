from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.auth import router as auth_router
from routers.project import router as project_router
from routers.ml import router as ml_router
from routers.llm import router as llm_router
from routers.databank import router as databank_router
from middleware.error_handler import ErrorHandlerMiddleware
from middleware.auth_middleware import SupabaseAuthMiddleware
from core.config import settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PRISM Backend",
    description="Backend API for PRISM - AI Model Security Platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add error handler middleware
app.add_middleware(ErrorHandlerMiddleware)

# Add Supabase authentication middleware
app.add_middleware(SupabaseAuthMiddleware)

# Include routers
app.include_router(auth_router)
app.include_router(project_router)
app.include_router(ml_router)
app.include_router(llm_router)
app.include_router(databank_router)

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    try:
        # Log application startup
        logger.info(f"Starting {settings.PROJECT_NAME} API")
        
        # Check Supabase connection
        from services.database import get_supabase_client
        supabase = get_supabase_client()
        logger.info("Successfully connected to Supabase")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} Backend API",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    ) 