from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "PRISM"
    
    # Supabase Settings
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    SUPABASE_JWT_SECRET: str = os.getenv("SUPABASE_JWT_SECRET", "")
    
    # Development Mode
    DEV_MODE: bool = True
    
    # CORS Settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",           # React frontend
        "http://localhost:5173",           # Vite frontend 
        "http://localhost:8000",           # FastAPI backend
        "https://ai-gov-dashboard.vercel.app",
        "https://api-service-blockconvey-aigovernance-dot-block-convey-p1.uc.r.appspot.com",
        "https://block-convey-p1.uc.r.appspot.com"
    ]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Optional[str] = None
    
    # File Storage
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # GCS Settings
    GCS_BUCKET_NAME: str = os.getenv("GCS_BUCKET_NAME", "")
    USE_GCS: bool = os.getenv("USE_GCS", "false").lower() == "true"
    
    # Model Settings
    
    SUPPORTED_MODEL_TYPES: List[str] = ["h5", "pkl", "pt", "pth", "onnx", "pb"]
    
    # Dataset Settings
    SUPPORTED_DATASET_TYPES: List[str] = ["csv", "parquet", "json", "jsonl"]
    
    # Output Directory
    OUTPUT_DIR: str = "outputs"
    
    # JWT Settings
    JWT_SECRET: str = os.getenv("JWT_SECRET", "supersecret")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # LLM Configuration - Optional as we may not need these for Supabase migration
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    
    # Feature Flags
    ENABLE_RED_TEAMING: bool = False
    ENABLE_BENCHMARKING: bool = False
    ENABLE_AUDIT: bool = False
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    RATE_LIMIT_PER_MINUTE_ANONYMOUS: int = 10
    
    # Monitoring
    ENABLE_METRICS: bool = False
    METRICS_PORT: int = 9090
    
    # For backward compatibility - will be removed later
    DATABASE_URL: Optional[str] = None  
    SECRET_KEY: Optional[str] = None
    ALGORITHM: Optional[str] = None
    
    # TensorFlow configuration
    TF_ENABLE_ONEDNN_OPTS: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields in the environment file

settings = Settings()

# Create necessary directories only if not using GCS
if not settings.USE_GCS:
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)