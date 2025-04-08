from supabase import create_client, Client
from core.config import settings
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Initialize Supabase client
def init_supabase_client() -> Client:
    """Initialize and return a Supabase client."""
    try:
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            raise ValueError("Supabase URL and API Key must be set in environment variables")
        
        client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        logger.info("Supabase client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {str(e)}")
        raise

# Create a global client instance
supabase_client: Client = init_supabase_client()

def get_db_client() -> Client:
    """
    Get the Supabase client instance.
    This can be used as a dependency in FastAPI routes.
    """
    return supabase_client 