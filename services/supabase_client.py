from supabase import create_client, Client
from core.config import settings

async def get_supabase_client() -> Client:
    """Get a Supabase client instance."""
    if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
        raise ValueError("Supabase URL and Key must be set in environment variables")
    
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

async def get_supabase_storage() -> Client:
    """Get a Supabase storage client instance."""
    client = await get_supabase_client()
    return client.storage

async def get_supabase_auth() -> Client:
    """Get a Supabase auth client instance."""
    client = await get_supabase_client()
    return client.auth 