from supabase import create_client, Client
from core.config import settings
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

def get_supabase_client() -> Client:
    """
    Get the Supabase client instance.
    This can be used as a dependency in FastAPI routes.
    """
    return supabase

# User operations
async def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    try:
        response = supabase.table('users').select("*").eq('id', user_id).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error(f"Error fetching user: {str(e)}")
        raise

async def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    try:
        response = supabase.table('users').select("*").eq('email', email).execute()
        logger.info(f"User lookup by email: {email}, found: {bool(response.data)}")
        if response.data:
            logger.info(f"User data retrieved, columns: {list(response.data[0].keys())}")
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error(f"Error fetching user by email: {str(e)}")
        raise

# Project operations
async def get_projects_by_user_id(user_id: int) -> List[Dict[str, Any]]:
    try:
        response = supabase.table('projects').select("*").eq('user_id', user_id).execute()
        return response.data
    except Exception as e:
        logger.error(f"Error fetching projects: {str(e)}")
        raise

async def get_project_by_id(project_id: int) -> Optional[Dict[str, Any]]:
    try:
        response = supabase.table('projects').select("*").eq('id', project_id).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error(f"Error fetching project: {str(e)}")
        raise

async def create_project(project_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        logger.info(f"Creating project with data: {project_data}")
        response = supabase.table('projects').insert(project_data).execute()
        logger.info(f"Project created, response: {response}")
        if not response.data:
            raise Exception("No data returned from Supabase after project creation")
        return response.data[0]
    except Exception as e:
        logger.error(f"Error creating project: {str(e)}")
        logger.exception("Detailed error information:")
        # Don't try to use rollback since Supabase doesn't support it
        raise Exception(f"Failed to create project: {str(e)}")

async def update_project(project_id: int, project_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = supabase.table('projects').update(project_data).eq('id', project_id).execute()
        return response.data[0]
    except Exception as e:
        logger.error(f"Error updating project: {str(e)}")
        raise

async def delete_project(project_id: int) -> Dict[str, Any]:
    try:
        response = supabase.table('projects').delete().eq('id', project_id).execute()
        return response.data[0]
    except Exception as e:
        logger.error(f"Error deleting project: {str(e)}")
        raise

# Model operations
async def get_models_by_project_id(project_id: int) -> List[Dict[str, Any]]:
    try:
        response = supabase.table('models').select("*").eq('project_id', project_id).execute()
        return response.data
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        raise

async def get_model_by_id(model_id: int) -> Optional[Dict[str, Any]]:
    try:
        response = supabase.table('models').select("*").eq('id', model_id).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error(f"Error fetching model: {str(e)}")
        raise

async def create_model(model_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = supabase.table('models').insert(model_data).execute()
        return response.data[0]
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise

# Dataset operations
async def get_datasets_by_project_id(project_id: int) -> List[Dict[str, Any]]:
    try:
        response = supabase.table('datasets').select("*").eq('project_id', project_id).execute()
        return response.data
    except Exception as e:
        logger.error(f"Error fetching datasets: {str(e)}")
        raise

async def get_dataset_by_id(dataset_id: int) -> Optional[Dict[str, Any]]:
    try:
        response = supabase.table('datasets').select("*").eq('id', dataset_id).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error(f"Error fetching dataset: {str(e)}")
        raise

async def create_dataset(dataset_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = supabase.table('datasets').insert(dataset_data).execute()
        return response.data[0]
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        raise

# Audit operations
async def get_audits_by_project_id(project_id: int) -> List[Dict[str, Any]]:
    try:
        response = supabase.table('audits').select("*").eq('project_id', project_id).execute()
        return response.data
    except Exception as e:
        logger.error(f"Error fetching audits: {str(e)}")
        raise

async def get_audit_by_id(audit_id: int) -> Optional[Dict[str, Any]]:
    try:
        response = supabase.table('audits').select("*").eq('id', audit_id).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error(f"Error fetching audit: {str(e)}")
        raise

async def create_audit(audit_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = supabase.table('audits').insert(audit_data).execute()
        return response.data[0]
    except Exception as e:
        logger.error(f"Error creating audit: {str(e)}")
        raise

async def update_audit(audit_id: int, audit_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = supabase.table('audits').update(audit_data).eq('id', audit_id).execute()
        return response.data[0]
    except Exception as e:
        logger.error(f"Error updating audit: {str(e)}")
        raise

# LLM Connector operations
async def get_llm_connectors_by_project_id(project_id: int) -> List[Dict[str, Any]]:
    try:
        response = supabase.table('llm_connectors').select("*").eq('project_id', project_id).execute()
        return response.data
    except Exception as e:
        logger.error(f"Error fetching LLM connectors: {str(e)}")
        raise

async def create_llm_connector(connector_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = supabase.table('llm_connectors').insert(connector_data).execute()
        return response.data[0]
    except Exception as e:
        logger.error(f"Error creating LLM connector: {str(e)}")
        raise

# Report operations
async def get_reports_by_project_id(project_id: int) -> List[Dict[str, Any]]:
    try:
        response = supabase.table('reports').select("*").eq('project_id', project_id).execute()
        return response.data
    except Exception as e:
        logger.error(f"Error fetching reports: {str(e)}")
        raise

async def create_report(report_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = supabase.table('reports').insert(report_data).execute()
        return response.data[0]
    except Exception as e:
        logger.error(f"Error creating report: {str(e)}")
        raise 