from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class AttackMode(str, Enum):
    MANUAL = "manual"
    TARGETED = "targeted"
    COMPREHENSIVE = "comprehensive"

class RedTeamingCategory(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    HARMUL_CONTENT = "harmful_content"
    BIAS = "bias"
    PRIVACY = "privacy"
    SECURITY = "security"
    PERFORMANCE = "performance"

class LLMConnectorBase(BaseModel):
    name: str
    provider: str  # "OPENAI", "AZURE", "GEMINI"
    api_key: str
    config: Optional[dict] = None
    user_id: int

class LLMConnectorCreate(LLMConnectorBase):
    pass

class LLMConnectorUpdate(BaseModel):
    name: Optional[str] = None
    provider: Optional[str] = None
    api_key: Optional[str] = None
    config: Optional[dict] = None

class LLMConnectorResponse(LLMConnectorBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str
    name: Optional[str] = None

class ChatResponse(BaseModel):
    messages: List[ChatMessage]
    total_tokens: Optional[int] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None

class LLMAuditBase(BaseModel):
    project_id: int
    user_id: int
    connector_id: int
    audit_type: str  # "red_teaming", "benchmark"
    status: str  # "pending", "running", "completed", "failed"
    parameters: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class LLMAuditCreate(LLMAuditBase):
    pass

class LLMAuditResponse(LLMAuditBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# Red Teaming Schemas
class RedTeamingBase(BaseModel):
    project_id: int
    user_id: int
    connector_id: int
    attack_mode: AttackMode
    target_categories: List[RedTeamingCategory]
    parameters: Optional[Dict[str, Any]] = None
    status: str = "pending"  # "pending", "running", "completed", "failed"
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class RedTeamingCreate(RedTeamingBase):
    pass

class RedTeamingResponse(RedTeamingBase):
    id: int
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class RedTeamingProgress(BaseModel):
    audit_id: int
    status: str
    progress: float  # 0-100
    current_step: str
    completed_steps: List[str]
    remaining_steps: List[str]
    estimated_time_remaining: Optional[int] = None  # in seconds
    last_updated: datetime

class RedTeamingVulnerabilities(BaseModel):
    audit_id: int
    vulnerabilities: List[Dict[str, Any]]
    total_count: int
    severity_distribution: Dict[str, int]
    category_distribution: Dict[str, int]
    last_updated: datetime

class RedTeamingMetrics(BaseModel):
    audit_id: int
    total_attempts: int
    successful_attacks: int
    failed_attacks: int
    average_response_time: float
    token_usage: Dict[str, int]
    cost_metrics: Dict[str, float]
    last_updated: datetime

class RedTeamingVisualization(BaseModel):
    audit_id: int
    chart_type: str
    data: Dict[str, Any]
    options: Optional[Dict[str, Any]] = None
    last_updated: datetime

class RedTeamingPerformance(BaseModel):
    audit_id: int
    response_times: List[float]
    token_usage: List[int]
    cost_tracking: List[float]
    success_rates: List[float]
    last_updated: datetime

class RedTeamingCategoryMetrics(BaseModel):
    audit_id: int
    category: str
    attempts: int
    success_rate: float
    average_response_time: float
    token_usage: int
    cost: float
    vulnerabilities: List[Dict[str, Any]]
    last_updated: datetime

class RedTeamingReport(BaseModel):
    id: int
    project_id: int
    model_id: Optional[int] = None
    status: str
    summary: Dict[str, Any]
    categories: Dict[str, Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
