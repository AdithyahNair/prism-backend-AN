from enum import Enum

class ProjectType(str, Enum):
    ML_AUDIT = "ml_audit"
    LLM_AUDIT = "llm_audit"

class ProjectStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    UNDEPLOYING = "undeploying"

class AuditType(str, Enum):
    PERFORMANCE = "performance"
    FAIRNESS = "fairness"
    ROBUSTNESS = "robustness"
    EXPLAINABILITY = "explainability"
    PRIVACY = "privacy"
    SECURITY = "security"
    RED_TEAMING = "red_teaming"
    BENCHMARKING = "benchmarking"

class ModelType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TEXT_GENERATION = "text_generation"
    CUSTOM = "custom"

class DatasetType(str, Enum):
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    CUSTOM = "custom"

class LLMProvider(str, Enum):
    OPENAI = "openai"
    AZURE = "azure"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    DEEPSEEK = "deepseek"
    AMAZON = "amazon"
    TOGETHER = "together"
    GOOGLE = "google"
    CUSTOM = "custom"

class LLMEvaluationType(str, Enum):
    BENCHMARKING = "benchmarking"
    RED_TEAMING = "red_teaming"


class RedTeamingCategory(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    HALLUCINATION = "hallucination"
    BIAS = "bias"
    TOXICITY = "toxicity"
    PRIVACY = "privacy"
    SECURITY = "security"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"

class AttackMode(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    ADVERSARIAL = "adversarial"
    BACKDOOR = "backdoor"
    MEMORIZATION = "memorization"
