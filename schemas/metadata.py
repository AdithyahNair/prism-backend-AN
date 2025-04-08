from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
from datetime import datetime
import os

class MetricsPaths(BaseModel):
    performance: str
    fairness: str
    drift: str
    explainability: str

    @validator('*')
    def validate_paths(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"Path does not exist: {v}")
        return v

class PlotPaths(BaseModel):
    performance: str
    fairness: str
    drift: str
    explainability: str

    @validator('*')
    def validate_paths(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"Path does not exist: {v}")
        return v

class ReportMetadata(BaseModel):
    timestamp: datetime
    model_name: str = Field(..., min_length=1, max_length=100)
    model_version: str = Field(..., regex=r'^\d+\.\d+\.\d+$')
    model_type: str = Field(..., regex=r'^(classification|regression|clustering|nlp)$')
    model_framework: str = Field(..., regex=r'^(tensorflow|pytorch|sklearn|onnx)$')
    dataset_type: str = Field(..., regex=r'^(tabular|text)$')
    feature_names: List[str]
    metrics_paths: MetricsPaths
    plot_paths: PlotPaths
    report_path: str

    @validator('report_path')
    def validate_report_path(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"Report path does not exist: {v}")
        if not v.endswith('.pdf'):
            raise ValueError("Report path must end with .pdf")
        return v

    @validator('feature_names')
    def validate_feature_names(cls, v):
        if not v:
            raise ValueError("Feature names list cannot be empty")
        if len(v) > 1000:
            raise ValueError("Too many feature names")
        return v

class AuditMetadata(BaseModel):
    timestamp: datetime
    model_type: str = Field(..., regex=r'^(classification|regression|clustering|nlp)$')
    dataset_type: str = Field(..., regex=r'^(tabular|text)$')
    feature_count: int = Field(..., ge=1)
    sample_count: int = Field(..., ge=1)
    metrics: Dict[str, Any]
    plots: Optional[Dict[str, str]] = None

    @validator('metrics')
    def validate_metrics(cls, v):
        if not v:
            raise ValueError("Metrics dictionary cannot be empty")
        return v

class ModelMetadata(BaseModel):
    original_filename: str
    content_type: str
    uploaded_at: datetime
    file_size: int = Field(..., ge=0)
    model_type: str = Field(..., regex=r'^(classification|regression|clustering|nlp)$')
    model_framework: str = Field(..., regex=r'^(tensorflow|pytorch|sklearn|onnx)$')
    version: str = Field(..., regex=r'^\d+\.\d+\.\d+$')

    @validator('file_size')
    def validate_file_size(cls, v):
        if v > 1_000_000_000:  # 1GB
            raise ValueError("File size too large")
        return v

class DatasetMetadata(BaseModel):
    original_filename: str
    content_type: str
    uploaded_at: datetime
    row_count: int = Field(..., ge=1)
    column_count: int = Field(..., ge=1)
    data_types: Dict[str, str]
    missing_values: Dict[str, int] = Field(default_factory=dict)

    @validator('row_count')
    def validate_row_count(cls, v):
        if v > 1_000_000:  # 1M rows
            raise ValueError("Too many rows")
        return v

    @validator('column_count')
    def validate_column_count(cls, v):
        if v > 1000:  # 1000 columns
            raise ValueError("Too many columns")
        return v 