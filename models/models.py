from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Boolean, Text, Float
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
from typing import Any

Base = declarative_base()

# User Management
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, index=True)
    email = Column(String(255), unique=True, index=True)
    hashed_password = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    projects = relationship("Project", back_populates="user")
    models = relationship("Model", back_populates="user")
    datasets = relationship("Dataset", back_populates="user")
    audits = relationship("Audit", back_populates="user")
    reports = relationship("Report", back_populates="user")
    llm_connectors = relationship("LLMConnector", back_populates="user")


# Project
class Project(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255))
    description = Column(Text, nullable=True)
    project_type = Column(String(50))
    status = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    user = relationship("User", back_populates="projects")
    models = relationship("Model", back_populates="project")
    datasets = relationship("Dataset", back_populates="project")
    audits = relationship("Audit", back_populates="project")
    reports = relationship("Report", back_populates="project")
    llm_connectors = relationship("LLMConnector", back_populates="project")


# Model
class Model(Base):
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(String(500))
    model_type = Column(String(50))  # classification, regression, clustering, nlp
    model_framework = Column(String(50))  # tensorflow, pytorch, sklearn, onnx
    version = Column(String(20))
    file_path = Column(String(500))
    project_id = Column(Integer, ForeignKey("projects.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    model_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    project = relationship("Project", back_populates="models")
    user = relationship("User", back_populates="models")
    reports = relationship("Report", back_populates="model")


# Dataset
class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255))
    description = Column(Text, nullable=True)
    dataset_type = Column(String(50))
    file_path = Column(String(255))
    dataset_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    project_id = Column(Integer, ForeignKey("projects.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    
    project = relationship("Project", back_populates="datasets")
    user = relationship("User", back_populates="datasets")
    reports = relationship("Report", back_populates="dataset")


# Audit
class Audit(Base):
    __tablename__ = "audits"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    model_id = Column(Integer, ForeignKey("models.id"), nullable=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Audit Configuration
    audit_type = Column(String(50))  # "red_teaming", "benchmark", "performance", etc.
    status = Column(String(50))  # "pending", "running", "completed", "failed"

    # Progress Tracking
    progress = Column(Float, default=0.0)  # 0-100

    # Results
    results = Column(JSON, nullable=True)  # Main audit results
    metrics = Column(JSON, nullable=True)  # Performance metrics
    vulnerabilities = Column(JSON, nullable=True)  # Found vulnerabilities
    recommendations = Column(JSON, nullable=True)  # Generated recommendations
    
    # Visualization Data
    visualization_data = Column(JSON, nullable=True)  # Data for charts/graphs

    error_details = Column(JSON, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="audits")
    user = relationship("User", back_populates="audits")


# TestDataset
class TestDataset(Base):
    """Model for benchmark test datasets"""
    __tablename__ = "test_datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String, nullable=True)
    file_path = Column(String)
    type = Column(String)  # text, image, tabular, etc.
    format = Column(String)  # json, csv, etc.
    size = Column(Integer)
    tags = Column(JSON, default=list)
    categories = Column(JSON, default=list)
    schema = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Cookbook
class Cookbook(Base):
    """Model for benchmark cookbooks"""
    __tablename__ = "cookbooks"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String, nullable=True)
    file_path = Column(String)
    version = Column(String)
    status = Column(String)  # active, deprecated, etc.
    tags = Column(JSON, default=list)
    categories = Column(JSON, default=list)
    datasets = Column(JSON, default=list)  # List of dataset names used
    metrics = Column(JSON, default=list)  # List of metric names used
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    

# LLMConnector
class LLMConnector(Base):
    """Model for LLM connectors"""
    __tablename__ = "llm_connectors"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255))
    provider = Column(String(50))  # "OPENAI", "AZURE", "GEMINI"
    api_key = Column(String(255))  # Added to match schema
    config = Column(JSON, nullable=True)  # Made nullable to match schema
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    project_id = Column(Integer, ForeignKey("projects.id"))
    user_id = Column(Integer, ForeignKey("users.id"))

    project = relationship("Project", back_populates="llm_connectors")
    user = relationship("User", back_populates="llm_connectors")


# Report
class Report(Base):
    """Model for storing audit reports with blockchain verification"""
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    model_id = Column(Integer, ForeignKey("models.id"))
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    report_type = Column(String)
    blockchain_hash = Column(String)
    file_path = Column(String)
    report_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="reports")
    model = relationship("Model", back_populates="reports")
    dataset = relationship("Dataset", back_populates="reports")
    user = relationship("User", back_populates="reports")


