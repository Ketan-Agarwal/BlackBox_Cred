"""
SQLAlchemy ORM models for the CredTech backend system.
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Company(Base):
    """Company entity with basic information."""
    __tablename__ = "companies"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    structured_data = relationship("StructuredData", back_populates="company")
    unstructured_data = relationship("UnstructuredData", back_populates="company")
    credit_score_history = relationship("CreditScoreHistory", back_populates="company")


class StructuredData(Base):
    """Structured financial data for companies."""
    __tablename__ = "structured_data"
    
    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    
    # Financial ratios
    current_ratio = Column(Float)
    quick_ratio = Column(Float)
    debt_to_equity = Column(Float)
    return_on_equity = Column(Float)
    return_on_assets = Column(Float)
    operating_margin = Column(Float)
    net_margin = Column(Float)
    asset_turnover = Column(Float)
    inventory_turnover = Column(Float)
    
    # Balance sheet items (for KMV model)
    total_assets = Column(Float)
    total_liabilities = Column(Float)
    total_equity = Column(Float)
    short_term_debt = Column(Float)
    long_term_debt = Column(Float)
    
    # Income statement items
    revenue = Column(Float)
    operating_income = Column(Float)
    net_income = Column(Float)
    ebitda = Column(Float)
    
    # Market data
    stock_price = Column(Float)
    market_cap = Column(Float)
    shares_outstanding = Column(Float)
    volatility = Column(Float)  # Stock price volatility for KMV
    
    # Calculated scores
    kmv_distance_to_default = Column(Float)
    altman_z_score = Column(Float)
    
    data_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    company = relationship("Company", back_populates="structured_data")


class UnstructuredData(Base):
    """Unstructured news and text data for companies."""
    __tablename__ = "unstructured_data"
    
    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    
    # News article information
    headline = Column(Text, nullable=False)
    content = Column(Text)
    source = Column(String(255))
    url = Column(Text)
    published_at = Column(DateTime)
    
    # Elasticsearch document ID for full text storage
    elasticsearch_doc_id = Column(String(255))
    
    # FinBERT analysis results
    sentiment_score = Column(Float)  # -1 to 1 scale
    sentiment_label = Column(String(50))  # Positive, Negative, Neutral
    finbert_confidence = Column(Float)  # Confidence score
    processed_score = Column(Float)  # 0-100 scale
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    company = relationship("Company", back_populates="unstructured_data")


class CreditScoreHistory(Base):
    """Historical credit scores and grades for trend analysis."""
    __tablename__ = "credit_score_history"
    
    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    
    # Scores
    structured_score = Column(Float, nullable=False)  # 0-100
    unstructured_score = Column(Float, nullable=False)  # 0-100
    final_score = Column(Float, nullable=False)  # 0-100
    credit_grade = Column(String(10), nullable=False)  # AAA, AA+, etc.
    
    # Weighting used in calculation
    news_weight = Column(Float, nullable=False)  # Weight given to unstructured score
    market_volatility_vix = Column(Float)  # VIX value at time of calculation
    
    # Metadata
    calculation_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    company = relationship("Company", back_populates="credit_score_history")


class ModelMetadata(Base):
    """Metadata for tracking model versions and performance."""
    __tablename__ = "model_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String(50), nullable=False)  # 'random_forest', 'finbert', etc.
    model_version = Column(String(50), nullable=False)
    model_path = Column(String(500))
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    # Training metadata
    training_data_size = Column(Integer)
    training_date = Column(DateTime)
    features_used = Column(JSON)  # List of features used
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
