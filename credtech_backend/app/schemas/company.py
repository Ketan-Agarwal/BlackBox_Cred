"""
Pydantic schemas for company-related API responses.
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class CompanyBase(BaseModel):
    """Base company schema."""
    symbol: str = Field(..., description="Stock symbol")
    name: str = Field(..., description="Company name")
    sector: Optional[str] = Field(None, description="Business sector")
    industry: Optional[str] = Field(None, description="Industry classification")


class CompanyCreate(CompanyBase):
    """Schema for creating a new company."""
    pass


class Company(CompanyBase):
    """Company schema with database fields."""
    id: int
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class CompanyWithScore(Company):
    """Company schema with latest credit score information."""
    latest_score: Optional[float] = Field(None, description="Latest credit score (0-100)")
    latest_grade: Optional[str] = Field(None, description="Latest credit grade")
    score_trend_7d: Optional[float] = Field(None, description="7-day score change")
    score_trend_90d: Optional[float] = Field(None, description="90-day score change")
    last_updated: Optional[datetime] = Field(None, description="Last score calculation time")


class StructuredDataResponse(BaseModel):
    """Schema for structured financial data."""
    current_ratio: Optional[float]
    quick_ratio: Optional[float]
    debt_to_equity: Optional[float]
    return_on_equity: Optional[float]
    return_on_assets: Optional[float]
    operating_margin: Optional[float]
    net_margin: Optional[float]
    total_assets: Optional[float]
    total_liabilities: Optional[float]
    market_cap: Optional[float]
    kmv_distance_to_default: Optional[float]
    altman_z_score: Optional[float]
    data_date: datetime
    
    class Config:
        from_attributes = True


class UnstructuredDataResponse(BaseModel):
    """Schema for unstructured news data."""
    headline: str
    source: Optional[str]
    published_at: Optional[datetime]
    sentiment_label: Optional[str]
    sentiment_score: Optional[float]
    processed_score: Optional[float]
    
    class Config:
        from_attributes = True
