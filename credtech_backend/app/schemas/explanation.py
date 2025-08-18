"""
Pydantic schemas for explanation API responses.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class FeatureContribution(BaseModel):
    """Individual feature contribution to the model."""
    feature_name: str = Field(..., description="Name of the feature")
    value: float = Field(..., description="Current value of the feature")
    contribution: float = Field(..., description="SHAP contribution value")
    importance_rank: int = Field(..., description="Rank by absolute contribution")


class StructuredModelExplanation(BaseModel):
    """Explanation of the structured model components."""
    random_forest_score: float = Field(..., description="Raw Random Forest output (0-100)")
    
    # KMV Model details
    kmv_distance_to_default: float = Field(..., description="KMV Distance-to-Default value")
    kmv_inputs: Dict[str, float] = Field(..., description="Inputs used in KMV calculation")
    
    # Z-Score details
    altman_z_score: float = Field(..., description="Altman Z-Score value")
    z_score_inputs: Dict[str, float] = Field(..., description="Financial ratios used in Z-Score")
    z_score_interpretation: str = Field(..., description="Z-Score risk category")
    
    # Feature contributions from SHAP
    top_feature_contributions: List[FeatureContribution] = Field(
        ..., description="Top contributing features from SHAP analysis"
    )


class UnstructuredModelExplanation(BaseModel):
    """Explanation of the unstructured (FinBERT) model."""
    finbert_score: float = Field(..., description="FinBERT processed score (0-100)")
    
    # News analysis
    latest_news_headline: str = Field(..., description="Most recent news headline analyzed")
    sentiment_classification: str = Field(..., description="Positive/Negative/Neutral")
    sentiment_confidence: float = Field(..., description="Model confidence in sentiment")
    raw_sentiment_probabilities: Dict[str, float] = Field(
        ..., description="Raw FinBERT probabilities for each class"
    )
    
    # News impact
    news_articles_analyzed: int = Field(..., description="Number of recent articles analyzed")
    news_date_range: Dict[str, datetime] = Field(..., description="Date range of news data")


class FusionExplanation(BaseModel):
    """Explanation of the dynamic fusion process."""
    final_score: float = Field(..., description="Final fused credit score (0-100)")
    credit_grade: str = Field(..., description="Credit grade (AAA, AA+, etc.)")
    
    # Weighting details
    structured_weight: float = Field(..., description="Weight given to structured model")
    unstructured_weight: float = Field(..., description="Weight given to unstructured model")
    current_vix: float = Field(..., description="Current VIX value used for weighting")
    market_condition: str = Field(..., description="Market condition interpretation")
    
    # Component scores
    structured_component_score: float = Field(..., description="Structured model contribution")
    unstructured_component_score: float = Field(..., description="Unstructured model contribution")


class TrendAnalysis(BaseModel):
    """Trend analysis for short-term and long-term patterns."""
    current_score: float = Field(..., description="Current credit score")
    
    # Short-term trends (7 days)
    score_7d_ago: Optional[float] = Field(None, description="Score 7 days ago")
    change_7d: Optional[float] = Field(None, description="7-day change")
    trend_7d: str = Field(..., description="7-day trend direction")
    
    # Long-term trends (90 days)
    score_90d_ago: Optional[float] = Field(None, description="Score 90 days ago")
    change_90d: Optional[float] = Field(None, description="90-day change")
    trend_90d: str = Field(..., description="90-day trend direction")
    
    # Volatility
    score_volatility: float = Field(..., description="Score volatility over the period")
    stability_assessment: str = Field(..., description="Overall stability assessment")


class PlainLanguageSummary(BaseModel):
    """Plain language explanation of the credit assessment."""
    overall_assessment: str = Field(..., description="High-level credit assessment")
    key_strengths: List[str] = Field(..., description="Main positive factors")
    key_concerns: List[str] = Field(..., description="Main risk factors")
    market_impact: str = Field(..., description="How market conditions affect the score")
    recommendation: str = Field(..., description="Investment/lending recommendation")


class ComprehensiveExplanation(BaseModel):
    """Complete explanation response combining all components."""
    company_symbol: str = Field(..., description="Company stock symbol")
    company_name: str = Field(..., description="Company name")
    calculation_timestamp: datetime = Field(..., description="When this explanation was generated")
    
    # Plain language summary
    plain_language_summary: PlainLanguageSummary = Field(
        ..., description="Human-readable explanation"
    )
    
    # Model explanations
    structured_model: StructuredModelExplanation = Field(
        ..., description="Structured model detailed explanation"
    )
    unstructured_model: UnstructuredModelExplanation = Field(
        ..., description="Unstructured model detailed explanation"
    )
    
    # Fusion and final result
    fusion_process: FusionExplanation = Field(
        ..., description="Dynamic fusion process explanation"
    )
    
    # Trend analysis
    trend_analysis: TrendAnalysis = Field(
        ..., description="Historical trend analysis"
    )
    
    # Raw data for transparency
    raw_data_sources: Dict[str, Any] = Field(
        ..., description="References to underlying data sources"
    )
