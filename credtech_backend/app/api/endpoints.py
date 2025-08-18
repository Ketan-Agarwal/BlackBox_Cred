"""
FastAPI endpoints for the CredTech backend system.
"""
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.db.models import Company, CreditScoreHistory
from app.schemas.company import Company as CompanySchema, CompanyWithScore
from app.schemas.explanation import ComprehensiveExplanation
from app.services.explainability_service import ExplainabilityService
from app.services.structured_service import StructuredModelService
from app.services.unstructured_service import UnstructuredModelService
from app.services.fusion_service import FusionService
from app.jobs.scheduler import DataIngestionScheduler

# Create router
router = APIRouter()

# Initialize services
explainability_service = ExplainabilityService()
structured_service = StructuredModelService()
unstructured_service = UnstructuredModelService()
fusion_service = FusionService()
scheduler = DataIngestionScheduler()


@router.get("/companies", response_model=List[CompanyWithScore])
async def get_companies(db: Session = Depends(get_db)):
    """
    Get list of all companies with their latest credit scores and trends.
    """
    try:
        companies = db.query(Company).all()
        result = []
        
        for company in companies:
            # Get latest score
            latest_score = (
                db.query(CreditScoreHistory)
                .filter(CreditScoreHistory.company_id == company.id)
                .order_by(CreditScoreHistory.calculation_timestamp.desc())
                .first()
            )
            
            # Get trend data
            trend_data = fusion_service.get_score_trends(company.id)
            
            company_with_score = CompanyWithScore(
                id=company.id,
                symbol=company.symbol,
                name=company.name,
                sector=company.sector,
                industry=company.industry,
                market_cap=company.market_cap,
                created_at=company.created_at,
                updated_at=company.updated_at,
                latest_score=latest_score.final_score if latest_score else None,
                latest_grade=latest_score.credit_grade if latest_score else None,
                score_trend_7d=trend_data['trends'].get('7d', {}).get('change'),
                score_trend_90d=trend_data['trends'].get('90d', {}).get('change'),
                last_updated=latest_score.calculation_timestamp if latest_score else None
            )
            
            result.append(company_with_score)
        
        # Sort by latest score (descending)
        result.sort(key=lambda x: x.latest_score or 0, reverse=True)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching companies: {str(e)}")


@router.get("/companies/{company_id}/explanation", response_model=ComprehensiveExplanation)
async def get_company_explanation(company_id: int, db: Session = Depends(get_db)):
    """
    Get comprehensive credit score explanation for a specific company.
    This is the main endpoint that combines all model outputs and explanations.
    """
    try:
        # Verify company exists
        company = db.query(Company).filter(Company.id == company_id).first()
        if not company:
            raise HTTPException(status_code=404, detail=f"Company with ID {company_id} not found")
        
        # Generate comprehensive explanation
        explanation = explainability_service.generate_explanation(company_id)
        
        return explanation
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating explanation: {str(e)}")


@router.get("/companies/{company_id}/scores/structured")
async def get_structured_score(company_id: int, db: Session = Depends(get_db)):
    """
    Get detailed structured model score and breakdown for a company.
    """
    try:
        company = db.query(Company).filter(Company.id == company_id).first()
        if not company:
            raise HTTPException(status_code=404, detail=f"Company with ID {company_id} not found")
        
        result = structured_service.get_structured_score(company_id)
        
        return {
            "company_id": company_id,
            "company_symbol": company.symbol,
            "structured_score": result['structured_score'],
            "kmv_distance_to_default": result['kmv_distance_to_default'],
            "altman_z_score": result['altman_z_score'],
            "feature_contributions": {
                "feature_names": result['feature_names'],
                "feature_values": result['feature_values'],
                "shap_values": result['shap_values']
            },
            "calculation_timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating structured score: {str(e)}")


@router.get("/companies/{company_id}/scores/unstructured")
async def get_unstructured_score(company_id: int, days_back: int = 7, db: Session = Depends(get_db)):
    """
    Get detailed unstructured model score and sentiment analysis for a company.
    """
    try:
        company = db.query(Company).filter(Company.id == company_id).first()
        if not company:
            raise HTTPException(status_code=404, detail=f"Company with ID {company_id} not found")
        
        result = unstructured_service.get_unstructured_score(company_id, days_back)
        
        return {
            "company_id": company_id,
            "company_symbol": company.symbol,
            "unstructured_score": result['unstructured_score'],
            "analysis_period_days": days_back,
            "articles_analyzed": result['article_count'],
            "latest_headline": result['latest_headline'],
            "sentiment_analysis": result['sentiment_analysis'],
            "date_range": result['date_range'],
            "sample_articles": result.get('raw_articles', []),
            "calculation_timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating unstructured score: {str(e)}")


@router.get("/companies/{company_id}/scores/final")
async def get_final_score(company_id: int, db: Session = Depends(get_db)):
    """
    Get final fused credit score with detailed breakdown.
    """
    try:
        company = db.query(Company).filter(Company.id == company_id).first()
        if not company:
            raise HTTPException(status_code=404, detail=f"Company with ID {company_id} not found")
        
        # Get component scores
        structured_result = structured_service.get_structured_score(company_id)
        unstructured_result = unstructured_service.get_unstructured_score(company_id)
        
        # Calculate final score
        fusion_result = fusion_service.calculate_final_score(
            structured_score=structured_result['structured_score'],
            unstructured_score=unstructured_result['unstructured_score'],
            company_id=company_id
        )
        
        return {
            "company_id": company_id,
            "company_symbol": company.symbol,
            "final_score": fusion_result['final_score'],
            "credit_grade": fusion_result['credit_grade'],
            "component_scores": fusion_result['component_scores'],
            "weights": fusion_result['weights'],
            "market_context": fusion_result['market_context'],
            "calculation_timestamp": fusion_result['calculation_timestamp']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating final score: {str(e)}")


@router.get("/companies/{company_id}/trends")
async def get_score_trends(company_id: int, db: Session = Depends(get_db)):
    """
    Get historical score trends and analysis for a company.
    """
    try:
        company = db.query(Company).filter(Company.id == company_id).first()
        if not company:
            raise HTTPException(status_code=404, detail=f"Company with ID {company_id} not found")
        
        trends = fusion_service.get_score_trends(company_id)
        
        return {
            "company_id": company_id,
            "company_symbol": company.symbol,
            "current_score": trends['current_score'],
            "current_grade": trends['current_grade'],
            "trends": trends['trends'],
            "last_calculation": trends.get('calculation_timestamp'),
            "analysis_timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting score trends: {str(e)}")


@router.post("/models/train")
async def train_structured_model(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Train or retrain the structured Random Forest model.
    This is a background task as training can take time.
    """
    try:
        def train_model():
            try:
                result = structured_service.train()
                print(f"Model training completed: {result}")
            except Exception as e:
                print(f"Model training failed: {e}")
        
        background_tasks.add_task(train_model)
        
        return {
            "message": "Model training started in background",
            "status": "initiated",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initiating model training: {str(e)}")


@router.post("/companies")
async def add_company(
    symbol: str,
    name: str,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Add a new company to the tracking system.
    """
    try:
        # Check if company already exists
        existing = db.query(Company).filter(Company.symbol == symbol).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Company with symbol {symbol} already exists")
        
        # Create new company
        company = Company(
            symbol=symbol.upper(),
            name=name,
            sector=sector,
            industry=industry
        )
        
        db.add(company)
        db.commit()
        db.refresh(company)
        
        # Add to scheduler tracking (background task)
        if background_tasks:
            background_tasks.add_task(
                scheduler.add_company,
                symbol.upper(),
                name,
                sector,
                industry
            )
        
        return {
            "message": f"Company {symbol} added successfully",
            "company_id": company.id,
            "symbol": company.symbol,
            "name": company.name,
            "created_at": company.created_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error adding company: {str(e)}")


@router.post("/data/fetch/financial")
async def trigger_financial_fetch(background_tasks: BackgroundTasks):
    """
    Manually trigger financial data fetch for all companies.
    """
    try:
        background_tasks.add_task(scheduler.fetch_financial_data)
        
        return {
            "message": "Financial data fetch initiated",
            "status": "started",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error triggering financial fetch: {str(e)}")


@router.post("/data/fetch/news")
async def trigger_news_fetch(background_tasks: BackgroundTasks):
    """
    Manually trigger news data fetch for all companies.
    """
    try:
        background_tasks.add_task(scheduler.fetch_news_data)
        
        return {
            "message": "News data fetch initiated",
            "status": "started",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error triggering news fetch: {str(e)}")


@router.post("/sentiment/analyze")
async def analyze_text_sentiment(text: str):
    """
    Analyze sentiment of arbitrary text using FinBERT (for testing purposes).
    """
    try:
        if not text or len(text.strip()) < 5:
            raise HTTPException(status_code=400, detail="Text must be at least 5 characters long")
        
        result = unstructured_service.analyze_single_text(text)
        
        return {
            "input_text": text,
            "sentiment_analysis": result,
            "analysis_timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint to verify system status.
    """
    try:
        # Check database connection
        db = next(get_db())
        db.execute("SELECT 1")
        db.close()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "version": "1.0.0",
            "services": {
                "database": "connected",
                "api": "running"
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )
