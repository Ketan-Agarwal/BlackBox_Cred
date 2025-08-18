"""
Main FastAPI application for the CredTech backend system.
"""
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.db.session import create_tables
from app.api.endpoints import router as api_router
from app.jobs.scheduler import DataIngestionScheduler

# Global scheduler instance
scheduler = DataIngestionScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Handles startup and shutdown events.
    """
    # Startup
    print("Starting CredTech Backend System...")
    
    # Create database tables
    try:
        create_tables()
        print("Database tables created/verified")
    except Exception as e:
        print(f"Error creating database tables: {e}")
    
    # Start data ingestion scheduler
    try:
        scheduler.start_scheduler()
        print("Data ingestion scheduler started")
    except Exception as e:
        print(f"Error starting scheduler: {e}")
    
    # Add some default companies for demonstration
    try:
        scheduler.add_company("AAPL", "Apple Inc.", "Technology", "Consumer Electronics")
        scheduler.add_company("MSFT", "Microsoft Corporation", "Technology", "Software")
        scheduler.add_company("GOOGL", "Alphabet Inc.", "Technology", "Internet Services")
        scheduler.add_company("AMZN", "Amazon.com Inc.", "Consumer Discretionary", "E-commerce")
        scheduler.add_company("TSLA", "Tesla Inc.", "Consumer Discretionary", "Electric Vehicles")
        scheduler.add_company("JPM", "JPMorgan Chase & Co.", "Financials", "Banking")
        scheduler.add_company("BAC", "Bank of America Corp.", "Financials", "Banking")
        scheduler.add_company("WMT", "Walmart Inc.", "Consumer Staples", "Retail")
        scheduler.add_company("JNJ", "Johnson & Johnson", "Healthcare", "Pharmaceuticals")
        scheduler.add_company("V", "Visa Inc.", "Financials", "Payment Processing")
        print("Default companies added to tracking system")
    except Exception as e:
        print(f"Note: Error adding default companies (may already exist): {e}")
    
    print("CredTech Backend System startup completed")
    
    yield
    
    # Shutdown
    print("Shutting down CredTech Backend System...")
    
    try:
        scheduler.stop_scheduler()
        print("Data ingestion scheduler stopped")
    except Exception as e:
        print(f"Error stopping scheduler: {e}")
    
    print("CredTech Backend System shutdown completed")


# Create FastAPI application
app = FastAPI(
    title="CredTech Backend API",
    description="""
    ## CredTech Dynamic Hybrid Expert Model API
    
    This API provides comprehensive credit score analysis using a Dynamic Hybrid Expert Model
    that combines structured financial data analysis with unstructured sentiment analysis.
    
    ### Key Features:
    
    * **Structured Analysis**: KMV Distance-to-Default, Altman Z-Score, and Random Forest models
    * **Unstructured Analysis**: FinBERT sentiment analysis of news and market data
    * **Dynamic Fusion**: VIX-based dynamic weighting of model components
    * **Explainable AI**: Detailed explanations using SHAP and plain language summaries
    * **Real-time Data**: Automated ingestion of financial data and news
    
    ### Main Endpoints:
    
    * `GET /api/companies` - List all companies with latest scores
    * `GET /api/companies/{id}/explanation` - Comprehensive credit analysis
    * `GET /api/companies/{id}/scores/*` - Detailed component scores
    
    ### Model Architecture:
    
    The system uses a sophisticated multi-model approach:
    1. **Structured Model**: Random Forest trained on KMV DD, Z-Score, and financial ratios
    2. **Unstructured Model**: FinBERT for sentiment analysis of news and text data
    3. **Fusion Engine**: Dynamic weighting based on market volatility (VIX)
    4. **Explainability Engine**: SHAP values and natural language explanations
    """,
    version="1.0.0",
    contact={
        "name": "CredTech Hackathon Team",
        "email": "team@credtech.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api", tags=["Credit Analysis"])


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/status")
async def status():
    """
    System status endpoint.
    """
    return {
        "system": "CredTech Backend API",
        "status": "running",
        "version": "1.0.0",
        "architecture": "Dynamic Hybrid Expert Model",
        "components": {
            "structured_model": "KMV + Z-Score + Random Forest",
            "unstructured_model": "FinBERT Sentiment Analysis",
            "fusion_engine": "VIX-based Dynamic Weighting",
            "explainability": "SHAP + Natural Language"
        },
        "endpoints": {
            "companies": "/api/companies",
            "explanation": "/api/companies/{id}/explanation",
            "health": "/api/health",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
