# api_main.py (Simplified Version)

import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# Import your main pipeline function and DB interface
from main_pipeline import run_credit_pipeline_with_db_cache
from neon_db_interface import NeonDBInterface

# --- Configuration ---
FRED_API_KEY = os.getenv("FRED_API_KEY", "1a39ebc94e4984ff4091baa2f84c0ba7")
NEON_DSN = os.getenv("NEON_DSN", "postgresql://neondb_owner:npg_CTOegl5r6oXV@ep-billowing-pine-adshdcaw-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Credit Risk Analysis API",
    description="A synchronous API to run a comprehensive credit risk assessment pipeline."
)

# --- Database Interface ---
# Initialize the DB interface once and reuse it.
try:
    db_interface = NeonDBInterface(dsn=NEON_DSN)
    logger.info("Successfully connected to NeonDB.")
except Exception as e:
    logger.error(f"Failed to connect to NeonDB: {e}")
    db_interface = None

# --- Pydantic Models ---
class AnalysisRequest(BaseModel):
    company_name: str
    ticker: str

# --- The Single, Simplified API Endpoint ---

@app.post("/analyze-and-wait")
async def analyze_and_wait(request: AnalysisRequest):
    """
    Accepts a company analysis request.
    1. Checks for a valid cached result in the database.
    2. If found, returns it immediately.
    3. If not found, runs the full pipeline, waits for it to complete,
       and then returns the new result.
    """
    logger.info(f"Received synchronous analysis request for {request.ticker}...")
    
    if not db_interface:
        raise HTTPException(status_code=503, detail="Database service is unavailable.")
        
    try:
        # Directly call your main pipeline function and wait for the result.
        # This function contains all the required cache-checking logic.
        report = run_credit_pipeline_with_db_cache(
            company_name=request.company_name,
            ticker=request.ticker,
            fred_api_key=FRED_API_KEY,
            db_interface=db_interface
        )
        
        logger.info(f"Successfully processed request for {request.ticker}.")
        return report

    except Exception as e:
        logger.error(f"An error occurred during the analysis for {request.ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")