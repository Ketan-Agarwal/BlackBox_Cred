# api_main.py (Simplified Version)

import os
import logging
import time
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any
import httpx
# Import your main pipeline function and DB interface
from fastAPI.app.pipeline.main_pipeline import run_credit_pipeline_with_db_cache
from neon_db_interface import NeonDBInterface
from dotenv import load_dotenv
import os

load_dotenv() 
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
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost:3000"] for Next.js dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "d2jn8q1r01qj8a5k949gd2jn8q1r01qj8a5k94a0")

@app.get("/search-ticker")
async def search_ticker(q: str = Query(..., min_length=1)):
    url = f"https://finnhub.io/api/v1/search?q={q}&token={FINNHUB_API_KEY}"
    async with httpx.AsyncClient() as client:
        r = await client.get(url)
        data = r.json()

    # Finnhub returns { "count": X, "result": [{symbol, description}, ...] }
    results = [
        {"ticker": item["symbol"], "name": item["description"]}
        for item in data.get("result", [])
        if item.get("symbol") and item.get("description")
    ]

    return {"results": results[:10]}  # limit to 10
