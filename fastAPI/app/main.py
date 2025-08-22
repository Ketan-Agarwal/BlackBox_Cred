import asyncio
import datetime
import os
from fastapi import Depends, FastAPI, HTTPException, Query
import httpx
from pydantic import BaseModel
from sqlalchemy import select
from db import init_db
from pipeline.main_pipeline import run_credit_pipeline_with_db_cache
from dependencies import get_db
from models import Analysis, Company
from routes import scores
from dotenv import load_dotenv
app = FastAPI(title="Credit Scoring Service", version="1.0.0")
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost:3000"] for Next.js dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register routes
# app.include_router(issuers.router, prefix="/api/issuers", tags=["Issuers"])
# app.include_router(scores.router, prefix="/api/scores", tags=["Scores"])
# app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])

@app.on_event("startup")
async def on_startup():
    await init_db()


@app.get("/")
def root():
    return {"message": "Credit Scoring API is running"}

FRED_API_KEY = os.getenv("FRED_API_KEY", "1a39ebc94e4984ff4091baa2f84c0ba7")
NEON_DSN = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_CTOegl5r6oXV@ep-billowing-pine-adshdcaw-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require")



class AnalysisRequest(BaseModel):
    company_name: str
    ticker: str


@app.post("/analyze-and-wait")
async def analyze_and_wait(request: AnalysisRequest, db: AsyncSession = Depends(get_db)):
    """
    1. Check if cached analysis exists within 6 hours.
    2. If exists, return it.
    3. Otherwise, run pipeline, save new analysis, return it.
    """

    if not db:
        raise HTTPException(status_code=503, detail="Database service is unavailable.")

    try:
        result = await db.execute(select(Company).where(Company.ticker == request.ticker))
        company = result.scalar_one_or_none()

        if company:
            analysis_result = await db.execute(
                select(Analysis)
                .where(Analysis.ticker == request.ticker)
                .order_by(Analysis.created_at.desc())
                .limit(1)
            )
            latest_analysis = analysis_result.scalar_one_or_none()

            if latest_analysis:
                hours_diff = (datetime.datetime.utcnow() - latest_analysis.created_at).total_seconds() / 3600
                if hours_diff < 6:
                    return latest_analysis

        report = await asyncio.to_thread(
            run_credit_pipeline_with_db_cache,
            company_name=request.company_name,
            ticker=request.ticker,
            fred_api_key=FRED_API_KEY,
        )

        if not company:
            company = Company(ticker=request.ticker, name=request.company_name)
            db.add(company)
            await db.commit()

        entry = Analysis(
            ticker=request.ticker,
            report=report,
        )
        db.add(entry)
        await db.commit()
        await db.refresh(entry)

        return entry

    except Exception as e:
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
