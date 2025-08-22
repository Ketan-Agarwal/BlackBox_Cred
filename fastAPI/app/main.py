from fastapi import FastAPI
from routes import scores

app = FastAPI(title="Credit Scoring Service", version="1.0.0")
from fastapi.middleware.cors import CORSMiddleware

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

@app.get("/")
def root():
    return {"message": "Credit Scoring API is running"}

