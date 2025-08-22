from sqlalchemy import Column, String, Float, JSON, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID
import uuid
from app.db import Base

class Issuer(Base):
    __tablename__ = "issuers"
    issuer_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    ticker = Column(String, unique=True, index=True)
    sector = Column(String)
    country = Column(String)

class Score(Base):
    __tablename__ = "scores"
    issuer_id = Column(UUID(as_uuid=True), primary_key=True)
    ts = Column(TIMESTAMP, primary_key=True)
    fused_score = Column(Float, nullable=False)
    band = Column(String)
    model_version = Column(String)
    confidence = Column(Float)

class Explanation(Base):
    __tablename__ = "explanations"
    issuer_id = Column(UUID(as_uuid=True), primary_key=True)
    ts = Column(TIMESTAMP, primary_key=True)
    ebm_feature_json = Column(JSON)
    news_analysis_json = Column(JSON)
    fusion_summary_text = Column(String)
    report_uri = Column(String)

class Alert(Base):
    __tablename__ = "alerts"
    alert_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    issuer_id = Column(UUID(as_uuid=True))
    ts = Column(TIMESTAMP)
    score_prev = Column(Float)
    score_new = Column(Float)
    delta = Column(Float)
    rules_json = Column(JSON)
    severity = Column(String)
    top_drivers = Column(JSON)
    news_factors = Column(JSON)
    fusion_snapshot = Column(JSON)
