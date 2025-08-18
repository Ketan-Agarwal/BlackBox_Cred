"""
Core configuration management for the CredTech backend system.
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Database Configuration
    database_url: str = Field(default="postgresql://username:password@localhost/credtech_db")
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="credtech_db")
    postgres_user: str = Field(default="username")
    postgres_password: str = Field(default="password")
    
    # Elasticsearch Configuration
    elasticsearch_url: str = Field(default="http://localhost:9200")
    elasticsearch_index_news: str = Field(default="credtech_news")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    debug: bool = Field(default=True)
    
    # News API Configuration
    news_api_key: Optional[str] = Field(default=None)
    news_api_url: str = Field(default="https://newsapi.org/v2/everything")
    
    # Model Configuration
    finbert_model_name: str = Field(default="ProsusAI/finbert")
    random_forest_model_path: str = Field(default="./models/structured_model.pkl")
    
    # Market Data Configuration
    vix_symbol: str = Field(default="^VIX")
    
    # Logging Configuration
    log_level: str = Field(default="INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
