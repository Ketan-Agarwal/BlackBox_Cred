"""
APScheduler job definitions for automated data ingestion.
"""
import requests
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from elasticsearch import Elasticsearch
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db_session
from app.db.models import Company, StructuredData, UnstructuredData


class DataIngestionScheduler:
    """Scheduler for automated data ingestion jobs."""
    
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.es_client = self._init_elasticsearch()
        
    def _init_elasticsearch(self) -> Optional[Elasticsearch]:
        """Initialize Elasticsearch client."""
        try:
            es = Elasticsearch([settings.elasticsearch_url])
            return es
        except Exception as e:
            print(f"Warning: Could not connect to Elasticsearch: {e}")
            return None
    
    def _get_company_list(self) -> List[Company]:
        """Get list of companies to track."""
        db = get_db_session()
        try:
            companies = db.query(Company).all()
            return companies
        finally:
            db.close()
    
    def fetch_financial_data(self):
        """
        Daily job to fetch financial ratios and market data.
        This job runs daily at 6 AM EST (after market close).
        """
        print(f"Starting financial data fetch at {datetime.utcnow()}")
        
        companies = self._get_company_list()
        db = get_db_session()
        
        try:
            for company in companies:
                try:
                    # Fetch data using yfinance
                    ticker = yf.Ticker(company.symbol)
                    
                    # Get financial data
                    info = ticker.info
                    financials = ticker.financials
                    balance_sheet = ticker.balance_sheet
                    
                    # Get recent stock data for volatility calculation
                    hist = ticker.history(period="1y")
                    if not hist.empty:
                        returns = hist['Close'].pct_change().dropna()
                        volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
                    else:
                        volatility = 0.3  # Default volatility
                    
                    # Extract financial metrics
                    structured_data = StructuredData(
                        company_id=company.id,
                        # Financial ratios (calculated from available data)
                        current_ratio=self._safe_divide(
                            info.get('totalCurrentAssets', 0),
                            info.get('totalCurrentLiabilities', 1)
                        ),
                        quick_ratio=self._safe_divide(
                            info.get('totalCurrentAssets', 0) - info.get('inventory', 0),
                            info.get('totalCurrentLiabilities', 1)
                        ),
                        debt_to_equity=self._safe_divide(
                            info.get('totalDebt', 0),
                            info.get('totalStockholderEquity', 1)
                        ),
                        return_on_equity=info.get('returnOnEquity', 0),
                        return_on_assets=info.get('returnOnAssets', 0),
                        operating_margin=info.get('operatingMargins', 0),
                        net_margin=info.get('profitMargins', 0),
                        asset_turnover=self._safe_divide(
                            info.get('totalRevenue', 0),
                            info.get('totalAssets', 1)
                        ),
                        
                        # Balance sheet items
                        total_assets=info.get('totalAssets', 0),
                        total_liabilities=info.get('totalDebt', 0),
                        total_equity=info.get('totalStockholderEquity', 0),
                        short_term_debt=info.get('shortTermDebt', 0),
                        long_term_debt=info.get('longTermDebt', 0),
                        
                        # Income statement items
                        revenue=info.get('totalRevenue', 0),
                        operating_income=info.get('operatingIncome', 0),
                        net_income=info.get('netIncomeToCommon', 0),
                        ebitda=info.get('ebitda', 0),
                        
                        # Market data
                        stock_price=info.get('currentPrice', 0),
                        market_cap=info.get('marketCap', 0),
                        shares_outstanding=info.get('sharesOutstanding', 0),
                        volatility=volatility,
                        
                        data_date=datetime.utcnow()
                    )
                    
                    db.add(structured_data)
                    print(f"Fetched financial data for {company.symbol}")
                    
                except Exception as e:
                    print(f"Error fetching data for {company.symbol}: {e}")
                    continue
            
            db.commit()
            print("Financial data fetch completed successfully")
            
        except Exception as e:
            print(f"Error in financial data fetch job: {e}")
            db.rollback()
        finally:
            db.close()
    
    def fetch_news_data(self):
        """
        Hourly job to fetch news headlines and store in Elasticsearch.
        This job runs every hour during market hours.
        """
        print(f"Starting news data fetch at {datetime.utcnow()}")
        
        if not settings.news_api_key:
            print("Warning: News API key not configured, skipping news fetch")
            return
        
        companies = self._get_company_list()
        db = get_db_session()
        
        try:
            for company in companies:
                try:
                    # Fetch news from News API
                    news_articles = self._fetch_company_news(company.symbol, company.name)
                    
                    for article in news_articles:
                        # Store in Elasticsearch (if available)
                        es_doc_id = None
                        if self.es_client:
                            es_doc_id = self._store_in_elasticsearch(article, company.symbol)
                        
                        # Store reference in PostgreSQL
                        unstructured_data = UnstructuredData(
                            company_id=company.id,
                            headline=article.get('title', ''),
                            content=article.get('description', ''),
                            source=article.get('source', {}).get('name', ''),
                            url=article.get('url', ''),
                            published_at=self._parse_date(article.get('publishedAt')),
                            elasticsearch_doc_id=es_doc_id
                        )
                        
                        db.add(unstructured_data)
                    
                    print(f"Fetched {len(news_articles)} news articles for {company.symbol}")
                    
                except Exception as e:
                    print(f"Error fetching news for {company.symbol}: {e}")
                    continue
            
            db.commit()
            print("News data fetch completed successfully")
            
        except Exception as e:
            print(f"Error in news data fetch job: {e}")
            db.rollback()
        finally:
            db.close()
    
    def _fetch_company_news(self, symbol: str, company_name: str) -> List[Dict]:
        """
        Fetch news articles for a company from News API.
        
        Args:
            symbol: Company stock symbol
            company_name: Company name
            
        Returns:
            List of news articles
        """
        try:
            # Search for news using both symbol and company name
            query = f"{symbol} OR {company_name}"
            
            params = {
                'q': query,
                'apiKey': settings.news_api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'from': (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                'pageSize': 10
            }
            
            response = requests.get(settings.news_api_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data.get('articles', [])
            
        except Exception as e:
            print(f"Error fetching news from API: {e}")
            return []
    
    def _store_in_elasticsearch(self, article: Dict, symbol: str) -> Optional[str]:
        """
        Store news article in Elasticsearch.
        
        Args:
            article: News article data
            symbol: Company symbol
            
        Returns:
            Elasticsearch document ID
        """
        if not self.es_client:
            return None
        
        try:
            doc = {
                'symbol': symbol,
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'content': article.get('content', ''),
                'source': article.get('source', {}).get('name', ''),
                'url': article.get('url', ''),
                'published_at': article.get('publishedAt'),
                'indexed_at': datetime.utcnow().isoformat()
            }
            
            result = self.es_client.index(
                index=settings.elasticsearch_index_news,
                document=doc
            )
            
            return result['_id']
            
        except Exception as e:
            print(f"Error storing article in Elasticsearch: {e}")
            return None
    
    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """Safely divide two numbers, returning 0 if denominator is 0."""
        try:
            return float(numerator) / float(denominator) if denominator != 0 else 0.0
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not date_str:
            return None
        
        try:
            # Handle ISO format with timezone
            if 'T' in date_str:
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return datetime.fromisoformat(date_str)
        except ValueError:
            return None
    
    def start_scheduler(self):
        """Start the background scheduler."""
        # Daily financial data fetch at 6 AM EST
        self.scheduler.add_job(
            func=self.fetch_financial_data,
            trigger=CronTrigger(hour=6, minute=0, timezone='US/Eastern'),
            id='daily_financial_fetch',
            name='Daily Financial Data Fetch',
            replace_existing=True
        )
        
        # Hourly news data fetch during business hours (9 AM - 5 PM EST)
        self.scheduler.add_job(
            func=self.fetch_news_data,
            trigger=CronTrigger(hour='9-17', minute=0, timezone='US/Eastern'),
            id='hourly_news_fetch',
            name='Hourly News Data Fetch',
            replace_existing=True
        )
        
        # For development/testing - run news fetch every 30 minutes
        if settings.debug:
            self.scheduler.add_job(
                func=self.fetch_news_data,
                trigger='interval',
                minutes=30,
                id='debug_news_fetch',
                name='Debug News Data Fetch',
                replace_existing=True
            )
        
        self.scheduler.start()
        print("Data ingestion scheduler started")
    
    def stop_scheduler(self):
        """Stop the background scheduler."""
        self.scheduler.shutdown()
        print("Data ingestion scheduler stopped")
    
    def add_company(self, symbol: str, name: str, sector: str = None, industry: str = None):
        """
        Add a new company to track.
        
        Args:
            symbol: Stock symbol
            name: Company name
            sector: Business sector
            industry: Industry classification
        """
        db = get_db_session()
        
        try:
            # Check if company already exists
            existing = db.query(Company).filter(Company.symbol == symbol).first()
            if existing:
                print(f"Company {symbol} already exists")
                return
            
            company = Company(
                symbol=symbol,
                name=name,
                sector=sector,
                industry=industry
            )
            
            db.add(company)
            db.commit()
            print(f"Added company {symbol} - {name} to tracking list")
            
        except Exception as e:
            print(f"Error adding company: {e}")
            db.rollback()
        finally:
            db.close()
