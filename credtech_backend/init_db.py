"""
Database initialization script for CredTech Backend.
This script creates the database schema and adds sample data.
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / 'app'))

from app.core.config import settings
from app.db.session import create_tables, get_db_session
from app.db.models import Company, StructuredData, UnstructuredData, ModelMetadata


def create_sample_companies():
    """Create sample companies for testing."""
    db = get_db_session()
    
    sample_companies = [
        {
            'symbol': 'AAPL',
            'name': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'market_cap': 3000000000000  # $3T
        },
        {
            'symbol': 'MSFT',
            'name': 'Microsoft Corporation',
            'sector': 'Technology',
            'industry': 'Software',
            'market_cap': 2800000000000  # $2.8T
        },
        {
            'symbol': 'GOOGL',
            'name': 'Alphabet Inc.',
            'sector': 'Technology',
            'industry': 'Internet Services',
            'market_cap': 1800000000000  # $1.8T
        },
        {
            'symbol': 'AMZN',
            'name': 'Amazon.com Inc.',
            'sector': 'Consumer Discretionary',
            'industry': 'E-commerce',
            'market_cap': 1500000000000  # $1.5T
        },
        {
            'symbol': 'TSLA',
            'name': 'Tesla Inc.',
            'sector': 'Consumer Discretionary',
            'industry': 'Electric Vehicles',
            'market_cap': 800000000000  # $800B
        },
        {
            'symbol': 'JPM',
            'name': 'JPMorgan Chase & Co.',
            'sector': 'Financials',
            'industry': 'Banking',
            'market_cap': 500000000000  # $500B
        },
        {
            'symbol': 'BAC',
            'name': 'Bank of America Corp.',
            'sector': 'Financials',
            'industry': 'Banking',
            'market_cap': 300000000000  # $300B
        },
        {
            'symbol': 'WMT',
            'name': 'Walmart Inc.',
            'sector': 'Consumer Staples',
            'industry': 'Retail',
            'market_cap': 450000000000  # $450B
        },
        {
            'symbol': 'JNJ',
            'name': 'Johnson & Johnson',
            'sector': 'Healthcare',
            'industry': 'Pharmaceuticals',
            'market_cap': 400000000000  # $400B
        },
        {
            'symbol': 'V',
            'name': 'Visa Inc.',
            'sector': 'Financials',
            'industry': 'Payment Processing',
            'market_cap': 520000000000  # $520B
        }
    ]
    
    try:
        for company_data in sample_companies:
            # Check if company already exists
            existing = db.query(Company).filter(Company.symbol == company_data['symbol']).first()
            if not existing:
                company = Company(**company_data)
                db.add(company)
                print(f"Added company: {company_data['symbol']} - {company_data['name']}")
        
        db.commit()
        print(f"✅ Successfully added {len(sample_companies)} sample companies")
        
    except Exception as e:
        print(f"❌ Error adding sample companies: {e}")
        db.rollback()
    finally:
        db.close()


def create_sample_structured_data():
    """Create sample structured financial data."""
    db = get_db_session()
    
    try:
        companies = db.query(Company).all()
        
        for company in companies:
            # Create sample financial data with realistic values
            if company.symbol == 'AAPL':
                # Strong financial profile
                structured_data = StructuredData(
                    company_id=company.id,
                    current_ratio=1.07,
                    quick_ratio=0.95,
                    debt_to_equity=1.69,
                    return_on_equity=0.175,
                    return_on_assets=0.125,
                    operating_margin=0.30,
                    net_margin=0.25,
                    asset_turnover=1.12,
                    inventory_turnover=15.5,
                    total_assets=365000000000,
                    total_liabilities=290000000000,
                    total_equity=75000000000,
                    short_term_debt=15000000000,
                    long_term_debt=110000000000,
                    revenue=394000000000,
                    operating_income=115000000000,
                    net_income=100000000000,
                    ebitda=125000000000,
                    stock_price=185.0,
                    market_cap=3000000000000,
                    shares_outstanding=16200000000,
                    volatility=0.25,
                    data_date=datetime.utcnow()
                )
            elif company.symbol == 'TSLA':
                # More volatile profile
                structured_data = StructuredData(
                    company_id=company.id,
                    current_ratio=1.29,
                    quick_ratio=0.88,
                    debt_to_equity=0.35,
                    return_on_equity=0.19,
                    return_on_assets=0.08,
                    operating_margin=0.095,
                    net_margin=0.075,
                    asset_turnover=0.68,
                    inventory_turnover=6.8,
                    total_assets=106000000000,
                    total_liabilities=43000000000,
                    total_equity=63000000000,
                    short_term_debt=2500000000,
                    long_term_debt=5000000000,
                    revenue=96000000000,
                    operating_income=8900000000,
                    net_income=7200000000,
                    ebitda=13500000000,
                    stock_price=250.0,
                    market_cap=800000000000,
                    shares_outstanding=3200000000,
                    volatility=0.45,
                    data_date=datetime.utcnow()
                )
            else:
                # Default reasonable values for other companies
                structured_data = StructuredData(
                    company_id=company.id,
                    current_ratio=1.2,
                    quick_ratio=0.9,
                    debt_to_equity=0.8,
                    return_on_equity=0.15,
                    return_on_assets=0.08,
                    operating_margin=0.15,
                    net_margin=0.12,
                    asset_turnover=0.8,
                    inventory_turnover=8.0,
                    total_assets=200000000000,
                    total_liabilities=120000000000,
                    total_equity=80000000000,
                    short_term_debt=10000000000,
                    long_term_debt=50000000000,
                    revenue=150000000000,
                    operating_income=22500000000,
                    net_income=18000000000,
                    ebitda=30000000000,
                    stock_price=100.0,
                    market_cap=company.market_cap or 500000000000,
                    shares_outstanding=5000000000,
                    volatility=0.3,
                    data_date=datetime.utcnow()
                )
            
            db.add(structured_data)
        
        db.commit()
        print(f"✅ Successfully added structured data for {len(companies)} companies")
        
    except Exception as e:
        print(f"❌ Error adding structured data: {e}")
        db.rollback()
    finally:
        db.close()


def create_sample_news_data():
    """Create sample news data for testing."""
    db = get_db_session()
    
    sample_news = [
        {
            'symbol': 'AAPL',
            'headlines': [
                'Apple Reports Record Q3 Earnings with Strong iPhone Sales',
                'Apple Announces New AI Features Coming to iOS',
                'Apple Stock Rises on Strong Services Revenue Growth',
                'Apple Expands Manufacturing Operations in India',
                'Apple Vision Pro Sales Exceed Initial Expectations'
            ]
        },
        {
            'symbol': 'TSLA',
            'headlines': [
                'Tesla Delivers Record Number of Vehicles in Q3',
                'Tesla Autopilot Technology Receives Safety Recognition',
                'Tesla Energy Division Shows Strong Growth in Solar',
                'Tesla Stock Volatile on Production Guidance Update',
                'Tesla Cybertruck Production Ramp Ahead of Schedule'
            ]
        },
        {
            'symbol': 'MSFT',
            'headlines': [
                'Microsoft Azure Growth Accelerates with AI Services',
                'Microsoft Teams Adds New Productivity Features',
                'Microsoft Announces Major Cloud Computing Investment',
                'Microsoft Stock Hits New High on AI Momentum',
                'Microsoft Office 365 Subscriber Growth Continues'
            ]
        }
    ]
    
    try:
        for news_group in sample_news:
            # Get company
            company = db.query(Company).filter(Company.symbol == news_group['symbol']).first()
            if not company:
                continue
                
            for i, headline in enumerate(news_group['headlines']):
                news_data = UnstructuredData(
                    company_id=company.id,
                    headline=headline,
                    content=f"Sample news content for: {headline}",
                    source="Sample News",
                    url=f"https://example.com/news/{i}",
                    published_at=datetime.utcnow() - timedelta(hours=i*6),
                    sentiment_score=0.2 if 'Growth' in headline or 'Record' in headline else -0.1 if 'Volatile' in headline else 0.0,
                    sentiment_label='positive' if 'Growth' in headline or 'Record' in headline else 'negative' if 'Volatile' in headline else 'neutral',
                    finbert_confidence=0.85,
                    processed_score=70 if 'Growth' in headline or 'Record' in headline else 40 if 'Volatile' in headline else 50
                )
                db.add(news_data)
        
        db.commit()
        print("✅ Successfully added sample news data")
        
    except Exception as e:
        print(f"❌ Error adding news data: {e}")
        db.rollback()
    finally:
        db.close()


def create_model_metadata():
    """Create model metadata record."""
    db = get_db_session()
    
    try:
        metadata = ModelMetadata(
            model_type='random_forest',
            model_version='1.0',
            model_path=settings.random_forest_model_path,
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85,
            training_data_size=1000,
            training_date=datetime.utcnow(),
            features_used=[
                'current_ratio', 'quick_ratio', 'debt_to_equity', 'return_on_equity',
                'return_on_assets', 'operating_margin', 'net_margin', 'asset_turnover',
                'inventory_turnover', 'kmv_distance_to_default', 'altman_z_score'
            ],
            is_active=True
        )
        
        db.add(metadata)
        db.commit()
        print("✅ Successfully added model metadata")
        
    except Exception as e:
        print(f"❌ Error adding model metadata: {e}")
        db.rollback()
    finally:
        db.close()


def main():
    """Main initialization function."""
    print("=" * 60)
    print("  CredTech Backend - Database Initialization")
    print("=" * 60)
    print()
    
    try:
        # Create database tables
        print("📋 Creating database tables...")
        create_tables()
        print("✅ Database tables created successfully")
        print()
        
        # Add sample data
        print("📊 Adding sample companies...")
        create_sample_companies()
        print()
        
        print("💰 Adding sample structured data...")
        create_sample_structured_data()
        print()
        
        print("📰 Adding sample news data...")
        create_sample_news_data()
        print()
        
        print("🤖 Adding model metadata...")
        create_model_metadata()
        print()
        
        print("=" * 60)
        print("🎉 Database initialization completed successfully!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Start the application: uvicorn app.main:app --reload")
        print("2. Visit API docs: http://127.0.0.1:8000/docs")
        print("3. Test the endpoints with sample data")
        print()
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
