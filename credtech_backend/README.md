# CredTech Backend - Dynamic Hybrid Expert Model

A sophisticated backend system for credit score analysis using a Dynamic Hybrid Expert Model that combines structured financial data analysis with unstructured sentiment analysis for maximum explainability and performance.

## 🏗️ Architecture Overview

The system implements a **Dynamic Hybrid Expert Model** with the following components:

### 1. Structured Score Engine
- **KMV Distance-to-Default Model**: Calculates probability of default using Merton model
- **Altman Z-Score**: Traditional bankruptcy prediction model
- **Random Forest Classifier**: ML model trained on financial ratios + KMV + Z-Score

### 2. Unstructured Score Engine
- **FinBERT Sentiment Analysis**: Pre-trained financial NLP model
- **News Aggregation**: Real-time news sentiment analysis
- **Score Normalization**: Converts sentiment to 0-100 credit score

### 3. Dynamic Fusion Engine
- **VIX-Based Weighting**: Dynamic model weighting based on market volatility
- **Real-time Market Data**: Fetches VIX for market condition assessment
- **Credit Grade Mapping**: Converts numerical scores to standard credit grades

### 4. Explainability Engine
- **SHAP Analysis**: Feature importance and contribution explanations
- **Plain Language Summaries**: Human-readable explanations
- **Trend Analysis**: Historical score tracking and pattern identification

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+**
2. **PostgreSQL** (running on localhost:5432)
3. **Elasticsearch** (running on localhost:9200) - Optional but recommended
4. **Git**

### Installation

1. **Clone and navigate to the project**:
```bash
cd c:\Users\asus\Desktop\CredCode\credtech_backend
```

2. **Create and activate virtual environment**:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
copy .env.example .env
# Edit .env file with your database credentials
```

5. **Create PostgreSQL database**:
```sql
-- Connect to PostgreSQL as superuser
CREATE DATABASE credtech_db;
CREATE USER credtech_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE credtech_db TO credtech_user;
```

6. **Update .env file**:
```bash
DATABASE_URL=postgresql://credtech_user:your_password@localhost/credtech_db
POSTGRES_USER=credtech_user
POSTGRES_PASSWORD=your_password
POSTGRES_DB=credtech_db

# Optional: Add News API key for real news data
NEWS_API_KEY=your_news_api_key_here
```

### Running the Application

1. **Start the server**:
```bash
uvicorn app.main:app --reload
```

2. **Access the API**:
- **API Documentation**: http://127.0.0.1:8000/docs
- **Alternative Docs**: http://127.0.0.1:8000/redoc
- **Health Check**: http://127.0.0.1:8000/api/health
- **System Status**: http://127.0.0.1:8000/status

## 📡 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/companies` | List all companies with latest scores |
| `GET` | `/api/companies/{id}/explanation` | **Main endpoint**: Comprehensive credit analysis |
| `GET` | `/api/companies/{id}/scores/structured` | Detailed structured model analysis |
| `GET` | `/api/companies/{id}/scores/unstructured` | Detailed sentiment analysis |
| `GET` | `/api/companies/{id}/scores/final` | Final fused score with breakdown |
| `GET` | `/api/companies/{id}/trends` | Historical trends and analysis |

### Management Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/companies` | Add new company to tracking |
| `POST` | `/api/models/train` | Train/retrain the Random Forest model |
| `POST` | `/api/data/fetch/financial` | Manually trigger financial data fetch |
| `POST` | `/api/data/fetch/news` | Manually trigger news data fetch |
| `POST` | `/api/sentiment/analyze` | Analyze sentiment of arbitrary text |

## 🎯 Key Features for Hackathon Evaluation

### 1. **Maximum Explainability**
- **Plain Language Summaries**: Every score includes human-readable explanations
- **Feature Contributions**: SHAP analysis shows exactly how each financial metric impacts the score
- **Latest Events Reasoning**: Direct links between news headlines and score changes
- **Trend Indicators**: Clear short-term (7-day) and long-term (90-day) trend analysis

### 2. **Performance & Accuracy**
- **Multi-Model Ensemble**: Combines traditional finance models (KMV, Z-Score) with modern ML
- **Dynamic Weighting**: Adapts to market conditions using VIX volatility index
- **Real-time Data**: Automated data ingestion ensures fresh analysis
- **Validated Models**: Uses proven financial models (KMV Distance-to-Default, Altman Z-Score)

### 3. **Industry Standards**
- **Credit Grades**: Standard credit rating scale (AAA to D)
- **Financial Ratios**: Industry-standard financial metrics
- **Risk Assessment**: Probability-based default prediction
- **Market Integration**: Real-time market volatility consideration

## 🔬 Technical Implementation

### Data Flow
1. **Ingestion**: Scheduled jobs fetch financial data (yfinance) and news (News API)
2. **Processing**: Structured and unstructured models process data independently
3. **Fusion**: Dynamic weighting based on current market volatility (VIX)
4. **Storage**: Results stored in PostgreSQL with Elasticsearch for news search
5. **Explanation**: SHAP analysis and template-based explanations

### Model Details

#### Structured Model Pipeline
```python
# 1. Calculate KMV Distance-to-Default
dd = kmv_model.calculate(market_cap, debt, volatility)

# 2. Calculate Altman Z-Score  
z_score = altman_model.calculate(financial_ratios)

# 3. Random Forest prediction
features = [ratios..., dd, z_score]
structured_score = random_forest.predict_proba(features)
```

#### Unstructured Model Pipeline
```python
# 1. Fetch recent news headlines
news = fetch_news(company, days=7)

# 2. FinBERT sentiment analysis
sentiments = [finbert.analyze(headline) for headline in news]

# 3. Aggregate and normalize
unstructured_score = aggregate_sentiment(sentiments) * 100
```

#### Dynamic Fusion
```python
# 1. Get current market volatility
vix = yfinance.get_vix()

# 2. Calculate dynamic weights
news_weight = calculate_weight(vix)  # Higher VIX = more news weight

# 3. Fuse scores
final_score = (structured_score * (1-news_weight)) + (unstructured_score * news_weight)
```

## 📊 Sample API Response

### Company Explanation Response
```json
{
  "company_symbol": "AAPL",
  "company_name": "Apple Inc.",
  "calculation_timestamp": "2025-08-18T10:30:00Z",
  "plain_language_summary": {
    "overall_assessment": "Apple demonstrates strong creditworthiness with an AA+ rating.",
    "key_strengths": ["Strong fundamental financial metrics", "Positive market sentiment"],
    "key_concerns": ["No significant concerns identified"],
    "market_impact": "Stable market conditions emphasizing fundamental metrics",
    "recommendation": "Favorable for investment and lending with standard terms."
  },
  "structured_model": {
    "random_forest_score": 88.5,
    "kmv_distance_to_default": 4.2,
    "altman_z_score": 3.8,
    "top_feature_contributions": [...]
  },
  "unstructured_model": {
    "finbert_score": 72.3,
    "latest_news_headline": "Apple Reports Strong Q3 Earnings",
    "sentiment_classification": "positive",
    "news_articles_analyzed": 15
  },
  "fusion_process": {
    "final_score": 85.7,
    "credit_grade": "AA+",
    "structured_weight": 0.65,
    "unstructured_weight": 0.35,
    "current_vix": 18.5
  },
  "trend_analysis": {
    "change_7d": 2.3,
    "change_90d": -1.1,
    "trend_7d": "improving",
    "trend_90d": "stable"
  }
}
```

## 🛠️ Development Commands

### Training the Model
```bash
curl -X POST "http://127.0.0.1:8000/api/models/train"
```

### Adding a Company
```bash
curl -X POST "http://127.0.0.1:8000/api/companies" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "NVDA", "name": "NVIDIA Corporation", "sector": "Technology"}'
```

### Testing Sentiment Analysis
```bash
curl -X POST "http://127.0.0.1:8000/api/sentiment/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "Company reports record profits and strong growth outlook"}'
```

## 🔧 Configuration

### Environment Variables
- `DATABASE_URL`: PostgreSQL connection string
- `ELASTICSEARCH_URL`: Elasticsearch endpoint (optional)
- `NEWS_API_KEY`: News API key for real news data
- `DEBUG`: Enable debug mode and extra logging
- `VIX_SYMBOL`: Symbol for market volatility (default: ^VIX)

### Scheduler Jobs
- **Daily (6 AM EST)**: Financial data fetch using yfinance
- **Hourly (9-5 EST)**: News data fetch using News API
- **Debug Mode**: Extra frequent updates for development

## 📈 Performance Metrics

The system tracks and provides:
- **Model Accuracy**: Random Forest classification accuracy
- **Response Time**: API endpoint performance
- **Data Freshness**: Last update timestamps
- **Coverage**: Number of companies and data points
- **Explanation Quality**: Feature importance rankings

## 🎯 Hackathon Deliverables

✅ **Complete Backend System**: Fully functional API with all required endpoints  
✅ **Explainable AI**: Detailed explanations for every prediction  
✅ **Real-time Integration**: Live market data and news integration  
✅ **Industry Standards**: Standard credit grades and financial metrics  
✅ **Scalable Architecture**: Modular design supporting easy extension  
✅ **Comprehensive Documentation**: API docs, setup guides, and examples  

## 🚀 Production Deployment

For production deployment:

1. **Environment**: Use production PostgreSQL and Elasticsearch clusters
2. **Security**: Add authentication and rate limiting
3. **Monitoring**: Implement logging and health checks
4. **Scaling**: Use container orchestration (Docker/Kubernetes)
5. **Data**: Configure real News API keys and data sources

## 📞 Support

For questions or issues during the hackathon:
- Check the API documentation at `/docs`
- Review log files for error details
- Verify database and service connections
- Test individual endpoints to isolate issues

---

**Built for the CredTech Hackathon 2025** 🏆  
*Dynamic Hybrid Expert Model for Maximum Explainability and Performance*
