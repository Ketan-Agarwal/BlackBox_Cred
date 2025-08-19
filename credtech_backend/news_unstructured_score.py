"""
news_unstructured_score.py

Enhanced pipeline to generate comprehensive risk scores (0-100) for companies based on 
recent news using NewsAPI and FinBERT. Returns ExpertAssessment compatible with MAESTRO global fusion.
"""

import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import datetime
import torch
import time
import re
import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter
from dataclasses import dataclass

# === FinBERT Model Setup ===
print("🤖 Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
labels = ['positive', 'negative', 'neutral']
print("✅ FinBERT model loaded successfully")

# === NewsAPI Configuration ===
NEWS_API_KEY = "a729fdf14377486f931f6fb1e0fc940e"
NEWS_API_BASE_URL = "https://newsapi.org/v2/everything"

@dataclass
class NewsAnalysisResult:
    """Structured container for comprehensive news analysis"""
    risk_score: float
    sentiment_score: float
    article_count: int
    risk_keywords: List[str]
    confidence: float
    temporal_trend: float
    explanation: str


class FinancialRiskDetector:
    """Advanced financial risk detection from news content"""
    
    def __init__(self):
        # Enhanced risk keyword dictionary with weights
        self.risk_categories = {
            'liquidity_crisis': {
                'keywords': ['cash flow', 'liquidity crisis', 'cash shortage', 'working capital', 
                           'credit facility', 'refinancing', 'cash burn', 'funding gap'],
                'weight': 1.5,  # High impact on credit risk
                'risk_boost': 25
            },
            'debt_distress': {
                'keywords': ['debt default', 'covenant violation', 'bankruptcy', 'insolvency',
                           'debt restructuring', 'creditor pressure', 'leverage concerns', 'debt burden'],
                'weight': 1.8,  # Very high impact
                'risk_boost': 35
            },
            'operational_issues': {
                'keywords': ['supply chain', 'production halt', 'strike', 'management departure',
                           'key personnel', 'operational disruption', 'facility closure'],
                'weight': 1.0,  # Moderate impact
                'risk_boost': 15
            },
            'regulatory_legal': {
                'keywords': ['lawsuit', 'investigation', 'regulatory penalty', 'compliance violation',
                           'fine', 'audit', 'legal action', 'regulatory scrutiny'],
                'weight': 1.2,  # Significant impact
                'risk_boost': 20
            },
            'market_competition': {
                'keywords': ['market share loss', 'competitive pressure', 'demand decline',
                           'pricing pressure', 'market downturn', 'revenue decline'],
                'weight': 0.8,  # Lower direct impact
                'risk_boost': 12
            },
            'cyber_technology': {
                'keywords': ['cyber attack', 'data breach', 'system failure', 'technology disruption',
                           'cybersecurity', 'digital transformation', 'IT outage'],
                'weight': 0.9,  # Moderate impact
                'risk_boost': 18
            }
        }
    
    def analyze_risk_factors(self, text: str) -> Dict[str, Any]:
        """Analyze text for financial risk factors and return detailed results"""
        text_lower = text.lower()
        detected_risks = {}
        total_risk_score = 0
        risk_keywords_found = []
        
        for category, info in self.risk_categories.items():
            matches = []
            for keyword in info['keywords']:
                if keyword in text_lower:
                    matches.append(keyword)
                    risk_keywords_found.append(keyword)
            
            if matches:
                category_score = len(matches) * info['weight'] * info['risk_boost']
                detected_risks[category] = {
                    'matches': matches,
                    'count': len(matches),
                    'score': category_score
                }
                total_risk_score += category_score
        
        return {
            'total_risk_score': min(100, total_risk_score),  # Cap at 100
            'categories': detected_risks,
            'keywords_found': risk_keywords_found,
            'risk_level': self._categorize_risk_level(total_risk_score)
        }
    
    def _categorize_risk_level(self, score: float) -> str:
        """Categorize risk level based on score"""
        if score < 20:
            return "LOW"
        elif score < 40:
            return "MODERATE"
        elif score < 70:
            return "HIGH"
        else:
            return "CRITICAL"


class EnhancedNewsAnalyzer:
    """Enhanced news analysis system using NewsAPI and FinBERT"""
    
    def __init__(self):
        self.risk_detector = FinancialRiskDetector()
        self.max_retries = 3
        self.retry_delay = 1
    
    def fetch_company_news(self, company: str, days_back: int = 7, max_articles: int = 20) -> List[Dict[str, Any]]:
        """
        Fetch recent news articles for a company using NewsAPI
        Enhanced version of the original function with better error handling
        """
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=days_back)
        
        # Build API request parameters
        params = {
            'q': f'"{company}" OR {company.replace(" ", " AND ")}',  # Better search query
            'from': start_date.isoformat(),
            'to': end_date.isoformat(),
            'sortBy': 'relevancy',  # Changed to relevancy for better results
            'language': 'en',
            'pageSize': min(max_articles, 100),  # API limit is 100
            'apiKey': NEWS_API_KEY
        }
        
        for attempt in range(self.max_retries):
            try:
                print(f"📡 Fetching news for {company} (attempt {attempt + 1})...")
                response = requests.get(NEWS_API_BASE_URL, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get('status') == 'error':
                    print(f"❌ NewsAPI Error: {data.get('message', 'Unknown error')}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return []
                
                articles = data.get('articles', [])
                
                # Filter out removed articles and duplicates
                valid_articles = []
                seen_titles = set()
                
                for article in articles:
                    title = article.get('title', '')
                    if (title and 
                        title not in seen_titles and 
                        title != '[Removed]' and
                        len(title) > 10):  # Filter out very short titles
                        
                        valid_articles.append({
                            'title': title,
                            'description': article.get('description', ''),
                            'content': article.get('content', ''),
                            'url': article.get('url', ''),
                            'publishedAt': article.get('publishedAt', ''),
                            'source': article.get('source', {}).get('name', 'Unknown')
                        })
                        seen_titles.add(title)
                
                print(f"✅ Retrieved {len(valid_articles)} valid articles for {company}")
                return valid_articles[:max_articles]
                
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    print("❌ All retry attempts failed")
                    return []
        
        return []
    
    def analyze_article_sentiment(self, title: str, description: str = "", content: str = "") -> Dict[str, float]:
        """
        Analyze sentiment of a single article using FinBERT
        Enhanced version with better text handling
        """
        try:
            # Combine all available text
            full_text = f"{title} {description} {content}".strip()
            
            # Truncate to reasonable length (FinBERT has token limits)
            if len(full_text) > 1000:
                full_text = full_text[:1000]
            
            # Tokenize and get model predictions
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs)
                scores = softmax(outputs.logits.detach().numpy(), axis=1)[0]
            
            # Extract individual sentiment scores
            positive_score = float(scores[labels.index('positive')])
            negative_score = float(scores[labels.index('negative')])
            neutral_score = float(scores[labels.index('neutral')])
            
            # Calculate net sentiment (-1 to 1, where -1 is very negative, 1 is very positive)
            net_sentiment = positive_score - negative_score
            
            return {
                'positive_score': positive_score,
                'negative_score': negative_score,
                'neutral_score': neutral_score,
                'net_sentiment': net_sentiment,
                'confidence': max(positive_score, negative_score, neutral_score)  # Confidence in prediction
            }
            
        except Exception as e:
            print(f"⚠️ Sentiment analysis error: {e}")
            return {
                'positive_score': 0.33,
                'negative_score': 0.33,
                'neutral_score': 0.34,
                'net_sentiment': 0.0,
                'confidence': 0.5
            }
    
    def calculate_temporal_trend(self, articles: List[Dict[str, Any]], sentiment_scores: List[float]) -> float:
        """Calculate temporal trend in sentiment (positive = improving, negative = deteriorating)"""
        try:
            if len(articles) < 2 or len(sentiment_scores) < 2:
                return 0.0
            
            # Create time-ordered list of sentiments
            article_sentiments = []
            for i, article in enumerate(articles):
                if i < len(sentiment_scores):
                    published_at = article.get('publishedAt', '')
                    try:
                        # Parse ISO date format
                        pub_date = datetime.datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        article_sentiments.append((pub_date, sentiment_scores[i]))
                    except:
                        # If date parsing fails, use index as proxy for time
                        article_sentiments.append((datetime.datetime.now() - datetime.timedelta(days=i), sentiment_scores[i]))
            
            # Sort by date
            article_sentiments.sort(key=lambda x: x[0])
            
            # Calculate trend using simple linear regression
            if len(article_sentiments) > 1:
                x_values = np.arange(len(article_sentiments))
                y_values = [sentiment for _, sentiment in article_sentiments]
                trend_slope = np.polyfit(x_values, y_values, 1)[0]
                return float(trend_slope)
            
            return 0.0
            
        except Exception as e:
            print(f"⚠️ Trend calculation error: {e}")
            return 0.0
    
    def compute_comprehensive_news_score(self, company: str, days_back: int = 7, 
                                       max_articles: int = 20) -> Dict[str, Any]:
        """
        Main function: Comprehensive news risk analysis for MAESTRO integration
        Enhanced version of the original fetch_news_sentiment function
        """
        start_time = time.time()
        
        try:
            print(f"🔍 Starting comprehensive news analysis for {company}...")
            
            # Step 1: Fetch news articles
            articles = self.fetch_company_news(company, days_back, max_articles)
            
            if not articles:
                print(f"⚠️ No articles found for {company}")
                return self._create_no_news_assessment(company, start_time)
            
            # Step 2: Analyze sentiment for each article
            sentiment_results = []
            risk_analysis_results = []
            all_risk_keywords = []
            
            for i, article in enumerate(articles):
                print(f"   📰 Analyzing article {i+1}/{len(articles)}: {article['title'][:50]}...")
                
                # Sentiment analysis
                sentiment = self.analyze_article_sentiment(
                    article['title'], 
                    article.get('description', ''), 
                    article.get('content', '')
                )
                sentiment_results.append(sentiment)
                
                # Risk factor analysis
                full_article_text = f"{article['title']} {article.get('description', '')} {article.get('content', '')}"
                risk_analysis = self.risk_detector.analyze_risk_factors(full_article_text)
                risk_analysis_results.append(risk_analysis)
                all_risk_keywords.extend(risk_analysis['keywords_found'])
            
            # Step 3: Calculate aggregate sentiment score (using original approach but enhanced)
            net_sentiments = [result['net_sentiment'] for result in sentiment_results]
            avg_net_sentiment = np.mean(net_sentiments) if net_sentiments else 0.0
            
            # Convert to 0-100 scale (like original function)
            base_sentiment_score = 50 * (avg_net_sentiment + 1)  # Original calculation
            
            # Step 4: Calculate risk factor boost
            risk_scores = [result['total_risk_score'] for result in risk_analysis_results]
            avg_risk_score = np.mean(risk_scores) if risk_scores else 0.0
            
            # Step 5: Calculate temporal trend
            sentiment_values = [50 * (s + 1) for s in net_sentiments]  # Convert to same scale
            temporal_trend = self.calculate_temporal_trend(articles, sentiment_values)
            
            # Step 6: Compute final risk score (inverse of sentiment, higher = more risk)
            final_risk_score = 100 - base_sentiment_score  # Invert sentiment to risk
            final_risk_score += avg_risk_score * 0.3  # Add risk factor boost
            
            # Temporal adjustment
            if temporal_trend < -2:  # Deteriorating sentiment
                final_risk_score += 10
            elif temporal_trend > 2:  # Improving sentiment
                final_risk_score -= 5
            
            # Bound the score
            final_risk_score = max(0, min(100, final_risk_score))
            
            # Step 7: Calculate confidence
            confidence = self._calculate_confidence(articles, sentiment_results)
            
            # Step 8: Generate explanation
            explanation = self._generate_explanation(
                company, articles, sentiment_results, risk_analysis_results, 
                final_risk_score, temporal_trend
            )
            
            computation_time = time.time() - start_time
            
            print(f"✅ Analysis complete for {company}: Risk Score = {final_risk_score:.1f}")
            
            # Return ExpertAssessment compatible format
            return {
                'expert_name': 'News Sentiment Expert',
                'risk_score': final_risk_score,
                'confidence': confidence,
                'explanation': explanation,
                'features_used': [
                    'finbert_sentiment', 'risk_keywords', 'temporal_trends', 
                    'news_volume', 'content_analysis'
                ],
                'computation_time': computation_time,
                
                # Detailed analysis for debugging/validation
                'detailed_analysis': {
                    'articles_analyzed': len(articles),
                    'base_sentiment_score': base_sentiment_score,
                    'avg_risk_score': avg_risk_score,
                    'temporal_trend': temporal_trend,
                    'risk_keywords': list(Counter(all_risk_keywords).most_common(10)),
                    'sentiment_distribution': {
                        'positive': sum(1 for r in sentiment_results if r['net_sentiment'] > 0.1),
                        'neutral': sum(1 for r in sentiment_results if -0.1 <= r['net_sentiment'] <= 0.1),
                        'negative': sum(1 for r in sentiment_results if r['net_sentiment'] < -0.1)
                    }
                },
                
                'sample_headlines': [article['title'] for article in articles[:5]]
            }
            
        except Exception as e:
            print(f"❌ Error in comprehensive news analysis: {e}")
            return self._create_error_assessment(company, str(e), start_time)
    
    def _calculate_confidence(self, articles: List[Dict], sentiment_results: List[Dict]) -> float:
        """Calculate confidence in the analysis based on data quality"""
        
        # Base confidence from article count
        article_count = len(articles)
        if article_count >= 15:
            count_confidence = 0.9
        elif article_count >= 10:
            count_confidence = 0.8
        elif article_count >= 5:
            count_confidence = 0.7
        else:
            count_confidence = 0.5
        
        # Confidence from sentiment prediction confidence
        if sentiment_results:
            avg_sentiment_confidence = np.mean([r['confidence'] for r in sentiment_results])
            sentiment_confidence = avg_sentiment_confidence
        else:
            sentiment_confidence = 0.5
        
        # Combined confidence
        overall_confidence = (count_confidence * 0.6 + sentiment_confidence * 0.4)
        
        return min(1.0, max(0.3, overall_confidence))
    
    def _generate_explanation(self, company: str, articles: List[Dict], 
                            sentiment_results: List[Dict], risk_results: List[Dict],
                            final_score: float, temporal_trend: float) -> str:
        """Generate comprehensive explanation of the news analysis"""
        
        explanation = f"News Sentiment Analysis for {company}:\n\n"
        
        # Risk level assessment
        if final_score < 30:
            risk_level = "LOW"
            risk_desc = "predominantly positive news sentiment with minimal risk indicators"
        elif final_score < 50:
            risk_level = "MODERATE-LOW"
            risk_desc = "mixed sentiment with some risk factors present"
        elif final_score < 70:
            risk_level = "MODERATE-HIGH" 
            risk_desc = "concerning sentiment patterns with notable risk factors"
        else:
            risk_level = "HIGH"
            risk_desc = "predominantly negative sentiment with significant risk indicators"
        
        explanation += f"Risk Assessment: {risk_level} ({final_score:.1f}/100)\n"
        explanation += f"Based on analysis of {len(articles)} recent news articles showing {risk_desc}.\n\n"
        
        # Sentiment breakdown
        if sentiment_results:
            positive_count = sum(1 for r in sentiment_results if r['net_sentiment'] > 0.1)
            negative_count = sum(1 for r in sentiment_results if r['net_sentiment'] < -0.1)
            neutral_count = len(sentiment_results) - positive_count - negative_count
            
            explanation += f"Sentiment Distribution: {positive_count} positive, {neutral_count} neutral, {negative_count} negative articles.\n"
        
        # Temporal trend
        if temporal_trend > 2:
            explanation += "Sentiment trend: Improving over time (reduced risk).\n"
        elif temporal_trend < -2:
            explanation += "Sentiment trend: Deteriorating over time (increased risk).\n"
        else:
            explanation += "Sentiment trend: Stable over analyzed period.\n"
        
        # Risk factors
        all_keywords = []
        for result in risk_results:
            all_keywords.extend(result['keywords_found'])
        
        if all_keywords:
            top_risks = Counter(all_keywords).most_common(5)
            explanation += f"\nKey Risk Indicators: {', '.join([risk for risk, _ in top_risks])}\n"
        
        # Sample headlines
        if articles:
            explanation += f"\nSample Headlines:\n"
            for i, article in enumerate(articles[:3]):
                explanation += f"• {article['title']}\n"
        
        return explanation
    
    def _create_no_news_assessment(self, company: str, start_time: float) -> Dict[str, Any]:
        """Create assessment when no news articles are found"""
        return {
            'expert_name': 'News Sentiment Expert',
            'risk_score': 50.0,  # Neutral risk
            'confidence': 0.2,   # Low confidence due to no data
            'explanation': f"No recent news articles found for {company}. Using neutral risk assessment.",
            'features_used': ['news_availability_check'],
            'computation_time': time.time() - start_time,
            'detailed_analysis': {
                'articles_analyzed': 0,
                'base_sentiment_score': 50.0,
                'avg_risk_score': 0.0,
                'temporal_trend': 0.0,
                'risk_keywords': [],
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0}
            },
            'sample_headlines': []
        }
    
    def _create_error_assessment(self, company: str, error_msg: str, start_time: float) -> Dict[str, Any]:
        """Create assessment when analysis fails"""
        return {
            'expert_name': 'News Sentiment Expert',
            'risk_score': 50.0,  # Neutral on error
            'confidence': 0.1,   # Very low confidence
            'explanation': f"Error analyzing news for {company}: {error_msg}. Using neutral assessment.",
            'features_used': ['error_handling'],
            'computation_time': time.time() - start_time,
            'detailed_analysis': {
                'error': error_msg,
                'articles_analyzed': 0,
                'base_sentiment_score': 50.0,
                'avg_risk_score': 0.0,
                'temporal_trend': 0.0,
                'risk_keywords': [],
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0}
            },
            'sample_headlines': []
        }


# === Main Interface Functions ===

def fetch_news_sentiment(company: str, date: datetime.date = datetime.date.today()) -> float:
    """
    Original function interface maintained for backward compatibility
    Enhanced with better error handling and analysis
    """
    analyzer = EnhancedNewsAnalyzer()
    result = analyzer.compute_comprehensive_news_score(company, days_back=5, max_articles=10)
    
    # Convert risk score back to sentiment score (0-100 where higher = better sentiment)
    sentiment_score = 100 - result['risk_score']
    return sentiment_score


def get_news_risk_assessment(company: str, days_back: int = 7, max_articles: int = 20) -> Dict[str, Any]:
    """
    Main function for MAESTRO integration - returns comprehensive news risk assessment
    """
    analyzer = EnhancedNewsAnalyzer()
    return analyzer.compute_comprehensive_news_score(company, days_back, max_articles)


# === Example Usage ===
if __name__ == "__main__":
    print("🚀 Testing Enhanced News Sentiment Analysis System")
    print("=" * 70)
    
    # Test companies
    test_companies = ["Apple", "Tesla", "Microsoft", "GameStop"]
    
    for company in test_companies:
        print(f"\n📊 Testing {company}...")
        
        # Test original function
        sentiment_score = fetch_news_sentiment(company)
        print(f"Original Function - Sentiment Score: {sentiment_score:.1f}/100")
        
        # Test comprehensive analysis
        comprehensive_result = get_news_risk_assessment(company, days_back=7, max_articles=15)
        print(f"Enhanced Analysis - Risk Score: {comprehensive_result['risk_score']:.1f}/100")
        print(f"Confidence: {comprehensive_result['confidence']:.1%}")
        print(f"Articles Analyzed: {comprehensive_result['detailed_analysis']['articles_analyzed']}")
        
        if comprehensive_result['sample_headlines']:
            print("Sample Headlines:")
            for headline in comprehensive_result['sample_headlines'][:2]:
                print(f"  • {headline}")
        
        print("-" * 70)
    
    print("\n✅ Testing completed!")
