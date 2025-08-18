"""
Unstructured model service implementing FinBERT for sentiment analysis.
"""
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

from app.core.config import settings
from app.db.session import get_db_session
from app.db.models import Company, UnstructuredData


class UnstructuredModelService:
    """Service for unstructured text analysis using FinBERT."""
    
    def __init__(self):
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _load_finbert_model(self):
        """Load the FinBERT model and tokenizer."""
        if self.model is None or self.tokenizer is None:
            try:
                print("Loading FinBERT model...")
                self.tokenizer = AutoTokenizer.from_pretrained(settings.finbert_model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(settings.finbert_model_name)
                self.model.to(self.device)
                self.model.eval()
                print("FinBERT model loaded successfully")
            except Exception as e:
                print(f"Error loading FinBERT model: {e}")
                raise
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of text using FinBERT.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        self._load_finbert_model()
        
        try:
            # Tokenize input text
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT classes: [negative, neutral, positive]
            class_names = ['negative', 'neutral', 'positive']
            probabilities = predictions.cpu().numpy()[0]
            
            # Get the predicted class
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = class_names[predicted_class_idx]
            confidence = float(probabilities[predicted_class_idx])
            
            # Calculate sentiment score (-1 to 1 scale)
            # Formula: (P_positive - P_negative)
            sentiment_score = float(probabilities[2] - probabilities[0])
            
            return {
                'probabilities': {
                    'negative': float(probabilities[0]),
                    'neutral': float(probabilities[1]),
                    'positive': float(probabilities[2])
                },
                'predicted_class': predicted_class,
                'confidence': confidence,
                'sentiment_score': sentiment_score  # -1 to 1 scale
            }
            
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {
                'probabilities': {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33},
                'predicted_class': 'neutral',
                'confidence': 0.33,
                'sentiment_score': 0.0
            }
    
    def _process_news_batch(self, news_articles: List[str]) -> Dict:
        """
        Process a batch of news articles and calculate aggregate sentiment.
        
        Args:
            news_articles: List of news article texts
            
        Returns:
            Aggregated sentiment analysis
        """
        if not news_articles:
            return {
                'aggregate_sentiment_score': 0.0,
                'processed_score': 50.0,  # Neutral score
                'article_count': 0,
                'sentiment_distribution': {'negative': 0, 'neutral': 0, 'positive': 0}
            }
        
        sentiment_scores = []
        sentiment_counts = {'negative': 0, 'neutral': 0, 'positive': 0}
        
        for article in news_articles:
            if article and len(article.strip()) > 10:  # Skip very short articles
                result = self._analyze_sentiment(article)
                sentiment_scores.append(result['sentiment_score'])
                sentiment_counts[result['predicted_class']] += 1
        
        if not sentiment_scores:
            return {
                'aggregate_sentiment_score': 0.0,
                'processed_score': 50.0,
                'article_count': 0,
                'sentiment_distribution': sentiment_counts
            }
        
        # Calculate aggregate sentiment (weighted average)
        aggregate_sentiment = np.mean(sentiment_scores)
        
        # Convert to 0-100 scale: ((sentiment + 1) / 2) * 100
        processed_score = ((aggregate_sentiment + 1) / 2) * 100
        
        return {
            'aggregate_sentiment_score': float(aggregate_sentiment),
            'processed_score': float(processed_score),
            'article_count': len(sentiment_scores),
            'sentiment_distribution': sentiment_counts
        }
    
    def get_unstructured_score(self, company_id: int, days_back: int = 7) -> Dict:
        """
        Calculate unstructured credit score for a company based on recent news.
        
        Args:
            company_id: Company ID
            days_back: Number of days back to analyze news
            
        Returns:
            Dictionary containing unstructured score and analysis
        """
        db = get_db_session()
        
        try:
            # Get company information
            company = db.query(Company).filter(Company.id == company_id).first()
            if not company:
                raise ValueError(f"Company with ID {company_id} not found")
            
            # Get recent news articles
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            news_articles = (
                db.query(UnstructuredData)
                .filter(
                    UnstructuredData.company_id == company_id,
                    UnstructuredData.published_at >= cutoff_date
                )
                .order_by(UnstructuredData.published_at.desc())
                .limit(50)  # Limit to 50 most recent articles
                .all()
            )
            
            if not news_articles:
                # No recent news, return neutral score
                return {
                    'unstructured_score': 50.0,
                    'article_count': 0,
                    'latest_headline': None,
                    'sentiment_analysis': {
                        'aggregate_sentiment_score': 0.0,
                        'sentiment_distribution': {'negative': 0, 'neutral': 0, 'positive': 0}
                    },
                    'date_range': {
                        'start_date': cutoff_date,
                        'end_date': datetime.utcnow()
                    }
                }
            
            # Process news articles
            headlines = [article.headline for article in news_articles if article.headline]
            
            # For demonstration, we'll analyze just headlines (in production, you'd analyze full content)
            sentiment_analysis = self._process_news_batch(headlines)
            
            # Update database with processed results
            for i, article in enumerate(news_articles):
                if i < len(headlines) and headlines[i]:  # Only process articles we analyzed
                    individual_result = self._analyze_sentiment(headlines[i])
                    article.sentiment_score = individual_result['sentiment_score']
                    article.sentiment_label = individual_result['predicted_class']
                    article.finbert_confidence = individual_result['confidence']
                    article.processed_score = ((individual_result['sentiment_score'] + 1) / 2) * 100
            
            db.commit()
            
            # Get the latest headline for explanation
            latest_headline = news_articles[0].headline if news_articles else None
            latest_sentiment = None
            if latest_headline:
                latest_sentiment = self._analyze_sentiment(latest_headline)
            
            return {
                'unstructured_score': sentiment_analysis['processed_score'],
                'article_count': sentiment_analysis['article_count'],
                'latest_headline': latest_headline,
                'latest_sentiment': latest_sentiment,
                'sentiment_analysis': sentiment_analysis,
                'date_range': {
                    'start_date': cutoff_date,
                    'end_date': datetime.utcnow()
                },
                'raw_articles': [
                    {
                        'headline': article.headline,
                        'published_at': article.published_at,
                        'source': article.source,
                        'sentiment_score': article.sentiment_score,
                        'sentiment_label': article.sentiment_label
                    }
                    for article in news_articles[:5]  # Return top 5 for reference
                ]
            }
            
        except Exception as e:
            print(f"Error calculating unstructured score: {e}")
            raise
        finally:
            db.close()
    
    def analyze_single_text(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text (for testing or real-time analysis).
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment analysis results
        """
        result = self._analyze_sentiment(text)
        processed_score = ((result['sentiment_score'] + 1) / 2) * 100
        
        return {
            'text': text,
            'sentiment_score': result['sentiment_score'],
            'processed_score': processed_score,
            'predicted_class': result['predicted_class'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities']
        }
