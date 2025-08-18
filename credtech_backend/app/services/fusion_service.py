"""
Dynamic fusion service for combining structured and unstructured scores.
"""
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Tuple
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db_session
from app.db.models import Company, CreditScoreHistory


class FusionService:
    """Service for dynamic weighted fusion of structured and unstructured scores."""
    
    def __init__(self):
        self.grade_thresholds = {
            'AAA': 95,
            'AA+': 90,
            'AA': 85,
            'AA-': 80,
            'A+': 75,
            'A': 70,
            'A-': 65,
            'BBB+': 60,
            'BBB': 55,
            'BBB-': 50,
            'BB+': 45,
            'BB': 40,
            'BB-': 35,
            'B+': 30,
            'B': 25,
            'B-': 20,
            'CCC': 15,
            'CC': 10,
            'C': 5,
            'D': 0
        }
    
    def _get_market_weight(self) -> Dict:
        """
        Calculate dynamic market weight based on VIX volatility index.
        
        Returns:
            Dictionary containing market weight and related information
        """
        try:
            # Fetch VIX data
            vix_ticker = yf.Ticker(settings.vix_symbol)
            vix_data = vix_ticker.history(period="5d")
            
            if vix_data.empty:
                # Fallback if VIX data is unavailable
                current_vix = 20.0  # Assume moderate volatility
                print("Warning: Could not fetch VIX data, using default value")
            else:
                current_vix = float(vix_data['Close'].iloc[-1])
            
            # Dynamic weighting logic based on VIX levels
            if current_vix <= 15:
                # Low volatility - market is calm, focus more on fundamentals
                news_weight = 0.20
                market_condition = "Low Volatility - Stable Market"
            elif current_vix <= 25:
                # Normal volatility - balanced approach
                news_weight = 0.35
                market_condition = "Normal Volatility - Balanced Market"
            elif current_vix <= 35:
                # High volatility - news and sentiment become more important
                news_weight = 0.50
                market_condition = "High Volatility - Uncertain Market"
            else:
                # Very high volatility - market driven by sentiment and news
                news_weight = 0.65
                market_condition = "Very High Volatility - Crisis/Panic Mode"
            
            structured_weight = 1.0 - news_weight
            
            return {
                'current_vix': current_vix,
                'news_weight': news_weight,
                'structured_weight': structured_weight,
                'market_condition': market_condition,
                'vix_interpretation': self._interpret_vix(current_vix)
            }
            
        except Exception as e:
            print(f"Error fetching market data: {e}")
            # Fallback to balanced weights
            return {
                'current_vix': 20.0,
                'news_weight': 0.35,
                'structured_weight': 0.65,
                'market_condition': "Unknown - Using Default Weights",
                'vix_interpretation': "Unable to determine market sentiment"
            }
    
    def _interpret_vix(self, vix_value: float) -> str:
        """
        Interpret VIX value in plain language.
        
        Args:
            vix_value: Current VIX value
            
        Returns:
            Human-readable interpretation
        """
        if vix_value <= 12:
            return "Market complacency - very low fear"
        elif vix_value <= 20:
            return "Normal market conditions - low to moderate fear"
        elif vix_value <= 30:
            return "Elevated uncertainty - moderate to high fear"
        elif vix_value <= 40:
            return "High stress - significant fear and uncertainty"
        else:
            return "Extreme fear - potential market panic"
    
    def _score_to_grade(self, score: float) -> str:
        """
        Convert numerical score to credit grade.
        
        Args:
            score: Numerical score (0-100)
            
        Returns:
            Credit grade string
        """
        for grade, threshold in self.grade_thresholds.items():
            if score >= threshold:
                return grade
        return 'D'  # Default to lowest grade
    
    def calculate_final_score(self, structured_score: float, unstructured_score: float, 
                            company_id: int) -> Dict:
        """
        Calculate final fused credit score using dynamic weighting.
        
        Args:
            structured_score: Score from structured model (0-100)
            unstructured_score: Score from unstructured model (0-100)
            company_id: Company ID for record keeping
            
        Returns:
            Dictionary containing final score and detailed breakdown
        """
        try:
            # Get dynamic market weights
            market_info = self._get_market_weight()
            
            # Calculate weighted final score
            final_score = (
                structured_score * market_info['structured_weight'] + 
                unstructured_score * market_info['news_weight']
            )
            
            # Convert to credit grade
            credit_grade = self._score_to_grade(final_score)
            
            # Store in database for historical tracking
            self._store_score_history(
                company_id=company_id,
                structured_score=structured_score,
                unstructured_score=unstructured_score,
                final_score=final_score,
                credit_grade=credit_grade,
                news_weight=market_info['news_weight'],
                vix_value=market_info['current_vix']
            )
            
            return {
                'final_score': round(final_score, 2),
                'credit_grade': credit_grade,
                'component_scores': {
                    'structured_score': structured_score,
                    'unstructured_score': unstructured_score,
                    'structured_contribution': structured_score * market_info['structured_weight'],
                    'unstructured_contribution': unstructured_score * market_info['news_weight']
                },
                'weights': {
                    'structured_weight': market_info['structured_weight'],
                    'unstructured_weight': market_info['news_weight']
                },
                'market_context': {
                    'current_vix': market_info['current_vix'],
                    'market_condition': market_info['market_condition'],
                    'vix_interpretation': market_info['vix_interpretation']
                },
                'calculation_timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            print(f"Error calculating final score: {e}")
            raise
    
    def _store_score_history(self, company_id: int, structured_score: float, 
                           unstructured_score: float, final_score: float, 
                           credit_grade: str, news_weight: float, vix_value: float):
        """
        Store score calculation in database for historical tracking.
        
        Args:
            company_id: Company ID
            structured_score: Structured model score
            unstructured_score: Unstructured model score
            final_score: Final fused score
            credit_grade: Credit grade
            news_weight: Weight given to news component
            vix_value: VIX value at calculation time
        """
        db = get_db_session()
        
        try:
            score_record = CreditScoreHistory(
                company_id=company_id,
                structured_score=structured_score,
                unstructured_score=unstructured_score,
                final_score=final_score,
                credit_grade=credit_grade,
                news_weight=news_weight,
                market_volatility_vix=vix_value,
                calculation_timestamp=datetime.utcnow()
            )
            
            db.add(score_record)
            db.commit()
            
        except Exception as e:
            print(f"Error storing score history: {e}")
            db.rollback()
        finally:
            db.close()
    
    def get_score_trends(self, company_id: int) -> Dict:
        """
        Calculate score trends for trend analysis.
        
        Args:
            company_id: Company ID
            
        Returns:
            Dictionary containing trend information
        """
        db = get_db_session()
        
        try:
            # Get current score
            current_score = (
                db.query(CreditScoreHistory)
                .filter(CreditScoreHistory.company_id == company_id)
                .order_by(CreditScoreHistory.calculation_timestamp.desc())
                .first()
            )
            
            if not current_score:
                return {
                    'current_score': None,
                    'trends': {
                        '7d': {'change': None, 'direction': 'unknown'},
                        '90d': {'change': None, 'direction': 'unknown'}
                    }
                }
            
            # Get score from 7 days ago
            week_ago = datetime.utcnow() - timedelta(days=7)
            score_7d = (
                db.query(CreditScoreHistory)
                .filter(
                    CreditScoreHistory.company_id == company_id,
                    CreditScoreHistory.calculation_timestamp <= week_ago
                )
                .order_by(CreditScoreHistory.calculation_timestamp.desc())
                .first()
            )
            
            # Get score from 90 days ago
            quarter_ago = datetime.utcnow() - timedelta(days=90)
            score_90d = (
                db.query(CreditScoreHistory)
                .filter(
                    CreditScoreHistory.company_id == company_id,
                    CreditScoreHistory.calculation_timestamp <= quarter_ago
                )
                .order_by(CreditScoreHistory.calculation_timestamp.desc())
                .first()
            )
            
            # Calculate trends
            trends = {}
            
            # 7-day trend
            if score_7d:
                change_7d = current_score.final_score - score_7d.final_score
                trends['7d'] = {
                    'change': round(change_7d, 2),
                    'direction': 'improving' if change_7d > 1 else 'declining' if change_7d < -1 else 'stable',
                    'previous_score': score_7d.final_score,
                    'previous_grade': score_7d.credit_grade
                }
            else:
                trends['7d'] = {'change': None, 'direction': 'insufficient_data'}
            
            # 90-day trend
            if score_90d:
                change_90d = current_score.final_score - score_90d.final_score
                trends['90d'] = {
                    'change': round(change_90d, 2),
                    'direction': 'improving' if change_90d > 2 else 'declining' if change_90d < -2 else 'stable',
                    'previous_score': score_90d.final_score,
                    'previous_grade': score_90d.credit_grade
                }
            else:
                trends['90d'] = {'change': None, 'direction': 'insufficient_data'}
            
            return {
                'current_score': current_score.final_score,
                'current_grade': current_score.credit_grade,
                'trends': trends,
                'calculation_timestamp': current_score.calculation_timestamp
            }
            
        except Exception as e:
            print(f"Error calculating score trends: {e}")
            return {
                'current_score': None,
                'trends': {
                    '7d': {'change': None, 'direction': 'error'},
                    '90d': {'change': None, 'direction': 'error'}
                }
            }
        finally:
            db.close()
