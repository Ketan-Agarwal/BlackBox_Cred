"""
Explainability service for generating comprehensive credit score explanations.
"""
from datetime import datetime
from typing import Dict, List, Any
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db_session
from app.db.models import Company
from app.services.structured_service import StructuredModelService
from app.services.unstructured_service import UnstructuredModelService
from app.services.fusion_service import FusionService
from app.schemas.explanation import (
    ComprehensiveExplanation, PlainLanguageSummary, StructuredModelExplanation,
    UnstructuredModelExplanation, FusionExplanation, TrendAnalysis, FeatureContribution
)


class ExplainabilityService:
    """Service for generating detailed, explainable credit score analysis."""
    
    def __init__(self):
        self.structured_service = StructuredModelService()
        self.unstructured_service = UnstructuredModelService()
        self.fusion_service = FusionService()
    
    def _generate_plain_language_summary(self, company_name: str, final_score: float, 
                                       credit_grade: str, structured_score: float,
                                       unstructured_score: float, market_condition: str,
                                       trend_info: Dict) -> PlainLanguageSummary:
        """
        Generate plain language explanation of the credit assessment.
        
        Args:
            company_name: Name of the company
            final_score: Final credit score
            credit_grade: Credit grade
            structured_score: Structured model score
            unstructured_score: Unstructured model score
            market_condition: Current market condition
            trend_info: Trend analysis information
            
        Returns:
            PlainLanguageSummary object
        """
        # Overall assessment
        if final_score >= 80:
            assessment = f"{company_name} demonstrates strong creditworthiness with a {credit_grade} rating."
        elif final_score >= 60:
            assessment = f"{company_name} shows moderate credit quality with a {credit_grade} rating."
        elif final_score >= 40:
            assessment = f"{company_name} presents elevated credit risk with a {credit_grade} rating."
        else:
            assessment = f"{company_name} indicates high credit risk with a {credit_grade} rating."
        
        # Key strengths
        strengths = []
        if structured_score > 70:
            strengths.append("Strong fundamental financial metrics")
        if unstructured_score > 70:
            strengths.append("Positive market sentiment and news coverage")
        if trend_info.get('trends', {}).get('7d', {}).get('direction') == 'improving':
            strengths.append("Recent improvement in credit profile")
        if trend_info.get('trends', {}).get('90d', {}).get('direction') == 'improving':
            strengths.append("Long-term positive credit trend")
        
        if not strengths:
            strengths = ["Limited positive factors identified"]
        
        # Key concerns
        concerns = []
        if structured_score < 50:
            concerns.append("Weak fundamental financial performance")
        if unstructured_score < 50:
            concerns.append("Negative market sentiment and news coverage")
        if trend_info.get('trends', {}).get('7d', {}).get('direction') == 'declining':
            concerns.append("Recent deterioration in credit profile")
        if trend_info.get('trends', {}).get('90d', {}).get('direction') == 'declining':
            concerns.append("Long-term negative credit trend")
        
        if not concerns:
            concerns = ["No significant concerns identified"]
        
        # Market impact explanation
        if "High Volatility" in market_condition:
            market_impact = "Current market volatility is increasing the influence of news and sentiment on the credit assessment."
        elif "Low Volatility" in market_condition:
            market_impact = "Stable market conditions are emphasizing fundamental financial metrics in the credit assessment."
        else:
            market_impact = "Balanced market conditions are providing equal weight to fundamentals and market sentiment."
        
        # Recommendation
        if final_score >= 70:
            recommendation = "Favorable for investment and lending with standard terms."
        elif final_score >= 50:
            recommendation = "Acceptable for investment and lending with enhanced monitoring."
        else:
            recommendation = "Requires careful evaluation and risk mitigation for any lending or investment."
        
        return PlainLanguageSummary(
            overall_assessment=assessment,
            key_strengths=strengths,
            key_concerns=concerns,
            market_impact=market_impact,
            recommendation=recommendation
        )
    
    def _create_structured_explanation(self, structured_result: Dict) -> StructuredModelExplanation:
        """
        Create detailed explanation of structured model components.
        
        Args:
            structured_result: Results from structured model service
            
        Returns:
            StructuredModelExplanation object
        """
        # Z-Score interpretation
        z_score = structured_result['altman_z_score']
        if z_score > 2.6:
            z_interpretation = "Low bankruptcy risk"
        elif z_score > 1.1:
            z_interpretation = "Moderate bankruptcy risk"
        else:
            z_interpretation = "High bankruptcy risk"
        
        # Create feature contributions from SHAP values or EBM explanations
        feature_contributions = []
        model_type = structured_result.get('model_type', 'RandomForest')
        
        if model_type == 'EBM' and structured_result.get('feature_contributions'):
            # EBM explanations
            ebm_contributions = structured_result['feature_contributions']
            feature_names = structured_result['feature_names']
            feature_values = structured_result['feature_values']
            
            # Create list of contributions and sort by absolute value
            contributions = [
                {
                    'feature_name': name,
                    'value': value,
                    'contribution': contrib
                }
                for name, value, contrib in zip(feature_names, feature_values, ebm_contributions)
            ]
            
            # Sort by absolute contribution and take top 5
            contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
            
            for i, contrib in enumerate(contributions[:5]):
                # Convert feature name to human-readable format
                readable_name = self._make_feature_readable(contrib['feature_name'])
                
                feature_contributions.append(
                    FeatureContribution(
                        feature_name=readable_name,
                        value=contrib['value'],
                        contribution=contrib['contribution'],
                        importance_rank=i + 1
                    )
                )
                
        elif structured_result['shap_values'] and structured_result['feature_names']:
            # SHAP values (legacy Random Forest)
            shap_values = structured_result['shap_values']
            feature_names = structured_result['feature_names']
            feature_values = structured_result['feature_values']
            
            # Create list of contributions and sort by absolute value
            contributions = [
                {
                    'feature_name': name,
                    'value': value,
                    'contribution': shap_val
                }
                for name, value, shap_val in zip(feature_names, feature_values, shap_values)
            ]
            
            # Sort by absolute contribution and take top 5
            contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
            
            for i, contrib in enumerate(contributions[:5]):
                readable_name = self._make_feature_readable(contrib['feature_name'])
                
                feature_contributions.append(
                    FeatureContribution(
                        feature_name=readable_name,
                        value=contrib['value'],
                        contribution=contrib['contribution'],
                        importance_rank=i + 1
                    )
                )
        
        # KMV inputs
        company_data = structured_result['company_data']
        kmv_inputs = {
            'market_cap': company_data.get('market_cap', 0),
            'total_liabilities': company_data.get('total_liabilities', 0),
            'volatility': company_data.get('volatility', 0),
            'asset_value': company_data.get('market_cap', 0) + company_data.get('total_liabilities', 0)
        }
        
        # Z-Score inputs (financial ratios)
        z_score_inputs = {
            'current_ratio': company_data.get('current_ratio', 0),
            'debt_to_equity': company_data.get('debt_to_equity', 0),
            'return_on_assets': company_data.get('return_on_assets', 0),
            'operating_margin': company_data.get('operating_margin', 0),
            'asset_turnover': company_data.get('asset_turnover', 0)
        }
        
        return StructuredModelExplanation(
            random_forest_score=structured_result['structured_score'],
            kmv_distance_to_default=structured_result['kmv_distance_to_default'],
            kmv_inputs=kmv_inputs,
            altman_z_score=structured_result['altman_z_score'],
            z_score_inputs=z_score_inputs,
            z_score_interpretation=z_interpretation,
            top_feature_contributions=feature_contributions
        )

    def _make_feature_readable(self, feature_name: str) -> str:
        """
        Convert technical feature names to human-readable format.
        
        Args:
            feature_name: Technical feature name
            
        Returns:
            Human-readable feature name
        """
        feature_mappings = {
            'Current Ratio': 'Current Ratio',
            'Debt/Equity Ratio': 'Debt-to-Equity Ratio',
            'Gross Margin': 'Gross Profit Margin',
            'Operating Margin': 'Operating Profit Margin',
            'EBIT Margin': 'EBIT Margin',
            'EBITDA Margin': 'EBITDA Margin',
            'Net Profit Margin': 'Net Profit Margin',
            'Asset Turnover': 'Asset Turnover Ratio',
            'ROE - Return On Equity': 'Return on Equity',
            'ROA - Return On Assets': 'Return on Assets',
            'altman_z_score': 'Altman Z-Score',
            'kmv_distance_to_default': 'KMV Distance-to-Default',
            'current_ratio': 'Current Ratio',
            'quick_ratio': 'Quick Ratio',
            'debt_to_equity': 'Debt-to-Equity Ratio',
            'return_on_equity': 'Return on Equity',
            'return_on_assets': 'Return on Assets',
            'operating_margin': 'Operating Margin',
            'net_margin': 'Net Profit Margin',
            'asset_turnover': 'Asset Turnover',
            'inventory_turnover': 'Inventory Turnover'
        }
        
        return feature_mappings.get(feature_name, feature_name)
    
    def _create_unstructured_explanation(self, unstructured_result: Dict) -> UnstructuredModelExplanation:
        """
        Create detailed explanation of unstructured model components.
        
        Args:
            unstructured_result: Results from unstructured model service
            
        Returns:
            UnstructuredModelExplanation object
        """
        latest_sentiment = unstructured_result.get('latest_sentiment', {})
        
        return UnstructuredModelExplanation(
            finbert_score=unstructured_result['unstructured_score'],
            latest_news_headline=unstructured_result.get('latest_headline', 'No recent news available'),
            sentiment_classification=latest_sentiment.get('predicted_class', 'neutral'),
            sentiment_confidence=latest_sentiment.get('confidence', 0.0),
            raw_sentiment_probabilities=latest_sentiment.get('probabilities', {}),
            news_articles_analyzed=unstructured_result['article_count'],
            news_date_range={
                'start_date': unstructured_result['date_range']['start_date'],
                'end_date': unstructured_result['date_range']['end_date']
            }
        )
    
    def _create_fusion_explanation(self, fusion_result: Dict) -> FusionExplanation:
        """
        Create detailed explanation of fusion process.
        
        Args:
            fusion_result: Results from fusion service
            
        Returns:
            FusionExplanation object
        """
        return FusionExplanation(
            final_score=fusion_result['final_score'],
            credit_grade=fusion_result['credit_grade'],
            structured_weight=fusion_result['weights']['structured_weight'],
            unstructured_weight=fusion_result['weights']['unstructured_weight'],
            current_vix=fusion_result['market_context']['current_vix'],
            market_condition=fusion_result['market_context']['market_condition'],
            structured_component_score=fusion_result['component_scores']['structured_contribution'],
            unstructured_component_score=fusion_result['component_scores']['unstructured_contribution']
        )
    
    def _create_trend_explanation(self, trend_result: Dict) -> TrendAnalysis:
        """
        Create detailed trend analysis.
        
        Args:
            trend_result: Results from trend analysis
            
        Returns:
            TrendAnalysis object
        """
        trends = trend_result.get('trends', {})
        
        # Calculate volatility (simplified)
        changes = []
        if trends.get('7d', {}).get('change') is not None:
            changes.append(abs(trends['7d']['change']))
        if trends.get('90d', {}).get('change') is not None:
            changes.append(abs(trends['90d']['change']) / 13)  # Normalize to weekly change
        
        volatility = sum(changes) / len(changes) if changes else 0.0
        
        # Stability assessment
        if volatility < 1:
            stability = "Highly stable credit profile"
        elif volatility < 3:
            stability = "Moderately stable credit profile"
        else:
            stability = "Volatile credit profile requiring monitoring"
        
        return TrendAnalysis(
            current_score=trend_result.get('current_score', 0.0),
            score_7d_ago=trends.get('7d', {}).get('previous_score'),
            change_7d=trends.get('7d', {}).get('change'),
            trend_7d=trends.get('7d', {}).get('direction', 'unknown'),
            score_90d_ago=trends.get('90d', {}).get('previous_score'),
            change_90d=trends.get('90d', {}).get('change'),
            trend_90d=trends.get('90d', {}).get('direction', 'unknown'),
            score_volatility=volatility,
            stability_assessment=stability
        )
    
    def generate_explanation(self, company_id: int) -> ComprehensiveExplanation:
        """
        Generate comprehensive explanation for a company's credit score.
        
        Args:
            company_id: Company ID
            
        Returns:
            ComprehensiveExplanation object with all details
        """
        db = get_db_session()
        
        try:
            # Get company information
            company = db.query(Company).filter(Company.id == company_id).first()
            if not company:
                raise ValueError(f"Company with ID {company_id} not found")
            
            # Get structured score and explanation
            structured_result = self.structured_service.get_structured_score(company_id)
            
            # Get unstructured score and explanation
            unstructured_result = self.unstructured_service.get_unstructured_score(company_id)
            
            # Calculate final fused score
            fusion_result = self.fusion_service.calculate_final_score(
                structured_score=structured_result['structured_score'],
                unstructured_score=unstructured_result['unstructured_score'],
                company_id=company_id
            )
            
            # Get trend analysis
            trend_result = self.fusion_service.get_score_trends(company_id)
            
            # Create component explanations
            structured_explanation = self._create_structured_explanation(structured_result)
            unstructured_explanation = self._create_unstructured_explanation(unstructured_result)
            fusion_explanation = self._create_fusion_explanation(fusion_result)
            trend_explanation = self._create_trend_explanation(trend_result)
            
            # Generate plain language summary
            plain_summary = self._generate_plain_language_summary(
                company_name=company.name,
                final_score=fusion_result['final_score'],
                credit_grade=fusion_result['credit_grade'],
                structured_score=structured_result['structured_score'],
                unstructured_score=unstructured_result['unstructured_score'],
                market_condition=fusion_result['market_context']['market_condition'],
                trend_info=trend_result
            )
            
            # Raw data sources for transparency
            raw_data_sources = {
                'structured_data_date': structured_result.get('company_data', {}).get('data_date'),
                'news_articles_count': unstructured_result['article_count'],
                'vix_value': fusion_result['market_context']['current_vix'],
                'calculation_method': 'Dynamic Hybrid Expert Model',
                'model_versions': {
                    'random_forest': 'v1.0',
                    'finbert': settings.finbert_model_name,
                    'fusion_algorithm': 'VIX-weighted dynamic fusion'
                }
            }
            
            return ComprehensiveExplanation(
                company_symbol=company.symbol,
                company_name=company.name,
                calculation_timestamp=datetime.utcnow(),
                plain_language_summary=plain_summary,
                structured_model=structured_explanation,
                unstructured_model=unstructured_explanation,
                fusion_process=fusion_explanation,
                trend_analysis=trend_explanation,
                raw_data_sources=raw_data_sources
            )
            
        except Exception as e:
            print(f"Error generating explanation: {e}")
            raise
        finally:
            db.close()
