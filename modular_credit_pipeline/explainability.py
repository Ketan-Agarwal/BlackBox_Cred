
"""
explainability.py

Comprehensive explainability module combining structured and unstructured explanations.
Self-contained implementation without external dependencies.
"""

import logging
import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Any, List
from collections import Counter
import datetime

logger = logging.getLogger(__name__)

# === EBM EXPLAINER (copied from ebm_exp.py) ===

class EBMExplainer:
    """Comprehensive explainability class for EBM credit scoring model."""
    
    def __init__(self, model_path):
        """Initialize the explainer with a trained model."""
        self.model_path = model_path
        self.model_data = None
        self.ebm_model = None
        self.scaler = None
        self.feature_columns = None
        self.load_model()
        
    def load_model(self):
        """Load the trained EBM model and associated components."""
        logger.info(f"Loading EBM model from {self.model_path}...")
        try:
            with open(self.model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            
            self.ebm_model = self.model_data['model']
            self.scaler = self.model_data['scaler']
            self.feature_columns = self.model_data['feature_columns']
            
            logger.info("EBM model loaded successfully.")
            logger.info(f"Model accuracy: {self.model_data.get('accuracy', 'N/A')}")
            logger.info(f"Features: {len(self.feature_columns)}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_feature_interpretation(self, feature_name, value, contribution):
        """Get business interpretation for a specific feature value."""
        
        interpretations = {
            'debt_to_equity': {
                'thresholds': [0.5, 1.0, 2.0, 3.0],
                'descriptions': [
                    "excellent capital structure with very low leverage",
                    "good capital structure with moderate leverage", 
                    "concerning leverage levels increasing risk",
                    "high leverage indicating significant financial risk",
                    "extremely high leverage suggesting potential distress"
                ]
            },
            'current_ratio': {
                'thresholds': [1.0, 1.5, 2.0, 3.0],
                'descriptions': [
                    "insufficient liquidity to meet short-term obligations",
                    "adequate liquidity but below optimal levels",
                    "good liquidity providing reasonable safety buffer",
                    "strong liquidity position with excellent coverage",
                    "very strong liquidity position"
                ]
            },
            'enhanced_z_score': {
                'thresholds': [1.8, 3.0, 4.5, 6.0],
                'descriptions': [
                    "high bankruptcy risk requiring immediate attention",
                    "moderate bankruptcy risk needing close monitoring",
                    "low bankruptcy risk indicating stable operations",
                    "very low bankruptcy risk with strong fundamentals",
                    "minimal bankruptcy risk with excellent financial health"
                ]
            }
        }
        
        if feature_name in interpretations:
            thresholds = interpretations[feature_name]['thresholds']
            descriptions = interpretations[feature_name]['descriptions']
            
            description_idx = 0
            for i, threshold in enumerate(thresholds):
                if value <= threshold:
                    description_idx = i
                    break
            else:
                description_idx = len(descriptions) - 1
            
            return descriptions[description_idx]
        else:
            if contribution > 0:
                return "contributing to the predicted rating"
            else:
                return "opposing the predicted rating"
    
    def explain_single_prediction(self, sample_data, company_name="Unknown Company"):
        """Generate detailed explanation for a single prediction."""
        
        if isinstance(sample_data, dict):
            sample_df = pd.DataFrame([sample_data])
        elif isinstance(sample_data, pd.Series):
            sample_df = sample_data.to_frame().T
        else:
            sample_df = sample_data.copy()
        
        available_features = [col for col in self.feature_columns if col in sample_df.columns]
        sample_df = sample_df[available_features]
        
        sample_df = sample_df.fillna(0)
        sample_df = sample_df.replace([np.inf, -np.inf], 0)
        
        sample_scaled = self.scaler.transform(sample_df)
        
        try:
            prediction = self.ebm_model.predict(sample_scaled)[0]
            prediction_proba = self.ebm_model.predict_proba(sample_scaled)[0]
        except Exception as e:
            logger.error(f"Error getting prediction: {e}")
            prediction = 0
            prediction_proba = [0.5, 0.5]
        
        # Get feature importance (simplified)
        feature_names = self.feature_columns
        feature_values = sample_df.iloc[0].to_dict()
        
        # Simple feature importance based on global model importance
        try:
            global_explanation = self.ebm_model.explain_global()
            global_data = global_explanation.data()
            global_scores = global_data.get('scores', [1.0] * len(feature_names))
            
            feature_scores = []
            sample_values = sample_df.iloc[0]
            
            for i, feature_name in enumerate(feature_names):
                if i < len(global_scores):
                    feature_value = sample_values.get(feature_name, 0)
                    approx_score = (feature_value / (abs(feature_value) + 1)) * global_scores[i] * 0.1
                    feature_scores.append(approx_score)
                else:
                    feature_scores.append(0.0)
                    
        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")
            feature_scores = [0.001] * len(feature_names)
        
        explanation = self._generate_detailed_explanation(
            company_name, prediction, prediction_proba, 
            feature_names, feature_scores, feature_values
        )
        
        return {
            'explanation_text': explanation,
            'prediction': 'Investment Grade' if prediction == 1 else 'Non-Investment Grade',
            'probability_investment_grade': prediction_proba[1],
            'probability_non_investment_grade': prediction_proba[0],
            'feature_contributions': dict(zip(feature_names, feature_scores))
        }
    
    def _generate_detailed_explanation(self, company_name, prediction, prediction_proba, 
                                     feature_names, feature_scores, feature_values):
        """Generate the detailed textual explanation."""
        
        feature_data = list(zip(feature_names, feature_scores, 
                               [feature_values.get(name, 0) for name in feature_names]))
        feature_data.sort(key=lambda x: abs(x[1]), reverse=True)
        
        explanation_lines = []
        explanation_lines.append(f"CREDIT RISK ANALYSIS FOR {company_name.upper()}")
        explanation_lines.append("=" * 60)
        explanation_lines.append("")
        
        grade = "INVESTMENT GRADE" if prediction == 1 else "NON-INVESTMENT GRADE"
        
        explanation_lines.append(f"OVERALL RATING: {grade}")
        explanation_lines.append(f"Investment Grade Probability: {prediction_proba[1]:.1%}")
        explanation_lines.append(f"Non-Investment Grade Probability: {prediction_proba[0]:.1%}")
        explanation_lines.append("")
        
        explanation_lines.append("DETAILED FEATURE ANALYSIS:")
        explanation_lines.append("-" * 40)
        explanation_lines.append("")
        
        total_abs_contribution = sum(abs(score) for _, score, _ in feature_data)
        
        for i, (feature_name, contribution, value) in enumerate(feature_data[:10]):
            
            contrib_pct = (abs(contribution) / max(total_abs_contribution, 1e-8)) * 100
            
            if prediction == 1:
                impact = "INCREASES investment grade probability" if contribution > 0 else "DECREASES investment grade probability"
                impact_symbol = "+" if contribution > 0 else "-"
            else:
                impact = "INCREASES non-investment grade probability" if contribution > 0 else "DECREASES non-investment grade probability"
                impact_symbol = "+" if contribution > 0 else "-"
            
            interpretation = self.get_feature_interpretation(feature_name, value, contribution)
            
            explanation_lines.append(f"{i+1}. {feature_name.replace('_', ' ').title()}:")
            explanation_lines.append(f"   Value: {value:.4f}")
            explanation_lines.append(f"   Interpretation: {interpretation}")
            explanation_lines.append(f"   Impact: [{impact_symbol}] {impact} by {contrib_pct:.1f}%")
            explanation_lines.append(f"   Contribution Score: {contribution:+.4f}")
            explanation_lines.append("")
        
        explanation_lines.append(f"Analysis generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(explanation_lines)

# === NEWS EXPLAINABILITY (copied from news_explainability.py) ===

class NewsExplainabilityEngine:
    """Comprehensive explainability engine for news-based risk analysis."""
    
    def __init__(self):
        self.risk_categories = {
            'liquidity_crisis': {
                'name': 'Liquidity & Cash Flow Issues',
                'description': 'Problems with immediate cash availability and working capital',
                'impact': 'High immediate risk to operations and debt servicing',
                'keywords': ['cash flow', 'liquidity crisis', 'cash shortage', 'working capital', 
                           'credit facility', 'refinancing', 'cash burn', 'funding gap']
            },
            'debt_distress': {
                'name': 'Debt & Financial Distress',
                'description': 'Issues with debt obligations and financial health',
                'impact': 'Very high risk of default or restructuring',
                'keywords': ['debt default', 'covenant violation', 'bankruptcy', 'insolvency',
                           'debt restructuring', 'creditor pressure', 'leverage concerns', 'debt burden']
            }
        }
    
    def analyze_news_impact(self, news_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Main function to analyze and explain news impact."""
        
        if not news_assessment or 'detailed_analysis' not in news_assessment:
            return self._create_no_data_explanation()
        
        detailed = news_assessment['detailed_analysis']
        risk_score = news_assessment.get('risk_score', 50)
        
        sentiment_analysis = self._analyze_sentiment_impact(detailed)
        risk_factor_analysis = self._analyze_risk_factors(news_assessment)
        temporal_analysis = self._analyze_temporal_trends(detailed)
        confidence_analysis = self._analyze_confidence_factors(news_assessment)
        
        explanation = self._generate_comprehensive_explanation(
            news_assessment, sentiment_analysis, risk_factor_analysis, 
            temporal_analysis, confidence_analysis
        )
        
        return {
            'company': news_assessment.get('company', 'Unknown'),
            'risk_score': risk_score,
            'risk_level': self._categorize_risk_level(risk_score),
            'confidence': news_assessment.get('confidence', 0.5),
            'explanation': explanation,
            'sentiment_analysis': sentiment_analysis,
            'risk_factor_analysis': risk_factor_analysis,
            'temporal_analysis': temporal_analysis,
            'confidence_analysis': confidence_analysis,
            'actionable_insights': self._generate_actionable_insights(
                risk_score, sentiment_analysis, risk_factor_analysis
            ),
            'data_quality': self._assess_data_quality(news_assessment)
        }
    
    def _analyze_sentiment_impact(self, detailed_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the sentiment distribution and its impact on risk"""
        
        sentiment_dist = detailed_analysis.get('sentiment_distribution', {})
        total_articles = sum(sentiment_dist.values())
        
        if total_articles == 0:
            return {'overall_sentiment': 'unknown', 'impact': 'Cannot assess', 'distribution': {}}
        
        sentiment_percentages = {
            sentiment: (count / total_articles) * 100 
            for sentiment, count in sentiment_dist.items()
        }
        
        if sentiment_percentages.get('negative', 0) > 60:
            overall_sentiment = 'predominantly_negative'
            impact_desc = 'High negative impact on perceived risk'
        elif sentiment_percentages.get('positive', 0) > 60:
            overall_sentiment = 'predominantly_positive'
            impact_desc = 'Positive impact reducing perceived risk'
        else:
            overall_sentiment = 'mixed'
            impact_desc = 'Neutral impact with mixed signals'
        
        return {
            'overall_sentiment': overall_sentiment,
            'impact': impact_desc,
            'distribution': sentiment_percentages,
            'total_articles': total_articles,
            'dominant_sentiment': max(sentiment_dist.items(), key=lambda x: x[1])[0] if sentiment_dist else 'unknown'
        }
    
    def _analyze_risk_factors(self, news_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze specific risk factors identified in the news"""
        
        risk_keywords = news_assessment.get('detailed_analysis', {}).get('risk_keywords', [])
        
        if not risk_keywords:
            return {'categories_detected': {}, 'total_risk_signals': 0, 'primary_concerns': []}
        
        categorized_risks = {}
        for category, info in self.risk_categories.items():
            category_matches = []
            for keyword, count in risk_keywords:
                if keyword.lower() in [k.lower() for k in info['keywords']]:
                    category_matches.append((keyword, count))
            
            if category_matches:
                total_mentions = sum(count for _, count in category_matches)
                categorized_risks[category] = {
                    'name': info['name'],
                    'description': info['description'],
                    'impact': info['impact'],
                    'matches': category_matches,
                    'total_mentions': total_mentions
                }
        
        primary_concerns = sorted(
            categorized_risks.items(), 
            key=lambda x: x[1]['total_mentions'], 
            reverse=True
        )[:3]
        
        return {
            'categories_detected': categorized_risks,
            'total_risk_signals': len(risk_keywords),
            'primary_concerns': [
                {
                    'category': cat,
                    'name': info['name'],
                    'mentions': info['total_mentions'],
                    'impact': info['impact']
                }
                for cat, info in primary_concerns
            ]
        }
    
    def _analyze_temporal_trends(self, detailed_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal trends in sentiment"""
        
        temporal_trend = detailed_analysis.get('temporal_trend', 0)
        
        if temporal_trend > 5:
            trend_desc = 'Strongly improving'
            impact = 'Positive - risk perception decreasing over time'
        elif temporal_trend > 2:
            trend_desc = 'Improving'
            impact = 'Somewhat positive - slight improvement in sentiment'
        elif temporal_trend > -2:
            trend_desc = 'Stable'
            impact = 'Neutral - consistent sentiment pattern'
        elif temporal_trend > -5:
            trend_desc = 'Deteriorating'
            impact = 'Concerning - sentiment worsening over time'
        else:
            trend_desc = 'Sharply deteriorating'
            impact = 'High concern - rapidly worsening sentiment'
        
        return {
            'trend_direction': trend_desc,
            'trend_value': temporal_trend,
            'impact_assessment': impact,
            'significance': 'High' if abs(temporal_trend) > 3 else 'Moderate' if abs(temporal_trend) > 1 else 'Low'
        }
    
    def _analyze_confidence_factors(self, news_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze factors affecting confidence in the assessment"""
        
        detailed = news_assessment.get('detailed_analysis', {})
        articles_count = detailed.get('articles_analyzed', 0)
        confidence = news_assessment.get('confidence', 0.5)
        
        confidence_factors = []
        
        if articles_count >= 15:
            confidence_factors.append('Sufficient news coverage (15+ articles)')
        elif articles_count >= 8:
            confidence_factors.append('Adequate news coverage (8-14 articles)')
        elif articles_count >= 3:
            confidence_factors.append('Limited news coverage (3-7 articles)')
        else:
            confidence_factors.append('Very limited news coverage (<3 articles)')
        
        confidence_factors.append('Analysis covers recent 7-day period')
        
        if confidence > 0.8:
            confidence_factors.append('High model confidence in predictions')
        elif confidence > 0.6:
            confidence_factors.append('Moderate model confidence')
        else:
            confidence_factors.append('Lower model confidence - interpret with caution')
        
        return {
            'overall_confidence': confidence,
            'confidence_level': 'High' if confidence > 0.7 else 'Moderate' if confidence > 0.5 else 'Low',
            'contributing_factors': confidence_factors,
            'data_sufficiency': 'Sufficient' if articles_count >= 10 else 'Limited' if articles_count >= 5 else 'Insufficient'
        }
    
    def _generate_comprehensive_explanation(self, news_assessment: Dict[str, Any], 
                                          sentiment_analysis: Dict[str, Any],
                                          risk_factor_analysis: Dict[str, Any],
                                          temporal_analysis: Dict[str, Any],
                                          confidence_analysis: Dict[str, Any]) -> str:
        """Generate the main comprehensive explanation text"""
        
        risk_score = news_assessment.get('risk_score', 50)
        company = news_assessment.get('company', 'the company')
        
        explanation = []
        
        explanation.append(f"🔍 NEWS SENTIMENT RISK ANALYSIS")
        explanation.append(f"Risk Score: {risk_score:.1f}/100 ({self._categorize_risk_level(risk_score)})")
        explanation.append("")
        
        explanation.append("📊 EXECUTIVE SUMMARY:")
        if risk_score < 30:
            explanation.append(f"News sentiment indicates LOW RISK for {company}. Recent coverage is predominantly positive with minimal risk indicators.")
        elif risk_score < 50:
            explanation.append(f"News sentiment indicates MODERATE-LOW RISK for {company}. Mixed sentiment with some areas of concern.")
        elif risk_score < 70:
            explanation.append(f"News sentiment indicates MODERATE-HIGH RISK for {company}. Notable negative sentiment and risk factors present.")
        else:
            explanation.append(f"News sentiment indicates HIGH RISK for {company}. Predominantly negative coverage with significant risk indicators.")
        explanation.append("")
        
        explanation.append("📈 SENTIMENT BREAKDOWN:")
        dist = sentiment_analysis['distribution']
        explanation.append(f"• Positive articles: {dist.get('positive', 0):.1f}%")
        explanation.append(f"• Neutral articles: {dist.get('neutral', 0):.1f}%")
        explanation.append(f"• Negative articles: {dist.get('negative', 0):.1f}%")
        explanation.append(f"• Overall sentiment: {sentiment_analysis['overall_sentiment'].replace('_', ' ').title()}")
        explanation.append("")
        
        if risk_factor_analysis['primary_concerns']:
            explanation.append("⚠️ PRIMARY RISK FACTORS:")
            for concern in risk_factor_analysis['primary_concerns']:
                explanation.append(f"• {concern['name']}: {concern['mentions']} mentions")
                explanation.append(f"  Impact: {concern['impact']}")
            explanation.append("")
        
        explanation.append("📅 TEMPORAL ANALYSIS:")
        explanation.append(f"• Trend direction: {temporal_analysis['trend_direction']}")
        explanation.append(f"• Impact: {temporal_analysis['impact_assessment']}")
        explanation.append("")
        
        if 'sample_headlines' in news_assessment and news_assessment['sample_headlines']:
            explanation.append("📰 SAMPLE HEADLINES:")
            for i, headline in enumerate(news_assessment['sample_headlines'][:3], 1):
                explanation.append(f"{i}. {headline}")
            explanation.append("")
        
        explanation.append("🎯 CONFIDENCE ASSESSMENT:")
        explanation.append(f"• Overall confidence: {confidence_analysis['overall_confidence']:.1%} ({confidence_analysis['confidence_level']})")
        explanation.append(f"• Data sufficiency: {confidence_analysis['data_sufficiency']}")
        
        return "\n".join(explanation)
    
    def _generate_actionable_insights(self, risk_score: float, 
                                    sentiment_analysis: Dict[str, Any],
                                    risk_factor_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable insights based on the analysis"""
        
        insights = []
        
        if risk_score > 70:
            insights.append("🚨 HIGH PRIORITY: Monitor for immediate developments that could affect liquidity or operations")
        elif risk_score > 50:
            insights.append("⚠️ MODERATE PRIORITY: Watch for trend continuation and specific risk factor developments")
        else:
            insights.append("✅ LOW PRIORITY: Maintain standard monitoring protocols")
        
        primary_concerns = risk_factor_analysis.get('primary_concerns', [])
        if primary_concerns:
            top_concern = primary_concerns[0]
            insights.append(f"🎯 Focus monitoring on: {top_concern['name']} ({top_concern['mentions']} mentions)")
        
        return insights
    
    def _assess_data_quality(self, news_assessment: Dict[str, Any]) -> Dict[str, str]:
        """Assess the quality and reliability of the underlying data"""
        
        detailed = news_assessment.get('detailed_analysis', {})
        articles_count = detailed.get('articles_analyzed', 0)
        
        if articles_count >= 15:
            coverage = "Excellent"
        elif articles_count >= 10:
            coverage = "Good"
        elif articles_count >= 5:
            coverage = "Adequate"
        else:
            coverage = "Limited"
        
        return {
            'coverage_quality': coverage,
            'data_recency': "Current (7-day window)",
            'source_diversity': "Multiple sources" if articles_count > 5 else "Limited sources",
            'overall_quality': coverage
        }
    
    def _categorize_risk_level(self, score: float) -> str:
        """Categorize risk level based on score"""
        if score < 25:
            return "LOW RISK"
        elif score < 45:
            return "MODERATE-LOW RISK"
        elif score < 65:
            return "MODERATE-HIGH RISK"
        else:
            return "HIGH RISK"
    
    def _create_no_data_explanation(self) -> Dict[str, Any]:
        """Create explanation when no data is available"""
        return {
            'risk_score': 50.0,
            'risk_level': 'MODERATE (NO DATA)',
            'explanation': 'No recent news data available for analysis. Using neutral risk assessment.',
            'confidence': 0.1,
            'data_quality': {'overall_quality': 'No Data'},
            'actionable_insights': ['Seek alternative data sources for risk assessment']
        }

def explain_news_assessment(news_assessment: Dict[str, Any]) -> Dict[str, Any]:
    """Main convenience function to generate explanations for news assessments."""
    engine = NewsExplainabilityEngine()
    return engine.analyze_news_impact(news_assessment)

# === MAIN EXPLAINABILITY FUNCTIONS ===

def explain_structured_score(processed_features: Dict[str, Any], company_name: str) -> Dict[str, Any]:
    """
    Generate explanation for structured risk score using embedded EBMExplainer.
    """
    try:
        logger.info(f"📊 Generating structured explanation for {company_name}")
        
        MODEL_PATH = r"C:\Users\asus\Documents\GitHub\BlackBox_Cred\credtech_backend\models\ebm_model_trained_on_csv.pkl"
        explainer = EBMExplainer(MODEL_PATH)
        
        explanation_result = explainer.explain_single_prediction(processed_features, company_name)
        
        logger.info(f"✅ Structured explanation generated for {company_name}")
        
        return {
            'explanation_text': explanation_result.get('explanation_text', 'No explanation available'),
            'prediction': explanation_result.get('prediction', 'Unknown'),
            'investment_grade_probability': explanation_result.get('probability_investment_grade', 0.5),
            'non_investment_grade_probability': explanation_result.get('probability_non_investment_grade', 0.5),
            'feature_contributions': explanation_result.get('feature_contributions', {}),
            'explanation_type': 'EBM Structured Analysis',
            'company': company_name,
            'explanation_confidence': 'High',
            'method': 'Explainable Boosting Machine (EBM) with SHAP-style explanations'
        }
        
    except Exception as e:
        logger.error(f"Error generating structured explanation for {company_name}: {e}")
        return {
            'explanation_text': f"Error generating structured explanation for {company_name}: {str(e)}",
            'prediction': 'Error',
            'investment_grade_probability': 0.5,
            'explanation_type': 'Error',
            'error': str(e),
            'company': company_name
        }

def explain_unstructured_score(unstructured_result: Dict[str, Any], company_name: str) -> Dict[str, Any]:
    """
    Generate explanation for unstructured risk score using embedded NewsExplainabilityEngine.
    """
    try:
        logger.info(f"📰 Generating unstructured explanation for {company_name}")
        
        news_assessment = {
            'company': company_name,
            'risk_score': unstructured_result.get('risk_score', 50.0),
            'confidence': unstructured_result.get('confidence', 0.5),
            'detailed_analysis': {
                'articles_analyzed': unstructured_result.get('articles_analyzed', 0),
                'sentiment_distribution': unstructured_result.get('sentiment_distribution', {}),
                'temporal_trend': unstructured_result.get('temporal_trend', 0.0),
                'risk_keywords': unstructured_result.get('risk_keywords', []),
                'base_sentiment_score': unstructured_result.get('base_sentiment_score', 50.0),
                'avg_risk_score': unstructured_result.get('avg_risk_score', 0.0)
            },
            'sample_headlines': unstructured_result.get('sample_headlines', [])
        }
        
        explanation_result = explain_news_assessment(news_assessment)
        
        logger.info(f"✅ Unstructured explanation generated for {company_name}")
        
        return {
            'explanation': explanation_result.get('explanation', 'No explanation available'),
            'risk_level': explanation_result.get('risk_level', 'Unknown'),
            'confidence': explanation_result.get('confidence', 0.5),
            'sentiment_analysis': explanation_result.get('sentiment_analysis', {}),
            'risk_factor_analysis': explanation_result.get('risk_factor_analysis', {}),
            'temporal_analysis': explanation_result.get('temporal_analysis', {}),
            'confidence_analysis': explanation_result.get('confidence_analysis', {}),
            'actionable_insights': explanation_result.get('actionable_insights', []),
            'data_quality': explanation_result.get('data_quality', {}),
            'explanation_type': 'News Sentiment Analysis',
            'company': company_name,
            'method': 'FinBERT + Financial Risk Detection + Temporal Analysis'
        }
        
    except Exception as e:
        logger.error(f"Error generating unstructured explanation for {company_name}: {e}")
        return {
            'explanation': f"Error generating unstructured explanation for {company_name}: {str(e)}",
            'risk_level': 'Error',
            'confidence': 0.1,
            'explanation_type': 'Error',
            'error': str(e),
            'company': company_name
        }

def explain_fusion(fusion_result: Dict[str, Any], structured_result: Dict[str, Any], 
                  unstructured_result: Dict[str, Any], company_name: str) -> Dict[str, Any]:
    """
    Generate explanation for the fusion process and final score.
    """
    try:
        logger.info(f"🔄 Generating fusion explanation for {company_name}")
        
        final_score = fusion_result.get('fused_score', 50.0)
        expert_agreement = fusion_result.get('expert_agreement', 0.5)
        market_regime = fusion_result.get('market_regime', 'NORMAL')
        dynamic_weights = fusion_result.get('dynamic_weights', {})
        expert_contributions = fusion_result.get('expert_contributions', {})
        regime_adjustment = fusion_result.get('regime_adjustment', 0.0)
        
        explanation_lines = []
        explanation_lines.append(f"MAESTRO FUSION ANALYSIS FOR {company_name.upper()}")
        explanation_lines.append("=" * 60)
        explanation_lines.append("")
        
        explanation_lines.append(f"FINAL FUSED RISK SCORE: {final_score:.1f}/100")
        explanation_lines.append(f"Expert Agreement Level: {expert_agreement:.1%}")
        explanation_lines.append(f"Market Regime: {market_regime}")
        explanation_lines.append("")
        
        explanation_lines.append("EXPERT CONTRIBUTIONS:")
        explanation_lines.append("-" * 30)
        
        if 'structured_expert' in expert_contributions:
            struct_contrib = expert_contributions['structured_expert']
            struct_weight = dynamic_weights.get('structured_expert', 0.5)
            explanation_lines.append(f"Structured Analysis (EBM):")
            explanation_lines.append(f"  Score: {struct_contrib.get('risk_score', 0):.1f}/100")
            explanation_lines.append(f"  Weight: {struct_weight:.1%}")
            explanation_lines.append(f"  Contribution: {struct_contrib.get('contribution', 0):.1f} points")
            explanation_lines.append("")
        
        if 'news_sentiment_expert' in expert_contributions:
            news_contrib = expert_contributions['news_sentiment_expert']
            news_weight = dynamic_weights.get('news_sentiment_expert', 0.5)
            explanation_lines.append(f"News Sentiment Analysis:")
            explanation_lines.append(f"  Score: {news_contrib.get('risk_score', 0):.1f}/100")
            explanation_lines.append(f"  Weight: {news_weight:.1%}")
            explanation_lines.append(f"  Contribution: {news_contrib.get('contribution', 0):.1f} points")
            explanation_lines.append("")
        
        if regime_adjustment != 0:
            explanation_lines.append(f"MARKET REGIME ADJUSTMENT:")
            explanation_lines.append(f"  Adjustment: {regime_adjustment:+.1f} points")
            explanation_lines.append("")
        
        explanation_lines.append("FUSION METHODOLOGY:")
        explanation_lines.append("MAESTRO (Multi-Agent Explainable Adaptive STructured-Textual Risk Oracle)")
        explanation_lines.append("Dynamic weighted fusion with market condition adjustments")
        
        fusion_explanation = "\n".join(explanation_lines)
        
        logger.info(f"✅ Fusion explanation generated for {company_name}")
        
        return {
            'explanation': fusion_explanation,
            'final_score': final_score,
            'expert_agreement': expert_agreement,
            'market_regime': market_regime,
            'dynamic_weights': dynamic_weights,
            'expert_contributions': expert_contributions,
            'regime_adjustment': regime_adjustment,
            'fusion_method': 'MAESTRO Dynamic Weighted Fusion',
            'explanation_type': 'Fusion Analysis',
            'company': company_name
        }
        
    except Exception as e:
        logger.error(f"Error generating fusion explanation for {company_name}: {e}")
        return {
            'explanation': f"Error generating fusion explanation for {company_name}: {str(e)}",
            'final_score': fusion_result.get('fused_score', 50.0),
            'explanation_type': 'Error',
            'error': str(e),
            'company': company_name
        }
        
        logger.info(f"📊 Generating structured explanation for {company_name}")
        
        # Initialize explainer with the exact model path from ebm_exp.py
        MODEL_PATH = r"C:\Users\asus\Documents\GitHub\BlackBox_Cred\credtech_backend\models\ebm_model_trained_on_csv.pkl"
        explainer = EBMExplainer(MODEL_PATH)
        
        # Generate single prediction explanation using exact method from ebm_exp.py
        explanation_result = explainer.explain_single_prediction(processed_features, company_name)
        
        logger.info(f"✅ Structured explanation generated for {company_name}")
        
        return {
            'explanation_text': explanation_result.get('explanation_text', 'No explanation available'),
            'prediction': explanation_result.get('prediction', 'Unknown'),
            'investment_grade_probability': explanation_result.get('probability_investment_grade', 0.5),
            'non_investment_grade_probability': explanation_result.get('probability_non_investment_grade', 0.5),
            'feature_contributions': explanation_result.get('feature_contributions', {}),
            'explanation_type': 'EBM Structured Analysis',
            'company': company_name,
            
            # Additional analysis for consistency
            'top_risk_factors': _extract_top_risk_factors(explanation_result.get('feature_contributions', {})),
            'top_positive_factors': _extract_top_positive_factors(explanation_result.get('feature_contributions', {})),
            'explanation_confidence': 'High',  # EBM explanations are always high confidence
            'method': 'Explainable Boosting Machine (EBM) with SHAP-style explanations'
        }
        
    except Exception as e:
        logger.error(f"Error generating structured explanation for {company_name}: {e}")
        return {
            'explanation_text': f"Error generating structured explanation for {company_name}: {str(e)}",
            'prediction': 'Error',
            'investment_grade_probability': 0.5,
            'explanation_type': 'Error',
            'error': str(e),
            'company': company_name
        }

def explain_unstructured_score(unstructured_result: Dict[str, Any], company_name: str) -> Dict[str, Any]:
    """
    Generate explanation for unstructured risk score using NewsExplainabilityEngine from news_explainability.py.
    
    Args:
        unstructured_result: Result from unstructured analysis
        company_name: Name of the company for personalized explanation
        
    Returns:
        Dictionary with unstructured explanation details
    """
    try:
        logger.info(f"📰 Generating unstructured explanation for {company_name}")
        
        # Create mock news assessment format for the explainer
        # The unstructured_result should contain the detailed analysis from news_unstructured_score.py
        news_assessment = {
            'company': company_name,
            'risk_score': unstructured_result.get('risk_score', 50.0),
            'confidence': unstructured_result.get('confidence', 0.5),
            'detailed_analysis': {
                'articles_analyzed': unstructured_result.get('articles_analyzed', 0),
                'sentiment_distribution': unstructured_result.get('sentiment_distribution', {}),
                'temporal_trend': unstructured_result.get('temporal_trend', 0.0),
                'risk_keywords': unstructured_result.get('risk_keywords', []),
                'base_sentiment_score': unstructured_result.get('base_sentiment_score', 50.0),
                'avg_risk_score': unstructured_result.get('avg_risk_score', 0.0)
            },
            'sample_headlines': unstructured_result.get('sample_headlines', [])
        }
        
        # Generate explanation using exact method from news_explainability.py
        explanation_result = explain_news_assessment(news_assessment)
        
        logger.info(f"✅ Unstructured explanation generated for {company_name}")
        
        return {
            'explanation': explanation_result.get('explanation', 'No explanation available'),
            'risk_level': explanation_result.get('risk_level', 'Unknown'),
            'confidence': explanation_result.get('confidence', 0.5),
            'sentiment_analysis': explanation_result.get('sentiment_analysis', {}),
            'risk_factor_analysis': explanation_result.get('risk_factor_analysis', {}),
            'temporal_analysis': explanation_result.get('temporal_analysis', {}),
            'confidence_analysis': explanation_result.get('confidence_analysis', {}),
            'actionable_insights': explanation_result.get('actionable_insights', []),
            'data_quality': explanation_result.get('data_quality', {}),
            'explanation_type': 'News Sentiment Analysis',
            'company': company_name,
            
            # Additional summary information
            'articles_count': news_assessment['detailed_analysis']['articles_analyzed'],
            'sentiment_summary': _summarize_sentiment(explanation_result.get('sentiment_analysis', {})),
            'risk_factors_summary': _summarize_risk_factors(explanation_result.get('risk_factor_analysis', {})),
            'method': 'FinBERT + Financial Risk Detection + Temporal Analysis'
        }
        
    except Exception as e:
        logger.error(f"Error generating unstructured explanation for {company_name}: {e}")
        return {
            'explanation': f"Error generating unstructured explanation for {company_name}: {str(e)}",
            'risk_level': 'Error',
            'confidence': 0.1,
            'explanation_type': 'Error',
            'error': str(e),
            'company': company_name
        }

def explain_fusion(fusion_result: Dict[str, Any], structured_result: Dict[str, Any], 
                  unstructured_result: Dict[str, Any], company_name: str) -> Dict[str, Any]:
    """
    Generate explanation for the fusion process and final score.
    
    Args:
        fusion_result: Result from MAESTRO fusion
        structured_result: Original structured analysis result
        unstructured_result: Original unstructured analysis result
        company_name: Name of the company
        
    Returns:
        Dictionary with fusion explanation details
    """
    try:
        logger.info(f"🔄 Generating fusion explanation for {company_name}")
        
        # Extract key fusion details
        final_score = fusion_result.get('fused_score', 50.0)
        expert_agreement = fusion_result.get('expert_agreement', 0.5)
        market_regime = fusion_result.get('market_regime', 'NORMAL')
        dynamic_weights = fusion_result.get('dynamic_weights', {})
        expert_contributions = fusion_result.get('expert_contributions', {})
        regime_adjustment = fusion_result.get('regime_adjustment', 0.0)
        
        # Generate fusion explanation text
        explanation_lines = []
        explanation_lines.append(f"MAESTRO FUSION ANALYSIS FOR {company_name.upper()}")
        explanation_lines.append("=" * 60)
        explanation_lines.append("")
        
        explanation_lines.append(f"FINAL FUSED RISK SCORE: {final_score:.1f}/100")
        explanation_lines.append(f"Expert Agreement Level: {expert_agreement:.1%}")
        explanation_lines.append(f"Market Regime: {market_regime}")
        explanation_lines.append("")
        
        explanation_lines.append("EXPERT CONTRIBUTIONS:")
        explanation_lines.append("-" * 30)
        
        # Structured expert contribution
        if 'structured_expert' in expert_contributions:
            struct_contrib = expert_contributions['structured_expert']
            struct_weight = dynamic_weights.get('structured_expert', 0.5)
            explanation_lines.append(f"Structured Analysis (EBM):")
            explanation_lines.append(f"  Score: {struct_contrib.get('risk_score', 0):.1f}/100")
            explanation_lines.append(f"  Weight: {struct_weight:.1%}")
            explanation_lines.append(f"  Contribution: {struct_contrib.get('contribution', 0):.1f} points")
            explanation_lines.append("")
        
        # Unstructured expert contribution
        if 'news_sentiment_expert' in expert_contributions:
            news_contrib = expert_contributions['news_sentiment_expert']
            news_weight = dynamic_weights.get('news_sentiment_expert', 0.5)
            explanation_lines.append(f"News Sentiment Analysis:")
            explanation_lines.append(f"  Score: {news_contrib.get('risk_score', 0):.1f}/100")
            explanation_lines.append(f"  Weight: {news_weight:.1%}")
            explanation_lines.append(f"  Contribution: {news_contrib.get('contribution', 0):.1f} points")
            explanation_lines.append("")
        
        # Market adjustment
        if regime_adjustment != 0:
            explanation_lines.append(f"MARKET REGIME ADJUSTMENT:")
            explanation_lines.append(f"  Market Regime: {market_regime}")
            explanation_lines.append(f"  Adjustment: {regime_adjustment:+.1f} points")
            explanation_lines.append(f"  Reason: Market conditions warrant risk adjustment")
            explanation_lines.append("")
        
        # Weight justification
        explanation_lines.append("DYNAMIC WEIGHT JUSTIFICATION:")
        explanation_lines.append("-" * 35)
        
        if dynamic_weights.get('news_sentiment_expert', 0.4) > 0.5:
            explanation_lines.append("• News sentiment given higher weight due to:")
            explanation_lines.append("  - High market volatility conditions")
            explanation_lines.append("  - Elevated economic stress indicators")
        elif dynamic_weights.get('structured_expert', 0.6) > 0.7:
            explanation_lines.append("• Structured analysis given higher weight due to:")
            explanation_lines.append("  - Stable market conditions")
            explanation_lines.append("  - High confidence in financial metrics")
        else:
            explanation_lines.append("• Balanced weighting applied:")
            explanation_lines.append("  - Normal market conditions")
            explanation_lines.append("  - Standard confidence levels")
        
        explanation_lines.append("")
        explanation_lines.append("FUSION METHODOLOGY:")
        explanation_lines.append("MAESTRO (Multi-Agent Explainable Adaptive STructured-Textual Risk Oracle)")
        explanation_lines.append("Dynamic weighted fusion with market condition adjustments")
        
        fusion_explanation = "\n".join(explanation_lines)
        
        logger.info(f"✅ Fusion explanation generated for {company_name}")
        
        return {
            'explanation': fusion_explanation,
            'final_score': final_score,
            'expert_agreement': expert_agreement,
            'market_regime': market_regime,
            'dynamic_weights': dynamic_weights,
            'expert_contributions': expert_contributions,
            'regime_adjustment': regime_adjustment,
            'fusion_method': 'MAESTRO Dynamic Weighted Fusion',
            'explanation_type': 'Fusion Analysis',
            'company': company_name,
            
            # Input score summary
            'input_summary': {
                'structured_score': structured_result.get('risk_score', 50.0),
                'unstructured_score': unstructured_result.get('risk_score', 50.0),
                'score_difference': abs(structured_result.get('risk_score', 50.0) - unstructured_result.get('risk_score', 50.0)),
                'agreement_level': 'High' if expert_agreement > 0.7 else 'Moderate' if expert_agreement > 0.5 else 'Low'
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating fusion explanation for {company_name}: {e}")
        return {
            'explanation': f"Error generating fusion explanation for {company_name}: {str(e)}",
            'final_score': fusion_result.get('fused_score', 50.0),
            'explanation_type': 'Error',
            'error': str(e),
            'company': company_name
        }

def _extract_top_risk_factors(feature_contributions: Dict[str, float], top_n: int = 5) -> list:
    """Extract top risk factors (features contributing to higher risk)"""
    # For investment grade prediction, positive contributions reduce risk, negative increase risk
    # For non-investment grade prediction, positive contributions increase risk
    risk_factors = [(feature, abs(contrib)) for feature, contrib in feature_contributions.items() if contrib < 0]
    risk_factors.sort(key=lambda x: x[1], reverse=True)
    return [factor[0] for factor in risk_factors[:top_n]]

def _extract_top_positive_factors(feature_contributions: Dict[str, float], top_n: int = 5) -> list:
    """Extract top positive factors (features contributing to lower risk)"""
    positive_factors = [(feature, contrib) for feature, contrib in feature_contributions.items() if contrib > 0]
    positive_factors.sort(key=lambda x: x[1], reverse=True)
    return [factor[0] for factor in positive_factors[:top_n]]

def _summarize_sentiment(sentiment_analysis: Dict[str, Any]) -> str:
    """Summarize sentiment analysis results"""
    overall_sentiment = sentiment_analysis.get('overall_sentiment', 'unknown')
    distribution = sentiment_analysis.get('distribution', {})
    
    if overall_sentiment == 'predominantly_negative':
        return f"Predominantly negative ({distribution.get('negative', 0):.0f}% negative articles)"
    elif overall_sentiment == 'predominantly_positive':
        return f"Predominantly positive ({distribution.get('positive', 0):.0f}% positive articles)"
    else:
        return f"Mixed sentiment ({distribution.get('positive', 0):.0f}% pos, {distribution.get('negative', 0):.0f}% neg)"

def _summarize_risk_factors(risk_factor_analysis: Dict[str, Any]) -> str:
    """Summarize risk factor analysis results"""
    primary_concerns = risk_factor_analysis.get('primary_concerns', [])
    if not primary_concerns:
        return "No significant risk factors detected"
    
    top_concern = primary_concerns[0]
    return f"Primary concern: {top_concern.get('name', 'Unknown')} ({top_concern.get('mentions', 0)} mentions)"
