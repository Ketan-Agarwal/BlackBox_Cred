"""
news_explainability.py

Advanced explainability system for news-based risk analysis.
Works with the enhanced news_unstructured_score.py pipeline to provide detailed,
actionable explanations of risk assessments based on financial news sentiment.
"""

import datetime
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import re
import json


class NewsExplainabilityEngine:
    """
    Comprehensive explainability engine for news-based risk analysis.
    Provides detailed, human-readable explanations that help understand
    how news sentiment translates to risk scores.
    """
    
    def __init__(self):
        # Enhanced risk categories with detailed explanations
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
            },
            'operational_issues': {
                'name': 'Operational Disruptions',
                'description': 'Problems with core business operations',
                'impact': 'Moderate risk affecting revenue generation',
                'keywords': ['supply chain', 'production halt', 'strike', 'management departure',
                           'key personnel', 'operational disruption', 'facility closure']
            },
            'regulatory_legal': {
                'name': 'Regulatory & Legal Challenges',
                'description': 'Legal issues and regulatory compliance problems',
                'impact': 'Significant risk from fines, penalties, and reputation damage',
                'keywords': ['lawsuit', 'investigation', 'regulatory penalty', 'compliance violation',
                           'fine', 'audit', 'legal action', 'regulatory scrutiny']
            },
            'market_competition': {
                'name': 'Market & Competitive Pressures',
                'description': 'Challenges from market conditions and competition',
                'impact': 'Lower direct risk but affects long-term profitability',
                'keywords': ['market share loss', 'competitive pressure', 'demand decline',
                           'pricing pressure', 'market downturn', 'revenue decline']
            },
            'cyber_technology': {
                'name': 'Technology & Cybersecurity Risks',
                'description': 'Technology disruptions and cyber threats',
                'impact': 'Moderate risk with potential for severe operational impact',
                'keywords': ['cyber attack', 'data breach', 'system failure', 'technology disruption',
                           'cybersecurity', 'digital transformation', 'IT outage']
            }
        }
        
        # Positive indicators for balanced analysis
        self.positive_indicators = {
            'financial_strength': ['strong earnings', 'profit growth', 'revenue increase', 'cash generation'],
            'market_expansion': ['market expansion', 'new market', 'geographic expansion', 'customer growth'],
            'innovation': ['innovation', 'new product', 'technology advancement', 'R&D investment'],
            'partnerships': ['strategic partnership', 'acquisition', 'merger', 'joint venture'],
            'operational_excellence': ['operational efficiency', 'cost reduction', 'productivity improvement']
        }
        
        # Sentiment impact weights for explanation
        self.sentiment_impact = {
            'very_negative': {'threshold': -0.7, 'description': 'Extremely negative', 'risk_impact': 'Very High'},
            'negative': {'threshold': -0.3, 'description': 'Negative', 'risk_impact': 'High'},
            'neutral': {'threshold': 0.3, 'description': 'Neutral/Mixed', 'risk_impact': 'Moderate'},
            'positive': {'threshold': 0.7, 'description': 'Positive', 'risk_impact': 'Low'},
            'very_positive': {'threshold': 1.0, 'description': 'Very positive', 'risk_impact': 'Very Low'}
        }
    
    def analyze_news_impact(self, news_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main function to analyze and explain news impact.
        Takes the output from get_news_risk_assessment() and generates detailed explanations.
        """
        
        if not news_assessment or 'detailed_analysis' not in news_assessment:
            return self._create_no_data_explanation()
        
        detailed = news_assessment['detailed_analysis']
        risk_score = news_assessment.get('risk_score', 50)
        
        # Core analysis components
        sentiment_analysis = self._analyze_sentiment_impact(detailed)
        risk_factor_analysis = self._analyze_risk_factors(news_assessment)
        temporal_analysis = self._analyze_temporal_trends(detailed)
        confidence_analysis = self._analyze_confidence_factors(news_assessment)
        
        # Generate comprehensive explanation
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
        
        # Calculate percentages
        sentiment_percentages = {
            sentiment: (count / total_articles) * 100 
            for sentiment, count in sentiment_dist.items()
        }
        
        # Determine overall sentiment
        if sentiment_percentages.get('negative', 0) > 60:
            overall_sentiment = 'predominantly_negative'
            impact_desc = 'High negative impact on perceived risk'
        elif sentiment_percentages.get('positive', 0) > 60:
            overall_sentiment = 'predominantly_positive'
            impact_desc = 'Positive impact reducing perceived risk'
        elif abs(sentiment_percentages.get('negative', 0) - sentiment_percentages.get('positive', 0)) < 15:
            overall_sentiment = 'mixed'
            impact_desc = 'Neutral impact with mixed signals'
        else:
            overall_sentiment = 'moderate'
            impact_desc = 'Moderate impact with slight bias'
        
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
        
        # Categorize risk keywords
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
        
        # Identify primary concerns
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
        
        # Article count factor
        if articles_count >= 15:
            confidence_factors.append('Sufficient news coverage (15+ articles)')
        elif articles_count >= 8:
            confidence_factors.append('Adequate news coverage (8-14 articles)')
        elif articles_count >= 3:
            confidence_factors.append('Limited news coverage (3-7 articles)')
        else:
            confidence_factors.append('Very limited news coverage (<3 articles)')
        
        # Time span factor (assume 7 days for now)
        confidence_factors.append('Analysis covers recent 7-day period')
        
        # Model confidence
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
        
        # Header
        explanation.append(f"🔍 NEWS SENTIMENT RISK ANALYSIS")
        explanation.append(f"Risk Score: {risk_score:.1f}/100 ({self._categorize_risk_level(risk_score)})")
        explanation.append("")
        
        # Executive Summary
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
        
        # Sentiment Analysis
        explanation.append("📈 SENTIMENT BREAKDOWN:")
        dist = sentiment_analysis['distribution']
        explanation.append(f"• Positive articles: {dist.get('positive', 0):.1f}%")
        explanation.append(f"• Neutral articles: {dist.get('neutral', 0):.1f}%")
        explanation.append(f"• Negative articles: {dist.get('negative', 0):.1f}%")
        explanation.append(f"• Overall sentiment: {sentiment_analysis['overall_sentiment'].replace('_', ' ').title()}")
        explanation.append(f"• Impact assessment: {sentiment_analysis['impact']}")
        explanation.append("")
        
        # Risk Factors
        if risk_factor_analysis['primary_concerns']:
            explanation.append("⚠️ PRIMARY RISK FACTORS:")
            for concern in risk_factor_analysis['primary_concerns']:
                explanation.append(f"• {concern['name']}: {concern['mentions']} mentions")
                explanation.append(f"  Impact: {concern['impact']}")
            explanation.append("")
        
        # Temporal Trends
        explanation.append("📅 TEMPORAL ANALYSIS:")
        explanation.append(f"• Trend direction: {temporal_analysis['trend_direction']}")
        explanation.append(f"• Impact: {temporal_analysis['impact_assessment']}")
        explanation.append(f"• Significance: {temporal_analysis['significance']}")
        explanation.append("")
        
        # Sample Headlines
        if 'sample_headlines' in news_assessment and news_assessment['sample_headlines']:
            explanation.append("📰 SAMPLE HEADLINES:")
            for i, headline in enumerate(news_assessment['sample_headlines'][:3], 1):
                explanation.append(f"{i}. {headline}")
            explanation.append("")
        
        # Confidence Assessment
        explanation.append("🎯 CONFIDENCE ASSESSMENT:")
        explanation.append(f"• Overall confidence: {confidence_analysis['overall_confidence']:.1%} ({confidence_analysis['confidence_level']})")
        explanation.append(f"• Data sufficiency: {confidence_analysis['data_sufficiency']}")
        for factor in confidence_analysis['contributing_factors']:
            explanation.append(f"• {factor}")
        
        return "\n".join(explanation)
    
    def _generate_actionable_insights(self, risk_score: float, 
                                    sentiment_analysis: Dict[str, Any],
                                    risk_factor_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable insights based on the analysis"""
        
        insights = []
        
        # Risk level specific insights
        if risk_score > 70:
            insights.append("🚨 HIGH PRIORITY: Monitor for immediate developments that could affect liquidity or operations")
            insights.append("📊 Consider increasing monitoring frequency and stress testing scenarios")
        elif risk_score > 50:
            insights.append("⚠️ MODERATE PRIORITY: Watch for trend continuation and specific risk factor developments")
            insights.append("📈 Consider scenario analysis for identified risk categories")
        else:
            insights.append("✅ LOW PRIORITY: Maintain standard monitoring protocols")
        
        # Specific risk factor insights
        primary_concerns = risk_factor_analysis.get('primary_concerns', [])
        if primary_concerns:
            top_concern = primary_concerns[0]
            insights.append(f"🎯 Focus monitoring on: {top_concern['name']} ({top_concern['mentions']} mentions)")
        
        # Sentiment specific insights
        if sentiment_analysis['overall_sentiment'] == 'predominantly_negative':
            insights.append("📰 Consider proactive communication strategy to address negative sentiment")
        elif sentiment_analysis['overall_sentiment'] == 'mixed':
            insights.append("🔍 Investigate specific drivers of negative sentiment within mixed coverage")
        
        return insights
    
    def _assess_data_quality(self, news_assessment: Dict[str, Any]) -> Dict[str, str]:
        """Assess the quality and reliability of the underlying data"""
        
        detailed = news_assessment.get('detailed_analysis', {})
        articles_count = detailed.get('articles_analyzed', 0)
        
        # Coverage assessment
        if articles_count >= 15:
            coverage = "Excellent"
        elif articles_count >= 10:
            coverage = "Good"
        elif articles_count >= 5:
            coverage = "Adequate"
        else:
            coverage = "Limited"
        
        # Recency assessment (assuming 7-day window)
        recency = "Current (7-day window)"
        
        # Source diversity (placeholder - could be enhanced with actual source analysis)
        source_diversity = "Multiple sources" if articles_count > 5 else "Limited sources"
        
        return {
            'coverage_quality': coverage,
            'data_recency': recency,
            'source_diversity': source_diversity,
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


# === Convenience Functions ===

def explain_news_assessment(news_assessment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main convenience function to generate explanations for news assessments.
    
    Args:
        news_assessment: Output from get_news_risk_assessment() function
        
    Returns:
        Comprehensive explanation dictionary
    """
    engine = NewsExplainabilityEngine()
    return engine.analyze_news_impact(news_assessment)


def get_company_news_explanation(company: str, days_back: int = 7, max_articles: int = 20) -> Dict[str, Any]:
    """
    Complete pipeline: Get news assessment from news_unstructured_score.py and generate explanation.
    
    Args:
        company: Company name to analyze
        days_back: Number of days to look back for news
        max_articles: Maximum articles to analyze
        
    Returns:
        Complete explanation with news assessment and detailed analysis
    """
    try:
        # Import the news assessment function
        from news_unstructured_score import get_news_risk_assessment
        
        print(f"🔍 Analyzing news for {company}...")
        
        # Get news risk assessment
        news_assessment = get_news_risk_assessment(company, days_back, max_articles)
        
        # Add company name to assessment if not present
        if 'company' not in news_assessment:
            news_assessment['company'] = company
        
        print(f"✅ News assessment complete. Risk Score: {news_assessment.get('risk_score', 'N/A')}")
        
        # Generate detailed explanation
        explanation_result = explain_news_assessment(news_assessment)
        
        # Combine both results
        complete_result = {
            'company': company,
            'news_assessment': news_assessment,
            'explanation_analysis': explanation_result,
            'integration_timestamp': datetime.datetime.now().isoformat()
        }
        
        print(f"📊 Explanation generated successfully")
        return complete_result
        
    except ImportError as e:
        print(f"❌ Error importing news_unstructured_score: {e}")
        return {
            'error': 'Could not import news_unstructured_score module',
            'company': company,
            'explanation_analysis': {'explanation': 'Module import failed'}
        }
    except Exception as e:
        print(f"❌ Error in complete pipeline: {e}")
        return {
            'error': str(e),
            'company': company,
            'explanation_analysis': {'explanation': f'Pipeline error: {e}'}
        }


def generate_explanation_report(company: str, news_assessment: Dict[str, Any], 
                              save_to_file: bool = True) -> str:
    """
    Generate a detailed explanation report and optionally save to file.
    
    Args:
        company: Company name
        news_assessment: Output from get_news_risk_assessment()
        save_to_file: Whether to save the report to a file
        
    Returns:
        Explanation text
    """
    explanation_data = explain_news_assessment(news_assessment)
    
    report_text = explanation_data.get('explanation', 'No explanation available')
    
    if save_to_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"explanations/news_explanation_{company.replace(' ', '_')}_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_text)
                f.write("\n\n" + "="*50)
                f.write(f"\nGenerated: {datetime.datetime.now()}")
                f.write(f"\nCompany: {company}")
                f.write(f"\nRisk Score: {explanation_data.get('risk_score', 'N/A')}")
                f.write(f"\nConfidence: {explanation_data.get('confidence', 'N/A'):.1%}")
            print(f"📁 Explanation report saved to: {filename}")
        except Exception as e:
            print(f"⚠️ Could not save report to file: {e}")
    
    return report_text


# === Example Usage ===
if __name__ == "__main__":
    print("🧪 News Explainability Engine - Test Mode")
    print("="*60)
    
    # Test with real integration
    test_companies = ["Apple", "Tesla", "Microsoft"]
    
    for company in test_companies:
        print(f"\n🏢 Testing complete pipeline for {company}...")
        print("-" * 40)
        
        try:
            # Test complete integration
            result = get_company_news_explanation(company, days_back=7, max_articles=10)
            
            if 'error' not in result:
                # Print key results
                news_data = result['news_assessment']
                explanation_data = result['explanation_analysis']
                
                print(f"📊 Risk Score: {news_data.get('risk_score', 'N/A'):.1f}/100")
                print(f"🎯 Confidence: {news_data.get('confidence', 'N/A'):.1%}")
                print(f"📰 Articles Analyzed: {news_data.get('detailed_analysis', {}).get('articles_analyzed', 0)}")
                
                # Print explanation summary
                explanation_text = explanation_data.get('explanation', 'No explanation available')
                lines = explanation_text.split('\n')
                print("\n📋 EXPLANATION SUMMARY:")
                for line in lines[:10]:  # First 10 lines
                    if line.strip():
                        print(f"  {line}")
                
                # Print actionable insights
                insights = explanation_data.get('actionable_insights', [])
                if insights:
                    print("\n💡 KEY INSIGHTS:")
                    for insight in insights[:3]:  # First 3 insights
                        print(f"  {insight}")
                
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Test failed for {company}: {e}")
        
        print("-" * 40)
    
    print("\n✅ Integration testing completed!")
    print("="*60)
    
    # Also test the mock assessment for backward compatibility
    print("\n🔬 Testing with mock data...")
    mock_assessment = {
        'risk_score': 72.5,
        'confidence': 0.85,
        'company': 'Example Corp',
        'detailed_analysis': {
            'articles_analyzed': 12,
            'sentiment_distribution': {'positive': 2, 'neutral': 3, 'negative': 7},
            'temporal_trend': -3.2,
            'risk_keywords': [('debt default', 3), ('investigation', 2), ('lawsuit', 1)]
        },
        'sample_headlines': [
            'Example Corp faces investigation into financial practices',
            'Debt concerns mount for Example Corp',
            'Example Corp stock drops on negative outlook'
        ]
    }
    
    # Generate explanation
    explanation = explain_news_assessment(mock_assessment)
    
    print("\n📊 MOCK EXPLANATION RESULT:")
    print(explanation['explanation'])
    
    print("\n💡 ACTIONABLE INSIGHTS:")
    for insight in explanation['actionable_insights']:
        print(f"  {insight}")
    
    print(f"\n📈 DATA QUALITY: {explanation['data_quality']['overall_quality']}")
    print("="*60)
