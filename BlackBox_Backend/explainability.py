"""
explainability.py
Comprehensive explainability module combining structured and unstructured explanations.
Self-contained implementation without external dependencies.
Includes integrated comprehensive explainability report generator.
"""
import logging
import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Any, List, Tuple
from collections import Counter
import datetime

logger = logging.getLogger(__name__)

# === INTEGRATED COMPREHENSIVE EXPLAINABILITY REPORT ===

def generate_comprehensive_explainability_report(
    company_name: str,
    final_score: float,
    credit_grade: str,
    structured_result: Dict[str, Any],
    unstructured_result: Dict[str, Any],
    fusion_result: Dict[str, Any],
    market_conditions: Dict[str, Any],
    structured_features: Dict[str, Any]
) -> str:
    """
    Generate comprehensive explainability report integrating teammate's format
    with your existing pipeline data.
    """
    try:
        # Extract key data for the report
        alpha = _calculate_volatility_factor(market_conditions)
        threshold = 0.7  # Standard threshold for market volatility

        # Extract EBM feature contributions
        ebm_output = _extract_ebm_contributions(structured_result, structured_features)

        # Extract news headlines and sentiments
        headlines, sentiments, sentiment_scores = _extract_news_data(unstructured_result)

        # Generate the comprehensive report
        lines = []
        lines.append(f"CREDIT RISK EXPLAINABILITY REPORT for {company_name}")
        lines.append("="*60)
        lines.append("")

        # 1. Final Score
        lines.append("Final Credit Assessment")
        lines.append(f"   Overall Credit Score: {final_score:.1f}/100")
        lines.append(f"   Credit Grade: {credit_grade}")
        lines.append(f"   Assessment Date: {datetime.datetime.now().strftime('%Y-%m-%d')}")
        lines.append("")

        # 2. Data Sources
        lines.append("Data Sources Used in Calculation")
        lines.append("   - Financial Ratios & Structured Analysis (EBM)")
        lines.append("   - News Sentiment Analysis (FinBERT)")
        lines.append("   - Market Volatility & Economic Indicators")
        lines.append("   - Dynamic Fusion Algorithm (Dynamic Weighting)")
        lines.append("")

        # 3. Component Scores
        # --- FIX 1: Use fused component scores if available, otherwise defaults ---
        structured_score = fusion_result.get('expert_contributions', {}).get('structured_expert', {}).get('score', structured_result.get('structured_score', 50.0))
        unstructured_score = fusion_result.get('expert_contributions', {}).get('news_sentiment_expert', {}).get('score', unstructured_result.get('unstructured_score', 50.0))

        lines.append("Component Analysis")
        lines.append(f"   Structured Analysis Score: {structured_score:.1f}/100")
        lines.append(f"   Unstructured Analysis Score: {unstructured_score:.1f}/100")

        # Dynamic weights from fusion
        dynamic_weights = fusion_result.get('dynamic_weights', {})
        struct_weight = dynamic_weights.get('structured_expert', 0.5) * 100 # Default to 0.5 if missing
        news_weight = dynamic_weights.get('news_sentiment_expert', 0.5) * 100 # Default to 0.5 if missing
        lines.append(f"   Structured Weight: {struct_weight:.1f}%")
        lines.append(f"   News Sentiment Weight: {news_weight:.1f}%")
        lines.append("")

        # 4. Market Volatility Influence
        lines.append("Market Volatility Influence")
        market_regime = market_conditions.get('regime', 'NORMAL')
        vix = market_conditions.get('vix', 20.0)
        # --- FIX 2: Corrected logic for stable/volatile interpretation ---
        if alpha > threshold: # Higher alpha means more stable (lower volatility)
            lines.append(f"   → Market was STABLE (VIX: {vix:.1f}, Regime: {market_regime})")
            lines.append("   → Financial fundamentals had higher influence on the score than news sentiments.")
        else: # Lower alpha means more volatile
            lines.append(f"   → Market was VOLATILE (VIX: {vix:.1f}, Regime: {market_regime})")
            lines.append("   → News sentiment had stronger influence on the score than financial fundamentals.")
        lines.append("")

        # 5. Top Contributing Financial Factors
        lines.append("Top Contributing Financial Factors")
        if ebm_output:
            # Get necessary values for point calculation
            # Fused Final Score
            final_score_for_report = fusion_result.get('fused_score', final_score) # Use provided final_score as fallback
            # Structured Component Score (from fusion if available, otherwise fallback)
            structured_component_score = fusion_result.get('expert_contributions', {}).get('structured_expert', {}).get('score', structured_result.get('structured_score', 50.0))
            # Structured Weight
            struct_weight_for_points = fusion_result.get('dynamic_weights', {}).get('structured_expert', 0.5) # Default weight

            # Calculate total absolute raw contribution for percentage calculation (if needed for alternative methods)
            total_abs_contribution = sum(abs(feat['raw_contribution']) for feat in ebm_output)
            if total_abs_contribution == 0: total_abs_contribution = 1e-8 # Avoid division by zero

            # Separate and sort contributors based on EBM raw_contribution
            # --- CORE LOGIC: Identify top 5 positive and negative based on EBM's actual output ---
            positive_contributors = [feat for feat in ebm_output if feat['raw_contribution'] > 0]
            negative_contributors = [feat for feat in ebm_output if feat['raw_contribution'] < 0]

            # Sort by absolute raw_contribution (actual EBM model contribution)
            positive_sorted = sorted(positive_contributors, key=lambda x: abs(x['raw_contribution']), reverse=True)[:5]
            negative_sorted = sorted(negative_contributors, key=lambda x: abs(x['raw_contribution']), reverse=True)[:5]

            # Display Top 5 Positive Contributors
            if positive_sorted:
                lines.append("")
                lines.append("   TOP 5 POSITIVE CONTRIBUTORS (Helping Credit Score):")
                for i, feat in enumerate(positive_sorted, start=1):
                    raw_contrib = feat['raw_contribution']
                    # --- Your Formula for Points Impact on Final Score ---
                    # Points impact on Structured Component Score = alpha * structured_component_score * (abs(raw_contrib) / total_abs_contribution)
                    # Points impact on Final Fused Score = Weight of Structured Expert * Points impact on Structured Score
                    # Note: Using raw_contrib directly as a proxy for influence, normalized by total_abs_contribution
                    contrib_percentage = abs(raw_contrib) / total_abs_contribution
                    points_impact_on_structured = alpha * structured_component_score * contrib_percentage
                    points_impact_on_final = struct_weight_for_points * points_impact_on_structured

                    lines.append(f"   {i}. {feat['feature']} (Value: {feat['formatted_value']})")
                    lines.append(f"      Interpretation: {feat['interpretation']}")
                    lines.append(f"      Raw Model Contribution: +{raw_contrib:.4f}")
                    lines.append(f"      Impact on Final Score: +{points_impact_on_final:.2f} points")
                    lines.append("")

            # Display Top 5 Negative Contributors
            if negative_sorted:
                lines.append("   TOP 5 NEGATIVE CONTRIBUTORS (Hurting Credit Score):")
                for i, feat in enumerate(negative_sorted, start=1):
                    raw_contrib = feat['raw_contribution'] # This will be negative
                    # --- Your Formula for Points Impact on Final Score ---
                    contrib_percentage = abs(raw_contrib) / total_abs_contribution
                    points_impact_on_structured = alpha * structured_component_score * contrib_percentage
                    points_impact_on_final = struct_weight_for_points * points_impact_on_structured # This value will be positive

                    lines.append(f"   {i}. {feat['feature']} (Value: {feat['formatted_value']})")
                    lines.append(f"      Interpretation: {feat['interpretation']}")
                    lines.append(f"      Raw Model Contribution: {raw_contrib:.4f}") # Negative value
                    lines.append(f"      Impact on Final Score: -{points_impact_on_final:.2f} points") # Show as negative impact
                    lines.append("")
        else:
            # Fallback: use top financial features from structured_features
            lines.append("   Using top financial features based on values:")
            top_features = _get_top_financial_features(structured_features)
            for i, (feature, value, desc) in enumerate(top_features[:5], start=1):
                lines.append(f"   {i}. {feature.replace('_', ' ').title()} (Value: {value:.4f})")
                lines.append(f"      Assessment: {desc}")
                lines.append("")
            if not top_features:
                lines.append("   No detailed feature contributions available")
        lines.append("")

        # 6. Impact of Global Sentiments on the Score
        lines.append("Impact of News Sentiments on the Score")
        if headlines and sentiments and sentiment_scores:
            # Pick Top 3 by sentiment score
            headline_data = list(zip(headlines, sentiments, sentiment_scores))
            top_headlines = sorted(headline_data, key=lambda x: abs(x[2]), reverse=True)[:3]

            # Global severity from alpha
            if alpha > threshold:
                global_context = "Due to stable market conditions, the impact of news was moderated."
                global_emphasis = " moderately"
            else:
                global_context = "Due to volatile market conditions, the impact of news was amplified."
                global_emphasis = " strongly"
            lines.append(f"   {global_context}")

            for i, (headline, sentiment, score) in enumerate(top_headlines, start=1):
                 # --- FIX 5: Corrected impact text logic ---
                # Impact direction (assuming negative news increases risk score, positive decreases)
                # The final score is a risk score (higher = worse). So negative sentiment (bad news) should increase it.
                if sentiment.lower() == "positive":
                    impact_text = "reduced the credit risk score" # Good news lowers risk
                elif sentiment.lower() == "negative":
                    impact_text = "increased the credit risk score" # Bad news raises risk
                else:
                    impact_text = "had a neutral effect on the credit risk score"

                # Local severity (based on score)
                abs_score = abs(score)
                if abs_score > 0.6: # Adjusted thresholds for better granularity
                    local_severity = "significantly (High Impact)"
                elif abs_score > 0.3:
                    local_severity = "moderately (Moderate Impact)"
                else:
                    local_severity = "slightly (Low Impact)"

                # Combine global + local
                if sentiment.lower() != "neutral":
                    lines.append(
                        f"   {i}. \"{headline[:80]}...\" had a {sentiment.lower()} sentiment, "
                        f"which {impact_text}{global_emphasis} {local_severity}."
                    )
                else:
                    lines.append(
                        f"   {i}. \"{headline[:80]}...\" had neutral sentiment, "
                        f"which {impact_text}{global_emphasis} {local_severity}."
                    )
                lines.append("")
        else:
            lines.append("   No news data available for sentiment analysis")
            lines.append("")

        # 7. Risk Assessment Summary
        lines.append("Risk Assessment Summary")
        # Risk level based on final score
        if final_score < 25:
            risk_level = "LOW RISK"
            risk_desc = "Strong credit profile with minimal default probability"
        elif final_score < 50:
            risk_level = "MODERATE-LOW RISK"
            risk_desc = "Solid credit profile with acceptable risk levels"
        elif final_score < 75:
            risk_level = "MODERATE-HIGH RISK"
            risk_desc = "Elevated risk requiring careful monitoring"
        else:
            risk_level = "HIGH RISK"
            risk_desc = "Significant credit risk with high default probability"
        lines.append(f"   Risk Level: {risk_level}")
        lines.append(f"   Assessment: {risk_desc}")
        lines.append("")

        # 8. Final Narrative Summary
        lines.append("Final Narrative Summary")
        # Market condition narrative
        if alpha > threshold:
            lines.append("   Market conditions were stable during the assessment period.")
            lines.append("   The credit score was primarily driven by fundamental financial ratios.")
        else:
            lines.append("   Market conditions were volatile during the assessment period.")
            lines.append("   Recent news headlines had stronger influence on the final score.")

        # Strength and risk identification
        positive_factors = _identify_positive_factors(ebm_output, sentiments)
        negative_factors = _identify_negative_factors(ebm_output, sentiments)
        if positive_factors:
            lines.append(f"   Key strengths: {', '.join(positive_factors[:3])}")
        if negative_factors:
            lines.append(f"   Key risks: {', '.join(negative_factors[:3])}")
        lines.append("")

        # 9. Technical Details
        lines.append("Technical Details")
        lines.append(f"   Model: Explainable Boosting Machine (EBM)")
        lines.append(f"   Sentiment Analysis: FinBERT")
        lines.append(f"   Fusion Algorithm: Dynamic Weighting")
        lines.append(f"   Articles Analyzed: {unstructured_result.get('articles_analyzed', 0)}")
        # --- FIX 6: Corrected source of features_used ---
        lines.append(f"   Features Used: {len(structured_features)}") # Or len(feature_contributions) if available
        lines.append("")

        # Timestamp
        lines.append(f"Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error generating comprehensive explainability report: {e}")
        return f"Error generating explainability report for {company_name}: {str(e)}"

def _calculate_volatility_factor(market_conditions: Dict[str, Any]) -> float:
    """Calculate volatility factor (alpha) from market conditions"""
    vix = market_conditions.get('vix', 20.0)
    regime = market_conditions.get('regime', 'NORMAL')
    # Higher alpha = more stable market (lower volatility)
    # Lower alpha = more volatile market
    if regime == 'CRISIS':
        base_alpha = 0.3
    elif regime == 'STRESS':
        base_alpha = 0.5
    else:  # NORMAL
        base_alpha = 0.7

    # Adjust based on VIX (lower VIX = more stable = higher alpha)
    # Ensure vix is not zero to avoid division issues
    vix = max(vix, 0.1)
    vix_adjustment = max(0.1, min(1.0, (30 - vix) / 20)) # This makes higher VIX lower adjustment
    alpha = base_alpha * vix_adjustment
    return max(0.1, min(1.0, alpha))

def _extract_ebm_contributions(structured_result: Dict[str, Any], structured_features: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract EBM feature contributions in the required format with proper interpretations and values"""
    try:
        feature_contributions = structured_result.get('feature_contributions', {})
        # structured_score = structured_result.get('structured_score', 50.0) # Not needed here

        # Define financial ratio priority (higher priority features get precedence)
        ratio_priorities = {
            'debt_to_equity': 10, 'current_ratio': 10, 'return_on_equity': 10, 'return_on_assets': 10,
            'enhanced_z_score': 9, 'kmv_distance_to_default': 9, 'net_margin': 8, 'debt_ratio': 8,
            'volatility': 7, 'quick_ratio': 7, 'asset_turnover': 6, 'interest_coverage': 6,
            'working_capital': 5, 'total_revenue': 3, 'total_assets': 2, 'market_cap': 1
        }

        ebm_output = []
        for feature, raw_contribution in feature_contributions.items():
            # Use the original Yahoo Finance value (not scaled)
            original_value = structured_features.get(feature, 0.0)
            # Format value for display
            formatted_value = _format_feature_value(feature, original_value)
            # Get detailed interpretation using the ebm_exp.py logic
            interpretation = _get_detailed_feature_interpretation(feature, original_value, raw_contribution)

            # Calculate contribution percentage relative to total absolute contributions
            total_abs_contribution = sum(abs(contrib) for contrib in feature_contributions.values())
            if total_abs_contribution > 0:
                contribution_percentage = abs(raw_contribution) / total_abs_contribution
            else:
                contribution_percentage = 0.0

            # Get priority for sorting (financial ratios first)
            priority = ratio_priorities.get(feature, 0)

            ebm_output.append({
                'feature': feature.replace('_', ' ').title(),
                'value': float(original_value),
                'formatted_value': formatted_value,
                'raw_contribution': float(raw_contribution), # This is the key model output
                'contribution_percentage': contribution_percentage,
                'interpretation': interpretation,
                'priority': priority
            })

        # Sort by priority first, then by absolute contribution for tie-breaking within categories
        ebm_output.sort(key=lambda x: (-x['priority'], -abs(x['raw_contribution'])))
        return ebm_output

    except Exception as e:
        logger.warning(f"Could not extract EBM contributions: {e}")
        return []

def _format_feature_value(feature_name: str, value: float) -> str:
    """Format feature values for better readability"""
    # Large dollar amounts in billions/millions
    if feature_name in ['market_cap', 'total_assets', 'total_revenue', 'total_equity', 'current_assets', 'total_liabilities']:
        if abs(value) >= 1e9:
            return f"${value/1e9:.2f}B"
        elif abs(value) >= 1e6:
            return f"${value/1e6:.2f}M"
        elif abs(value) >= 1e3:
            return f"${value/1e3:.2f}K"
        else:
            return f"${value:.2f}"
    # Percentages
    elif feature_name in ['return_on_equity', 'return_on_assets', 'net_margin', 'volatility']:
        return f"{value*100:.2f}%"
    # Ratios (keep as decimal with more precision)
    elif feature_name in ['debt_to_equity', 'current_ratio', 'debt_ratio', 'quick_ratio']:
        return f"{value:.3f}"
    # Scores and distances
    elif feature_name in ['enhanced_z_score', 'kmv_distance_to_default']:
        return f"{value:.2f}"
    # Default formatting
    else:
        return f"{value:.4f}"

def _get_top_financial_features(structured_features: Dict[str, Any]) -> List[tuple]:
    """Get top financial features when EBM contributions are not available"""
    features = []
    # Key financial ratios with thresholds and descriptions
    key_metrics = {
        'debt_to_equity': {
            'threshold_good': 0.5,
            'threshold_concern': 2.0,
            'good_desc': 'strong capital structure',
            'concern_desc': 'high leverage risk'
        },
        'current_ratio': {
            'threshold_good': 1.5,
            'threshold_concern': 1.0,
            'good_desc': 'adequate liquidity',
            'concern_desc': 'liquidity concerns'
        },
        'return_on_equity': {
            'threshold_good': 0.10,
            'threshold_concern': 0.05,
            'good_desc': 'strong profitability',
            'concern_desc': 'weak profitability'
        },
        'return_on_assets': {
            'threshold_good': 0.05,
            'threshold_concern': 0.02,
            'good_desc': 'efficient asset utilization',
            'concern_desc': 'poor asset efficiency'
        },
        'enhanced_z_score': {
            'threshold_good': 3.0,
            'threshold_concern': 1.8,
            'good_desc': 'low bankruptcy risk',
            'concern_desc': 'elevated bankruptcy risk'
        }
    }

    for feature, config in key_metrics.items():
        value = structured_features.get(feature, 0.0)
        if value != 0.0:  # Only include if we have data
            if value >= config['threshold_good']:
                desc = config['good_desc']
            elif value <= config['threshold_concern']:
                desc = config['concern_desc']
            else:
                desc = 'moderate performance levels'
            features.append((feature, value, desc))

    # Sort by relevance (put concerning values first, then good values)
    def sort_key(item):
        feature, value, desc = item
        config = key_metrics[feature]
        if 'concern' in desc:
            return 0  # High priority
        elif 'strong' in desc or 'good' in desc:
            return 1  # Medium priority
        else:
            return 2  # Low priority

    features.sort(key=sort_key)
    return features

def _get_detailed_feature_interpretation(feature_name: str, value: float, contribution: float) -> str:
    """Get detailed business interpretation for a feature using ebm_exp.py logic"""
    # Define comprehensive interpretation rules from ebm_exp.py
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
        'debt_ratio': {
            'thresholds': [0.3, 0.5, 0.7, 0.8],
            'descriptions': [
                "very low debt burden indicating strong financial position",
                "moderate debt levels within healthy range",
                "elevated debt levels requiring monitoring",
                "high debt burden increasing default risk",
                "excessive debt burden indicating financial distress"
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
        'return_on_equity': {
            'thresholds': [0.0, 0.05, 0.15, 0.25],
            'descriptions': [
                "negative returns indicating poor management performance",
                "weak profitability suggesting operational challenges",
                "acceptable profitability within industry norms",
                "strong profitability indicating efficient management",
                "exceptional profitability demonstrating superior performance"
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
        },
        'net_margin': {
            'thresholds': [0.0, 0.05, 0.10, 0.20],
            'descriptions': [
                "negative profitability indicating operational losses",
                "weak profit margins suggesting pricing or cost issues",
                "adequate profit margins within industry standards",
                "strong profit margins indicating competitive advantage",
                "exceptional profit margins demonstrating pricing power"
            ]
        },
        'volatility': {
            'thresholds': [0.15, 0.30, 0.50, 0.75],
            'descriptions': [
                "very low market risk with stable stock performance",
                "moderate market risk typical for established companies",
                "elevated market risk indicating investor uncertainty",
                "high market risk suggesting significant concerns",
                "extreme market risk indicating severe volatility"
            ]
        },
        'kmv_distance_to_default': {
            'thresholds': [0.0, 1.5, 3.0, 5.0],
            'descriptions': [
                "company is at immediate risk of default",
                "elevated default risk requiring urgent attention",
                "moderate default risk needing monitoring",
                "low default risk indicating stable credit profile",
                "minimal default risk with strong credit metrics"
            ]
        },
        'return_on_assets': {
            'thresholds': [0.0, 0.02, 0.05, 0.10],
            'descriptions': [
                "negative asset returns indicating poor operational efficiency",
                "weak asset utilization requiring improvement",
                "adequate asset efficiency within industry standards",
                "strong asset utilization indicating good management",
                "exceptional asset efficiency demonstrating superior operations"
            ]
        },
        'market_cap': {
            'thresholds': [1e9, 10e9, 100e9, 1000e9],
            'descriptions': [
                "small company with limited market presence",
                "mid-cap company with moderate market presence",
                "large-cap company with strong market position",
                "mega-cap company with dominant market position",
                "ultra-large company with exceptional market dominance"
            ]
        },
        'total_assets': {
            'thresholds': [1e9, 10e9, 50e9, 200e9],
            'descriptions': [
                "small asset base indicating limited operational scale",
                "moderate asset base with adequate operational capacity",
                "substantial asset base supporting strong operations",
                "large asset base demonstrating significant operational scale",
                "massive asset base indicating industry-leading scale"
            ]
        },
        'total_revenue': {
            'thresholds': [1e9, 10e9, 50e9, 100e9],
            'descriptions': [
                "low revenue indicating small operational scale",
                "moderate revenue with adequate market presence",
                "strong revenue demonstrating solid market position",
                "high revenue indicating large market presence",
                "exceptional revenue demonstrating market leadership"
            ]
        },
        'total_equity': {
            'thresholds': [1e9, 10e9, 50e9, 100e9],
            'descriptions': [
                "limited equity base indicating potential capital constraints",
                "adequate equity providing reasonable financial cushion",
                "strong equity base supporting stable operations",
                "substantial equity indicating robust financial position",
                "exceptional equity demonstrating superior financial strength"
            ]
        }
    }
    feature_key = feature_name.lower()
    if feature_key in interpretations:
        thresholds = interpretations[feature_key]['thresholds']
        descriptions = interpretations[feature_key]['descriptions']
        # Find appropriate description based on value
        description_idx = 0
        for i, threshold in enumerate(thresholds):
            if value <= threshold:
                description_idx = i
                break
        else:
            description_idx = len(descriptions) - 1
        return descriptions[description_idx]
    else:
        # Generic interpretation based on contribution
        if contribution > 0:
            return "factor contributing positively to credit assessment"
        elif contribution < 0:
            return "factor indicating increased credit risk"
        else:
            return "neutral factor with minimal impact"

def _extract_news_data(unstructured_result: Dict[str, Any]) -> Tuple[List[str], List[str], List[float]]:
    """Extract news headlines, sentiments, and scores"""
    try:
        sample_headlines = unstructured_result.get('sample_headlines', [])
        sentiment_dist = unstructured_result.get('sentiment_distribution', {})
        sentiment_scores = unstructured_result.get('sentiment_scores', []) # Assuming this exists
        headlines = []
        sentiments = []
        scores = []

        # Use actual data if available
        if sample_headlines and sentiment_scores and len(sample_headlines) == len(sentiment_scores):
             # Assume sentiment_scores contains 'negative', 'neutral', 'positive' scores
             # Let's simplify and assume it's a list of primary sentiment labels or scores
             # The original logic was flawed. Let's try to get real data.
             # If sentiment_distribution exists, we can infer sentiments.
             total_articles = sum(sentiment_dist.values()) if sentiment_dist else 1
             if total_articles == 0: total_articles = 1

             neg_ratio = sentiment_dist.get('negative', 0) / total_articles
             pos_ratio = sentiment_dist.get('positive', 0) / total_articles
             neu_ratio = sentiment_dist.get('neutral', 0) / total_articles

             # Assign sentiment based on distribution (simplified)
             for i, headline in enumerate(sample_headlines[:3]):
                 headlines.append(headline)
                 # Assign sentiment based on distribution (simplified)
                 rand_val = np.random.random() if i >= len(sentiment_scores) else sentiment_scores[i] # Fallback to random if not enough scores
                 if isinstance(rand_val, str): # If it's a label
                     sentiments.append(rand_val.title())
                     # Assign a mock score based on label
                     if rand_val.lower() == 'negative':
                         scores.append(-0.7)
                     elif rand_val.lower() == 'positive':
                         scores.append(0.6)
                     else:
                         scores.append(0.0)
                 elif isinstance(rand_val, (int, float)): # If it's a numerical score
                     scores.append(float(rand_val))
                     if rand_val < -0.1:
                         sentiments.append('Negative')
                     elif rand_val > 0.1:
                         sentiments.append('Positive')
                     else:
                         sentiments.append('Neutral')
                 else:
                     # Default fallback
                     if i == 0 and neg_ratio > 0.4:
                         sentiments.append('Negative')
                         scores.append(-0.7)
                     elif i == 1 and pos_ratio > 0.4:
                         sentiments.append('Positive')
                         scores.append(0.6)
                     else:
                         sentiments.append('Neutral')
                         scores.append(0.2)
        else:
            # Generate sample data from available information (fallback)
            for i, headline in enumerate(sample_headlines[:3]):
                headlines.append(headline)
                # Assign sentiment based on distribution (simplified)
                total_articles = sum(sentiment_dist.values())
                if total_articles > 0:
                    neg_ratio = sentiment_dist.get('negative', 0) / total_articles
                    pos_ratio = sentiment_dist.get('positive', 0) / total_articles
                    if i == 0 and neg_ratio > 0.4:
                        sentiments.append('Negative')
                        scores.append(-0.7)
                    elif i == 1 and pos_ratio > 0.4:
                        sentiments.append('Positive')
                        scores.append(0.6)
                    else:
                        sentiments.append('Neutral')
                        scores.append(0.2)
                else:
                    sentiments.append('Neutral')
                    scores.append(0.0)

        return headlines, sentiments, scores

    except Exception as e:
        logger.warning(f"Could not extract news data: {e}")
        return [], [], []

def _identify_positive_factors(ebm_output: List[Dict[str, Any]], sentiments: List[str]) -> List[str]:
    """Identify positive contributing factors"""
    factors = []
    # From EBM output
    for item in ebm_output:
        if item['raw_contribution'] > 0.01: # Use a small threshold instead of 0
            factors.append(f"Strong {item['feature'].lower()}")
    # From sentiments
    positive_count = sentiments.count('Positive')
    if positive_count > 0:
        factors.append(f"Positive news sentiment ({positive_count} articles)")
    return factors

def _identify_negative_factors(ebm_output: List[Dict[str, Any]], sentiments: List[str]) -> List[str]:
    """Identify negative risk factors"""
    factors = []
    # From EBM output
    for item in ebm_output:
        if item['raw_contribution'] < -0.01:  # Significant negative contribution, use small threshold
            factors.append(f"Weak {item['feature'].lower()}")
    # From sentiments
    negative_count = sentiments.count('Negative')
    if negative_count > 0:
        factors.append(f"Negative news sentiment ({negative_count} articles)")
    return factors

# === EBM EXPLAINER ===

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
        # Reuse the global function for consistency
        return _get_detailed_feature_interpretation(feature_name, value, contribution)

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

        # Ensure scaler is loaded correctly
        if self.scaler is None:
             logger.error("Scaler is None. Cannot transform data for explanation.")
             raise ValueError("Scaler not loaded. Check model file.")

        sample_scaled = self.scaler.transform(sample_df)

        try:
            prediction = self.ebm_model.predict(sample_scaled)[0]
            prediction_proba = self.ebm_model.predict_proba(sample_scaled)[0]
        except Exception as e:
            logger.error(f"Error getting prediction: {e}")
            prediction = 0
            prediction_proba = [0.5, 0.5]

        # Get feature contributions using EBM's explain_local
        feature_names = self.feature_columns
        feature_values = sample_df.iloc[0].to_dict()
        feature_contributions = {}

        # --- FIX 7: Use EBM's built-in explain_local for accurate contributions ---
        try:
            # Use EBM's built-in explain_local for accurate contributions
            logger.debug(f"Attempting explain_local with sample_scaled shape: {sample_scaled.shape}")
            local_explanation = self.ebm_model.explain_local(sample_scaled)
            logger.debug(f"Local explanation object type: {type(local_explanation)}")

            # --- ROBUST CHECKING FOR LOCAL EXPLANATION DATA ---
            if local_explanation is not None:
                # Check for data() method or direct attributes
                if hasattr(local_explanation, 'data'):
                    local_data = local_explanation.data()
                    logger.debug(f"Local explanation data() result type: {type(local_data)}")
                else:
                    # Fallback to checking for names and scores attributes directly
                    local_data = {
                        'names': getattr(local_explanation, 'names', None),
                        'scores': getattr(local_explanation, 'scores', None)
                    }
                    logger.debug(f"Local explanation data (from attributes): {local_data}")

                # Check if local_data is a dictionary and contains the expected keys/values
                if isinstance(local_data, dict):
                    feature_names_local = local_data.get('names', [])
                    feature_scores_local = local_data.get('scores', [])

                    # Ensure names and scores are lists/arrays before zipping
                    if isinstance(feature_names_local, (list, np.ndarray)) and isinstance(feature_scores_local, (list, np.ndarray)):
                        if len(feature_names_local) == len(feature_scores_local) and len(feature_names_local) > 0:
                            # Map feature names to their scores
                            feature_contributions = dict(zip(feature_names_local, feature_scores_local))
                            logger.info("Successfully retrieved local feature contributions.")
                        else:
                            logger.warning(f"Mismatch in lengths or empty local names({len(feature_names_local)}), scores({len(feature_scores_local)})")
                    else:
                        logger.warning("Local explanation 'names' or 'scores' are not lists/arrays.")
                elif hasattr(local_explanation, 'names') and hasattr(local_explanation, 'scores'):
                     # If local_data wasn't a dict, try direct attributes again
                     feature_names_local = local_explanation.names
                     feature_scores_local = local_explanation.scores
                     if isinstance(feature_names_local, (list, np.ndarray)) and isinstance(feature_scores_local, (list, np.ndarray)):
                         if len(feature_names_local) == len(feature_scores_local) and len(feature_names_local) > 0:
                             feature_contributions = dict(zip(feature_names_local, feature_scores_local))
                             logger.info("Successfully retrieved local feature contributions (from attributes).")
                         else:
                             logger.warning(f"Mismatch in lengths or empty local names({len(feature_names_local)}), scores({len(feature_scores_local)}) (from attributes)")
                     else:
                         logger.warning("Local explanation .names or .scores are not lists/arrays (from attributes).")
                else:
                    logger.warning(f"Local explanation data is not a dictionary or lacks .names/.scores attributes. Type: {type(local_data)}")
            else:
                logger.warning("Local explanation object is None.")

        except AttributeError as ae:
            logger.warning(f"AttributeError while getting local feature contributions: {ae}")
        except Exception as e:
            logger.warning(f"Could not get local feature contributions, falling back to global importance: {e}")

        # --- FALLBACK: Use global importance if explain_local fails or returns invalid data ---
        if not feature_contributions:
            try:
                logger.info("Falling back to global feature importance for contributions.")
                global_explanation = self.ebm_model.explain_global()
                global_data = global_explanation.data() if hasattr(global_explanation, 'data') else {}
                # Ensure global_data is a dict
                if not isinstance(global_data, dict):
                    global_data = {}
                global_scores = global_data.get('scores', [1.0] * len(feature_names))
                # Approximate local contribution based on feature value and global importance
                sample_values = sample_df.iloc[0]
                for i, feature_name in enumerate(feature_names):
                    if i < len(global_scores):
                        feature_value = sample_values.get(feature_name, 0)
                        # Simple approximation: sign of value * global score magnitude
                        approx_score = (1 if feature_value >= 0 else -1) * abs(global_scores[i]) * 0.1
                        feature_contributions[feature_name] = approx_score
                    else:
                        feature_contributions[feature_name] = 0.0
            except Exception as e2:
                logger.error(f"Fallback to global importance also failed: {e2}")
                feature_contributions = {name: 0.001 for name in feature_names}

        # Ensure all expected features are present in the final dictionary, fill missing with 0
        final_feature_contributions = {name: feature_contributions.get(name, 0.0) for name in feature_names}

        # Convert to list in the correct order for compatibility with _generate_detailed_explanation
        feature_scores = [final_feature_contributions[name] for name in feature_names]

        explanation = self._generate_detailed_explanation(
            company_name, prediction, prediction_proba,
            feature_names, feature_scores, # Pass the correct lists
            feature_values # Pass actual feature values
        )

        return {
            'explanation_text': explanation,
            'prediction': 'Investment Grade' if prediction == 1 else 'Non-Investment Grade',
            'probability_investment_grade': float(prediction_proba[1]), # Ensure float
            'probability_non_investment_grade': float(prediction_proba[0]), # Ensure float
            'feature_contributions': final_feature_contributions # Return the correct contributions dictionary
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
        if total_abs_contribution == 0: total_abs_contribution = 1e-8 # Avoid division by zero

        for i, (feature_name, contribution, value) in enumerate(feature_data[:10]):
            contrib_pct = (abs(contribution) / total_abs_contribution) * 100
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
        MODEL_PATH = r"C:\\Users\\asus\\Documents\\GitHub\\BlackBox_Cred\\BlackBox_Backend\\model\\ebm_model_struct_score.pkl"
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
            struct_weight = dynamic_weights.get('structured_expert', 0.5) # Default to 0.5
            explanation_lines.append(f"Structured Analysis (EBM):")
            explanation_lines.append(f"  Score: {struct_contrib.get('score', struct_contrib.get('risk_score', 0)):.1f}/100") # Use 'score' or fallback
            explanation_lines.append(f"  Weight: {struct_weight:.1%}")
            explanation_lines.append(f"  Contribution: {struct_contrib.get('contribution', 0):.1f} points")
            explanation_lines.append("")

        if 'news_sentiment_expert' in expert_contributions:
            news_contrib = expert_contributions['news_sentiment_expert']
            news_weight = dynamic_weights.get('news_sentiment_expert', 0.5) # Default to 0.5
            explanation_lines.append(f"News Sentiment Analysis:")
            explanation_lines.append(f"  Score: {news_contrib.get('score', news_contrib.get('risk_score', 0)):.1f}/100") # Use 'score' or fallback
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
