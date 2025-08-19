"""
global_fusion.py

Implements the Global Fusion system that combines multiple expert assessments
using dynamic weighting based on macroeconomic conditions, market volatility, and expert confidence.
Enhanced version based on Multi-Agent Explainable Adaptive STructured-Textual Risk Oracle (MAESTRO).
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ExpertAssessment:
    """Data class for individual expert assessment results"""
    expert_name: str
    risk_score: float  # 0-100 scale
    features_used: List[str]
    computation_time: float


@dataclass 
class MarketConditions:
    """Data class for market condition indicators"""
    vix: float
    unemployment_rate: float
    credit_spread: float
    yield_curve_slope: float
    economic_stress_index: float
    financial_conditions_index: float
    regime: str  # 'NORMAL', 'STRESS', 'CRISIS'


class MAESTROGlobalFusion:
    """
    Advanced global fusion system that combines multiple expert assessments
    using dynamic weighting based on market conditions and expert reliability.
    """
    
    def __init__(self):
        self.expert_base_weights = {
            'structured_expert': 0.60,          # EBM structured data analysis
            'news_sentiment_expert': 0.40       # News sentiment & risk detection
        }
        
        self.regime_adjustments = {
            'NORMAL': {'volatility_factor': 1.0, 'text_boost': 1.0},
            'STRESS': {'volatility_factor': 1.3, 'text_boost': 1.4},
            'CRISIS': {'volatility_factor': 1.6, 'text_boost': 1.8}
        }
    
    def compute_dynamic_weights(self, expert_assessments: Dict[str, ExpertAssessment], 
                              market_conditions: MarketConditions) -> Dict[str, float]:
        """
        Compute dynamic expert weights based on market conditions and expert performance.
        
        Args:
            expert_assessments: Dictionary of expert assessment results
            market_conditions: Current market condition indicators
            
        Returns:
            Dictionary of normalized expert weights
        """
        
        # Start with base weights
        dynamic_weights = self.expert_base_weights.copy()
        
        # Adjust based on market regime
        regime_adj = self.regime_adjustments.get(market_conditions.regime, 
                                               self.regime_adjustments['NORMAL'])
        
        # 1. VIX-based adjustments (high volatility = more weight on news expert)
        if market_conditions.vix > 30:  # High volatility
            dynamic_weights['news_sentiment_expert'] *= 1.4
            dynamic_weights['structured_expert'] *= 0.9
        elif market_conditions.vix < 15:  # Low volatility
            dynamic_weights['news_sentiment_expert'] *= 0.8
            dynamic_weights['structured_expert'] *= 1.1
        
        # 2. Economic stress adjustments
        stress_factor = market_conditions.economic_stress_index / 100
        dynamic_weights['news_sentiment_expert'] *= (1 + stress_factor * 0.6)
        
        # 3. Credit market stress adjustments
        if market_conditions.credit_spread > 4.0:  # High credit stress
            dynamic_weights['news_sentiment_expert'] *= 1.3
            dynamic_weights['structured_expert'] *= 1.1
        
        # 4. Confidence-based adjustments (SKIPPED for structured data)
        
        # 5. Agreement-based adjustments
        expert_agreement = self._calculate_expert_agreement(expert_assessments)
        if expert_agreement < 0.6:  # Low agreement indicates uncertainty
            # Reduce weights for extreme assessments, boost consensus
            dynamic_weights = self._apply_consensus_adjustment(dynamic_weights, expert_assessments)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(dynamic_weights.values())
        normalized_weights = {k: v/total_weight for k, v in dynamic_weights.items()}
        
        return normalized_weights
    

    # Confidence-based adjustments are not used for structured data
    
    def _calculate_expert_agreement(self, expert_assessments: Dict[str, ExpertAssessment]) -> float:
        """Calculate agreement level among expert assessments"""
        risk_scores = [assessment.risk_score for assessment in expert_assessments.values()]
        
        if len(risk_scores) < 2:
            return 1.0
        
        # Calculate coefficient of variation (lower = higher agreement)
        mean_score = np.mean(risk_scores)
        std_score = np.std(risk_scores)
        
        if mean_score == 0:
            return 1.0
        
        cv = std_score / mean_score
        agreement = max(0, 1 - cv)  # Convert to agreement measure
        
        return agreement
    
    def _apply_consensus_adjustment(self, weights: Dict[str, float], 
                                  expert_assessments: Dict[str, ExpertAssessment]) -> Dict[str, float]:
        """Apply consensus-based adjustments when experts disagree significantly"""
        
        risk_scores = {name: assessment.risk_score for name, assessment in expert_assessments.items()}
        median_score = np.median(list(risk_scores.values()))
        
        adjusted_weights = weights.copy()
        
        for expert_name, score in risk_scores.items():
            if expert_name in adjusted_weights:
                # Penalize experts that deviate significantly from median
                deviation = abs(score - median_score) / 100  # Normalize to 0-1 scale
                penalty = max(0.7, 1 - deviation * 0.6)  # Max 30% penalty
                adjusted_weights[expert_name] *= penalty
        
        return adjusted_weights
    
    def compute_enhanced_fusion_score(self, expert_assessments: Dict[str, ExpertAssessment],
                                    market_conditions: MarketConditions) -> Dict:
        """
        Compute enhanced fusion score combining both expert risk scores with dynamic weighting.
        Focus purely on score fusion without explanations.
        
        Args:
            expert_assessments: Dictionary of expert assessment results
            market_conditions: Current market condition indicators
            
        Returns:
            Fusion result with final score and weights
        """
        
        # Calculate dynamic weights
        dynamic_weights = self.compute_dynamic_weights(expert_assessments, market_conditions)
        
        # Calculate weighted fusion score
        fused_score = 0.0
        expert_contributions = {}
        
        for expert_name, assessment in expert_assessments.items():
            weight = dynamic_weights.get(expert_name, 0.0)
            contribution = weight * assessment.risk_score
            fused_score += contribution
            expert_contributions[expert_name] = {
                'risk_score': assessment.risk_score,
                'weight': weight,
                'contribution': contribution
            }
        
        # Expert agreement measure
        expert_agreement = self._calculate_expert_agreement(expert_assessments)
        
        # Market regime impact
        regime_impact = self._calculate_regime_impact(market_conditions, fused_score)
        
        # Final adjusted score
        final_score = fused_score + regime_impact
        final_score = max(0, min(100, final_score))  # Bound between 0-100
        
        return {
            'fused_risk_score': final_score,
            'base_fused_score': fused_score,
            'regime_adjustment': regime_impact,
            'expert_agreement': expert_agreement,
            'market_regime': market_conditions.regime,
            'expert_contributions': expert_contributions,
            'dynamic_weights': dynamic_weights,
            'market_conditions': {
                'vix': market_conditions.vix,
                'economic_stress_index': market_conditions.economic_stress_index,
                'credit_spread': market_conditions.credit_spread,
                'yield_curve_slope': market_conditions.yield_curve_slope
            },
            'fusion_metadata': {
                'fusion_time': datetime.now().isoformat(),
                'number_of_experts': len(expert_assessments),
                'fusion_method': 'MAESTRO Dynamic Weighted Fusion'
            }
        }
    
    def _calculate_regime_impact(self, market_conditions: MarketConditions, base_score: float) -> float:
        """Calculate market regime impact on final score"""
        
        if market_conditions.regime == 'CRISIS':
            # During crisis, increase risk scores (add up to 15 points)
            stress_multiplier = market_conditions.economic_stress_index / 100
            return min(15, stress_multiplier * 20)
        
        elif market_conditions.regime == 'STRESS':
            # During stress, moderate increase (add up to 8 points)
            stress_multiplier = market_conditions.economic_stress_index / 100
            return min(8, stress_multiplier * 12)
        
        else:  # NORMAL
            # During normal times, slight adjustment based on conditions
            if market_conditions.vix > 25:
                return min(3, (market_conditions.vix - 25) / 5)
            return 0


# Simple fusion example with focus on score combination only
if __name__ == "__main__":
    # Import the news analysis function
    from news_unstructured_score import get_news_risk_assessment
    from ebm_training import FEATURE_COLUMNS  # Assuming this contains your EBM features
    
    # Example company for analysis
    company_name = "Apple"
    
    print("=" * 60)
    print(f"SIMPLIFIED GLOBAL FUSION FOR {company_name}")
    print("=" * 60)
    
    # 1. Get structured expert assessment (EBM)
    structured_assessment = ExpertAssessment(
        expert_name='Structured Expert (EBM)',
        risk_score=25.0,  # Low risk from EBM analysis
        features_used=FEATURE_COLUMNS,  # Your specified features only
        computation_time=0.15
    )
    
    # 2. Get news sentiment assessment
    print(f"🔍 Analyzing news sentiment for {company_name}...")
    news_result = get_news_risk_assessment(company_name, days_back=7, max_articles=20)
    
    news_assessment = ExpertAssessment(
        expert_name='News Sentiment Expert',
        risk_score=news_result['risk_score'],
        features_used=news_result['features_used'],
        computation_time=news_result['computation_time']
    )
    
    # Combine expert assessments
    expert_assessments = {
        'structured_expert': structured_assessment,
        'news_sentiment_expert': news_assessment
    }
    
    # Create sample market conditions
    market_conditions = MarketConditions(
        vix=22.5,  # Current VIX level
        unemployment_rate=3.8,
        credit_spread=2.8,
        yield_curve_slope=1.2,
        economic_stress_index=35.0,  # Low stress
        financial_conditions_index=45.0,
        regime='NORMAL'
    )
    
    # Initialize MAESTRO fusion system
    fusion_system = MAESTROGlobalFusion()
    
    # Compute fusion (without explanations)
    result = fusion_system.compute_enhanced_fusion_score(expert_assessments, market_conditions)
    
    # Display core results
    print(f"\n📊 FUSION RESULTS:")
    print(f"Final Risk Score: {result['fused_risk_score']:.1f}%")
    print(f"Base Score (before regime adj): {result['base_fused_score']:.1f}%")
    print(f"Regime Adjustment: +{result['regime_adjustment']:.1f} points")
    print(f"Expert Agreement: {result['expert_agreement']:.1%}")
    print(f"Market Regime: {result['market_regime']}")
    
    print(f"\n🎯 EXPERT CONTRIBUTIONS:")
    for expert, contrib in result['expert_contributions'].items():
        print(f"  {expert.replace('_', ' ').title()}:")
        print(f"    Score: {contrib['risk_score']:.1f}%")
        print(f"    Weight: {contrib['weight']:.2f}")
        print(f"    Contribution: {contrib['contribution']:.1f} points")
    
    print(f"\n⚖️ DYNAMIC WEIGHTS:")
    for expert, weight in result['dynamic_weights'].items():
        print(f"  {expert.replace('_', ' ').title()}: {weight:.1%}")
    
    print(f"\n📈 MARKET CONDITIONS:")
    mc = result['market_conditions']
    print(f"  VIX: {mc['vix']:.1f}")
    print(f"  Economic Stress: {mc['economic_stress_index']:.1f}/100")
    print(f"  Credit Spread: {mc['credit_spread']:.2f}%")
    
    print(f"\n🔄 FUSION PROCESS:")
    print(f"1. Structured Expert: {structured_assessment.risk_score:.1f}% × {result['dynamic_weights']['structured_expert']:.2f} = {result['expert_contributions']['structured_expert']['contribution']:.1f}")
    print(f"2. News Expert: {news_assessment.risk_score:.1f}% × {result['dynamic_weights']['news_sentiment_expert']:.2f} = {result['expert_contributions']['news_sentiment_expert']['contribution']:.1f}")
    print(f"3. Base Fusion: {result['base_fused_score']:.1f}%")
    print(f"4. Market Adjustment: +{result['regime_adjustment']:.1f}")
    print(f"5. Final Score: {result['fused_risk_score']:.1f}%")
    
    print("=" * 60)
