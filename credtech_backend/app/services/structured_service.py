"""
Structured model service implementing KMV, Z-Score, and EBM models.
"""
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from sqlalchemy.orm import Session
from scipy.optimize import fsolve
from scipy.stats import norm
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
# import shap # Removed SHAP import
from app.core.config import settings
from app.db.session import get_db_session
from app.db.models import Company, StructuredData, CreditScoreHistory

# S&P 500 Sector-Specific Benchmarks 
# Sources: Bloomberg, S&P Capital IQ, FactSet, Federal Reserve Data (2023-2024 data)
SP500_SECTOR_BENCHMARKS = {
    'Information Technology': {
        'Current Ratio': {'excellent': 2.5, 'good': 2.0, 'fair': 1.5, 'poor': 1.2},
        'Debt/Equity Ratio': {'excellent': 0.2, 'good': 0.4, 'fair': 0.7, 'poor': 1.2},
        'ROE - Return On Equity': {'excellent': 25, 'good': 18, 'fair': 12, 'poor': 6},
        'ROA - Return On Assets': {'excellent': 15, 'good': 10, 'fair': 6, 'poor': 3},
        'Operating Margin': {'excellent': 25, 'good': 18, 'fair': 12, 'poor': 6},
        'Net Profit Margin': {'excellent': 20, 'good': 15, 'fair': 10, 'poor': 5},
        'Asset Turnover': {'excellent': 1.2, 'good': 0.9, 'fair': 0.7, 'poor': 0.5},
        'altman_z_score': {'excellent': 5.0, 'good': 4.0, 'fair': 3.0, 'poor': 2.0},
        'kmv_distance_to_default': {'excellent': 5.0, 'good': 3.5, 'fair': 2.5, 'poor': 1.5}
    },
    'Communication Services': {
        'Current Ratio': {'excellent': 2.0, 'good': 1.5, 'fair': 1.2, 'poor': 0.9},
        'Debt/Equity Ratio': {'excellent': 0.4, 'good': 0.8, 'fair': 1.3, 'poor': 2.0},
        'ROE - Return On Equity': {'excellent': 20, 'good': 14, 'fair': 9, 'poor': 4},
        'ROA - Return On Assets': {'excellent': 8, 'good': 5, 'fair': 3, 'poor': 1},
        'Operating Margin': {'excellent': 20, 'good': 14, 'fair': 8, 'poor': 3},
        'Net Profit Margin': {'excellent': 15, 'good': 10, 'fair': 6, 'poor': 2},
        'Asset Turnover': {'excellent': 0.8, 'good': 0.6, 'fair': 0.4, 'poor': 0.25},
        'altman_z_score': {'excellent': 4.0, 'good': 3.2, 'fair': 2.4, 'poor': 1.6},
        'kmv_distance_to_default': {'excellent': 4.0, 'good': 2.8, 'fair': 2.0, 'poor': 1.2}
    },
    'Consumer Discretionary': {
        'Current Ratio': {'excellent': 1.8, 'good': 1.4, 'fair': 1.1, 'poor': 0.9},
        'Debt/Equity Ratio': {'excellent': 0.3, 'good': 0.6, 'fair': 1.0, 'poor': 1.8},
        'ROE - Return On Equity': {'excellent': 20, 'good': 14, 'fair': 9, 'poor': 4},
        'ROA - Return On Assets': {'excellent': 8, 'good': 5, 'fair': 3, 'poor': 1},
        'Operating Margin': {'excellent': 12, 'good': 8, 'fair': 5, 'poor': 2},
        'Net Profit Margin': {'excellent': 8, 'good': 5, 'fair': 3, 'poor': 1},
        'Asset Turnover': {'excellent': 1.8, 'good': 1.3, 'fair': 1.0, 'poor': 0.7},
        'altman_z_score': {'excellent': 3.8, 'good': 3.0, 'fair': 2.2, 'poor': 1.5},
        'kmv_distance_to_default': {'excellent': 3.8, 'good': 2.6, 'fair': 1.8, 'poor': 1.1}
    },
    'Consumer Staples': {
        'Current Ratio': {'excellent': 1.6, 'good': 1.3, 'fair': 1.0, 'poor': 0.8},
        'Debt/Equity Ratio': {'excellent': 0.8, 'good': 1.2, 'fair': 1.8, 'poor': 2.8},
        'ROE - Return On Equity': {'excellent': 18, 'good': 14, 'fair': 10, 'poor': 6},
        'ROA - Return On Assets': {'excellent': 8, 'good': 6, 'fair': 4, 'poor': 2},
        'Operating Margin': {'excellent': 10, 'good': 7, 'fair': 5, 'poor': 3},
        'Net Profit Margin': {'excellent': 7, 'good': 5, 'fair': 3.5, 'poor': 2},
        'Asset Turnover': {'excellent': 1.4, 'good': 1.0, 'fair': 0.8, 'poor': 0.6},
        'altman_z_score': {'excellent': 3.5, 'good': 2.8, 'fair': 2.1, 'poor': 1.5},
        'kmv_distance_to_default': {'excellent': 3.5, 'good': 2.5, 'fair': 1.8, 'poor': 1.2}
    },
    'Energy': {
        'Current Ratio': {'excellent': 2.2, 'good': 1.6, 'fair': 1.2, 'poor': 0.9},
        'Debt/Equity Ratio': {'excellent': 0.4, 'good': 0.8, 'fair': 1.5, 'poor': 2.5},
        'ROE - Return On Equity': {'excellent': 15, 'good': 10, 'fair': 5, 'poor': 0},  # Can be near zero in downturns
        'ROA - Return On Assets': {'excellent': 8, 'good': 5, 'fair': 2, 'poor': 0},
        'Operating Margin': {'excellent': 20, 'good': 12, 'fair': 5, 'poor': -2},  # Can go negative
        'Net Profit Margin': {'excellent': 15, 'good': 8, 'fair': 3, 'poor': -2},
        'Asset Turnover': {'excellent': 1.0, 'good': 0.7, 'fair': 0.5, 'poor': 0.3},
        'altman_z_score': {'excellent': 3.2, 'good': 2.5, 'fair': 1.8, 'poor': 1.2},
        'kmv_distance_to_default': {'excellent': 3.2, 'good': 2.2, 'fair': 1.5, 'poor': 0.9}
    },
    'Financials': {
        'Current Ratio': {'excellent': 1.15, 'good': 1.08, 'fair': 1.02, 'poor': 0.95},  # Less relevant for banks
        'Debt/Equity Ratio': {'excellent': 6.0, 'good': 9.0, 'fair': 13.0, 'poor': 18.0},  # Leverage is normal
        'ROE - Return On Equity': {'excellent': 15, 'good': 12, 'fair': 9, 'poor': 6},
        'ROA - Return On Assets': {'excellent': 1.3, 'good': 1.0, 'fair': 0.7, 'poor': 0.4},
        'Operating Margin': {'excellent': 35, 'good': 28, 'fair': 20, 'poor': 12},  # Net Interest Margin proxy
        'Net Profit Margin': {'excellent': 25, 'good': 20, 'fair': 15, 'poor': 10},
        'Asset Turnover': {'excellent': 0.08, 'good': 0.06, 'fair': 0.04, 'poor': 0.025},
        'altman_z_score': {'excellent': 2.8, 'good': 2.2, 'fair': 1.7, 'poor': 1.2},  # Lower for financials
        'kmv_distance_to_default': {'excellent': 2.8, 'good': 2.0, 'fair': 1.4, 'poor': 0.9}
    },
    'Healthcare': {
        'Current Ratio': {'excellent': 2.5, 'good': 1.8, 'fair': 1.4, 'poor': 1.0},
        'Debt/Equity Ratio': {'excellent': 0.3, 'good': 0.5, 'fair': 0.8, 'poor': 1.4},
        'ROE - Return On Equity': {'excellent': 18, 'good': 13, 'fair': 9, 'poor': 5},
        'ROA - Return On Assets': {'excellent': 12, 'good': 8, 'fair': 5, 'poor': 2},
        'Operating Margin': {'excellent': 20, 'good': 14, 'fair': 9, 'poor': 4},  # Varies by subsector
        'Net Profit Margin': {'excellent': 15, 'good': 11, 'fair': 7, 'poor': 3},
        'Asset Turnover': {'excellent': 0.9, 'good': 0.6, 'fair': 0.4, 'poor': 0.25},
        'altman_z_score': {'excellent': 4.2, 'good': 3.3, 'fair': 2.5, 'poor': 1.7},
        'kmv_distance_to_default': {'excellent': 4.2, 'good': 3.0, 'fair': 2.1, 'poor': 1.4}
    },
    'Industrials': {
        'Current Ratio': {'excellent': 2.0, 'good': 1.5, 'fair': 1.2, 'poor': 0.9},
        'Debt/Equity Ratio': {'excellent': 0.4, 'good': 0.7, 'fair': 1.2, 'poor': 2.0},
        'ROE - Return On Equity': {'excellent': 16, 'good': 12, 'fair': 8, 'poor': 4},
        'ROA - Return On Assets': {'excellent': 8, 'good': 6, 'fair': 4, 'poor': 2},
        'Operating Margin': {'excellent': 15, 'good': 10, 'fair': 6, 'poor': 3},
        'Net Profit Margin': {'excellent': 10, 'good': 7, 'fair': 4, 'poor': 2},
        'Asset Turnover': {'excellent': 1.2, 'good': 0.9, 'fair': 0.7, 'poor': 0.5},
        'altman_z_score': {'excellent': 3.8, 'good': 3.0, 'fair': 2.2, 'poor': 1.6},
        'kmv_distance_to_default': {'excellent': 3.8, 'good': 2.7, 'fair': 1.9, 'poor': 1.3}
    },
    'Materials': {
        'Current Ratio': {'excellent': 2.2, 'good': 1.6, 'fair': 1.2, 'poor': 0.9},
        'Debt/Equity Ratio': {'excellent': 0.4, 'good': 0.7, 'fair': 1.2, 'poor': 2.0},
        'ROE - Return On Equity': {'excellent': 14, 'good': 10, 'fair': 6, 'poor': 2},  # Cyclical sector
        'ROA - Return On Assets': {'excellent': 7, 'good': 5, 'fair': 3, 'poor': 1},
        'Operating Margin': {'excellent': 15, 'good': 10, 'fair': 6, 'poor': 2},
        'Net Profit Margin': {'excellent': 10, 'good': 7, 'fair': 4, 'poor': 1},
        'Asset Turnover': {'excellent': 1.1, 'good': 0.8, 'fair': 0.6, 'poor': 0.4},
        'altman_z_score': {'excellent': 3.4, 'good': 2.7, 'fair': 2.0, 'poor': 1.4},
        'kmv_distance_to_default': {'excellent': 3.4, 'good': 2.4, 'fair': 1.7, 'poor': 1.1}
    },
    'Real Estate': {  # REITs
        'Current Ratio': {'excellent': 1.3, 'good': 1.1, 'fair': 0.9, 'poor': 0.7},
        'Debt/Equity Ratio': {'excellent': 0.6, 'good': 0.9, 'fair': 1.4, 'poor': 2.2},
        'ROE - Return On Equity': {'excellent': 12, 'good': 9, 'fair': 6, 'poor': 3},  # Lower for REITs
        'ROA - Return On Assets': {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1},
        'Operating Margin': {'excellent': 35, 'good': 25, 'fair': 18, 'poor': 10},  # NOI margin
        'Net Profit Margin': {'excellent': 20, 'good': 15, 'fair': 10, 'poor': 5},
        'Asset Turnover': {'excellent': 0.10, 'good': 0.08, 'fair': 0.06, 'poor': 0.04},
        'altman_z_score': {'excellent': 2.8, 'good': 2.2, 'fair': 1.7, 'poor': 1.2},
        'kmv_distance_to_default': {'excellent': 2.8, 'good': 2.0, 'fair': 1.4, 'poor': 0.9}
    },
    'Utilities': {
        'Current Ratio': {'excellent': 1.4, 'good': 1.1, 'fair': 0.9, 'poor': 0.7},
        'Debt/Equity Ratio': {'excellent': 1.0, 'good': 1.4, 'fair': 2.2, 'poor': 3.2},  # Higher debt normal
        'ROE - Return On Equity': {'excellent': 11, 'good': 9, 'fair': 7, 'poor': 5},
        'ROA - Return On Assets': {'excellent': 3.5, 'good': 2.8, 'fair': 2.0, 'poor': 1.2},
        'Operating Margin': {'excellent': 22, 'good': 18, 'fair': 14, 'poor': 10},
        'Net Profit Margin': {'excellent': 15, 'good': 12, 'fair': 9, 'poor': 6},
        'Asset Turnover': {'excellent': 0.35, 'good': 0.28, 'fair': 0.22, 'poor': 0.18},
        'altman_z_score': {'excellent': 2.8, 'good': 2.2, 'fair': 1.7, 'poor': 1.2},
        'kmv_distance_to_default': {'excellent': 2.8, 'good': 2.0, 'fair': 1.4, 'poor': 0.9}
    },
    # Fallback for unknown sectors
    'General': {
        'Current Ratio': {'excellent': 2.0, 'good': 1.5, 'fair': 1.2, 'poor': 0.9},
        'Debt/Equity Ratio': {'excellent': 0.3, 'good': 0.6, 'fair': 1.0, 'poor': 1.8},
        'ROE - Return On Equity': {'excellent': 15, 'good': 11, 'fair': 7, 'poor': 3},
        'ROA - Return On Assets': {'excellent': 8, 'good': 5, 'fair': 3, 'poor': 1},
        'Operating Margin': {'excellent': 15, 'good': 10, 'fair': 6, 'poor': 3},
        'Net Profit Margin': {'excellent': 10, 'good': 6, 'fair': 3, 'poor': 1},
        'Asset Turnover': {'excellent': 1.5, 'good': 1.1, 'fair': 0.8, 'poor': 0.6},
        'altman_z_score': {'excellent': 3.5, 'good': 2.8, 'fair': 2.1, 'poor': 1.4},
        'kmv_distance_to_default': {'excellent': 3.5, 'good': 2.5, 'fair': 1.7, 'poor': 1.0}
    }
}


class StructuredModelService:
    """Service for structured financial data analysis and scoring."""
    
    def __init__(self):
        self.model: Optional[ExplainableBoostingClassifier] = None
        self.feature_columns: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.csv_file_path = "corporateCreditRatingWithFinancialRatios.csv"

    def _calculate_kmv(self, company_data: Dict) -> float:
        """
        Calculate RAD-KMV (Risk-Adjusted Distance) for enhanced default prediction.
        RAD-KMV improves traditional KMV by incorporating:
        - Time-varying volatility
        - Credit spread information
        - Market liquidity adjustments
        - Regime-dependent parameters
        Args:
            company_data: Dictionary containing financial data
        Returns:
            RAD-KMV Risk-Adjusted Distance-to-Default value
        """
        try:
            # Extract required data
            equity_value = company_data.get('market_cap', 0)
            debt_value = company_data.get('total_liabilities', 0)
            base_volatility = company_data.get('volatility', 0.3)
            risk_free_rate = 0.03  # 3% risk-free rate
            time_horizon = 1.0  # 1 year

            if equity_value <= 0 or debt_value <= 0:
                return 0.0

            # Asset value estimation
            asset_value = equity_value + debt_value
            default_point = debt_value * 0.8  # Use 80% of debt as default barrier

            # RAD-KMV Enhancements
            # 1. Time-Varying Volatility Adjustment
            # Incorporate regime-switching volatility based on market conditions
            # current_ratio = company_data.get('current_ratio', 1.0) # Already calculated below
            debt_ratio = debt_value / asset_value
            
            # Volatility regime adjustment
            if debt_ratio > 0.6:  # High leverage regime
                volatility_multiplier = 1.3
            elif debt_ratio < 0.3:  # Low leverage regime
                volatility_multiplier = 0.8
            else:  # Normal regime
                volatility_multiplier = 1.0
            
            adjusted_volatility = base_volatility * volatility_multiplier

            # 2. Credit Spread Adjustment
            # Estimate credit spread based on financial health
            roe = company_data.get('return_on_equity', 0)
            operating_margin = company_data.get('operating_margin', 0)
            
            # Credit quality score (0-1, higher is better)
            credit_quality = min(1.0, (roe * 0.4 + operating_margin * 0.6) / 0.2)
            
            # Credit spread (higher spread for lower quality)
            credit_spread = 0.01 + (1 - credit_quality) * 0.05  # 1-6% spread
            risk_adjusted_rate = risk_free_rate + credit_spread

            # 3. Market Liquidity Adjustment
            # Adjust for market liquidity conditions
            market_cap = company_data.get('market_cap', 0)
            
            # Liquidity factor (large cap = more liquid)
            if market_cap > 10e9:  # Large cap (>$10B)
                liquidity_adjustment = 1.0
            elif market_cap > 2e9:  # Mid cap ($2-10B)
                liquidity_adjustment = 1.1
            else:  # Small cap (<$2B)
                liquidity_adjustment = 1.2
            
            final_volatility = adjusted_volatility * liquidity_adjustment

            # 4. Enhanced Distance-to-Default Calculation
            # Traditional KMV d2 with risk adjustments
            mu = risk_adjusted_rate - 0.5 * final_volatility**2
            
            # Calculate distance-to-default
            distance_numerator = np.log(asset_value / default_point) + mu * time_horizon
            distance_denominator = final_volatility * np.sqrt(time_horizon)
            distance_to_default = distance_numerator / distance_denominator

            # 5. RAD-KMV Regime-Dependent Adjustment
            # Further adjust based on economic regime indicators
            # Financial health indicator
            current_ratio = company_data.get('current_ratio', 1.0)
            quick_ratio = company_data.get('quick_ratio', current_ratio * 0.8)
            
            # Health score (0-1)
            liquidity_health = min(1.0, (current_ratio + quick_ratio) / 3.0)
            profitability_health = min(1.0, max(0, roe / 0.15))
            overall_health = (liquidity_health + profitability_health) / 2

            # Regime adjustment factor
            if overall_health > 0.7:  # Healthy regime
                regime_factor = 1.1  # Slightly better DD
            elif overall_health < 0.3:  # Distressed regime
                regime_factor = 0.8  # Worse DD
            else:  # Normal regime
                regime_factor = 1.0

            # 6. Final RAD-KMV calculation
            rad_kmv_distance = distance_to_default * regime_factor

            # 7. Bounds checking and smoothing
            # Apply bounds to prevent extreme values
            rad_kmv_distance = max(-5.0, min(10.0, rad_kmv_distance))
            
            return float(rad_kmv_distance)
        except Exception as e:
            print(f"Error calculating RAD-KMV: {e}")
            return 0.0

    def _calculate_z_score(self, company_data: Dict) -> float:
        """
        Calculate Enhanced Z-Score for advanced bankruptcy prediction.
        Enhanced Z-Score = 1.5×X1 + 1.6×X2 + 3.5×X3 + 0.8×X4 + 1.2×X5
        Args:
            company_data: Dictionary containing financial ratios
        Returns:
            Enhanced Z-Score value
        """
        try:
            # X1: Dynamic Liquidity Stress Index
            total_assets = company_data.get('total_assets', 1) # Store for reuse
            current_ratio = company_data.get('current_ratio', 1.0) # Store for reuse
            
            current_assets = total_assets * current_ratio
            current_liabilities = current_assets / max(current_ratio, 0.1)
            working_capital = current_assets - current_liabilities
            wc_ta = working_capital / max(total_assets, 1)
            quick_ratio = company_data.get('quick_ratio', current_ratio * 0.8)
            cash_ratio = company_data.get('cash_ratio', quick_ratio * 0.5)  # Approximation
            cash_conversion_cycle = company_data.get('cash_conversion_cycle', 60)  # Default 60 days
            
            x1 = (0.4 * wc_ta + 
                  0.3 * (quick_ratio / 2.0) + 
                  0.2 * cash_ratio + 
                  0.1 * max(0, (90 - cash_conversion_cycle) / 90))

            # X2: Multi-Period Earning Quality Score
            total_equity = company_data.get('total_equity', 0) # Store for reuse
            retained_earnings_ta = total_equity / max(total_assets, 1)
            
            # Cash Earnings Quality (approximated from operating cash flow vs net income)
            operating_margin = company_data.get('operating_margin', 0)
            net_margin = company_data.get('net_margin', 0)
            cash_earnings_quality = (operating_margin / max(net_margin, 0.01)) if net_margin > 0 else 1.0
            
            # ROE Stability (simplified as consistency measure)
            roe = company_data.get('return_on_equity', 0)
            roe_stability = min(1.0, roe / 0.15) if roe > 0 else 0
            
            # Margin Trend (simplified)
            margin_trend = min(1.0, max(-1.0, (net_margin - 0.05) / 0.1))  # Relative to 5% baseline
            
            x2 = (0.4 * retained_earnings_ta + 
                  0.25 * max(0, cash_earnings_quality - 1.0) + 
                  0.25 * max(0, roe_stability) + 
                  0.1 * np.tanh(margin_trend))

            # X3: Risk-Adjusted Operational Performance
            operating_income = company_data.get('operating_income', 0) # Store for reuse
            ebit_ta = operating_income / max(total_assets, 1)
            asset_volatility = company_data.get('volatility', 0.3)
            risk_adjustment = max(0.5, 1 - (asset_volatility / 0.4))
            
            # Industry adjustment (simplified - can be enhanced with sector data)
            industry_adjustment = 0.02  # Neutral adjustment
            
            # Operating leverage penalty (simplified)
            total_liabilities = company_data.get('total_liabilities', 0) # Store for reuse
            debt_ratio = total_liabilities / max(total_assets, 1)
            operating_leverage_penalty = debt_ratio * 0.01
            
            x3 = (ebit_ta * risk_adjustment + 
                  0.3 * industry_adjustment - 
                  operating_leverage_penalty)

            # X4: Multi-Dimensional Solvency Score
            market_cap = company_data.get('market_cap', 0) # Store for reuse
            market_cap_debt = min(market_cap / max(total_liabilities, 1), 3.0)
            
            # Interest Coverage (approximated from operating income)
            estimated_interest = total_liabilities * 0.05  # Assume 5% average interest rate
            interest_coverage = operating_income / max(estimated_interest, 1) if estimated_interest > 0 else 10
            
            # Debt Maturity Risk (simplified)
            debt_maturity_risk = min(0.5, debt_ratio)
            
            # Debt Service Ratio (approximated)
            debt_service_ratio = debt_ratio * 2  # Simplified approximation
            
            # Market Risk Penalty (based on volatility)
            market_risk_penalty = asset_volatility * 0.1
            
            x4 = (0.4 * market_cap_debt + 
                  0.25 * min(interest_coverage / 3.0, 1.0) + 
                  0.2 * max(0, 1 - debt_maturity_risk) + 
                  0.1 * max(0, 1 - debt_service_ratio / 4.0) + 
                  0.05 * max(0, -market_risk_penalty))

            # X5: Dynamic Asset Efficiency Index
            asset_turnover = company_data.get('asset_turnover', 0)
            
            # Turnover Trend (simplified - positive if above industry average)
            turnover_trend = max(0, (asset_turnover - 1.0) / 1.0)
            
            # Asset Quality Ratio (approximated from tangible assets)
            asset_quality = 0.8  # Simplified assumption
            
            # Working Capital Turnover (approximated)
            revenue = total_assets * asset_turnover
            wc_turnover = revenue / max(abs(working_capital), 1) if working_capital != 0 else 5
            
            # Revenue per Employee Factor (simplified)
            revenue_per_employee_factor = 1.0  # Neutral since we don't have employee data
            
            x5 = (0.4 * asset_turnover + 
                  0.2 * max(0, turnover_trend * 10) + 
                  0.2 * asset_quality + 
                  0.1 * min(wc_turnover / 5.0, 1.0) + 
                  0.1 * (revenue_per_employee_factor - 1.0))

            # Enhanced Z-Score calculation
            enhanced_z_score = 1.5*x1 + 1.6*x2 + 3.5*x3 + 0.8*x4 + 1.2*x5
            
            return float(enhanced_z_score)
        except Exception as e:
            print(f"Error calculating Enhanced Z-Score: {e}")
            return 0.0

    # Removed _prepare_features (legacy Random Forest support)

    def _prepare_features_for_ebm(self, company_data: Dict, precomputed_z_score: Optional[float] = None, precomputed_kmv: Optional[float] = None) -> np.ndarray:
        """
        Prepare features for EBM model from company data.
        Args:
            company_data: Dictionary containing financial data
            precomputed_z_score: Optional precomputed Z-Score to avoid recalculation
            precomputed_kmv: Optional precomputed KMV to avoid recalculation
        Returns:
            NumPy array of features matching CSV training data
        """
        # Calculate KMV and Z-Score if not provided
        kmv_dd = precomputed_kmv if precomputed_kmv is not None else self._calculate_kmv(company_data)
        z_score = precomputed_z_score if precomputed_z_score is not None else self._calculate_z_score(company_data)

        # Map database fields to CSV-like structure for consistency
        # Note: Approximations are made for margins due to data limitations.
        # If distinct margin data is available, map them correctly.
        operating_margin_pct = company_data.get('operating_margin', 0) * 100
        net_margin_pct = company_data.get('net_margin', 0) * 100
        roe_pct = company_data.get('return_on_equity', 0) * 100
        roa_pct = company_data.get('return_on_assets', 0) * 100

        features = [
            company_data.get('current_ratio', 0),
            company_data.get('debt_to_equity', 0),
            operating_margin_pct,  # Gross Margin (approximation)
            operating_margin_pct,  # Operating Margin
            operating_margin_pct,  # EBIT Margin (approximation)
            operating_margin_pct,  # EBITDA Margin (approximation)
            net_margin_pct,        # Net Profit Margin
            company_data.get('asset_turnover', 0),
            roe_pct,               # ROE - Return On Equity
            roa_pct,               # ROA - Return On Assets
            z_score,
            kmv_dd
        ]
        return np.array(features).reshape(1, -1)

    def _load_csv_data(self) -> pd.DataFrame:
        """
        Load and preprocess data from CSV file.
        Returns:
            Preprocessed DataFrame with financial data
        """
        try:
            # Load CSV data
            df = pd.read_csv(self.csv_file_path)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Remove rows with missing critical data
            df = df.dropna(subset=['Current Ratio', 'Debt/Equity Ratio', 'ROE - Return On Equity'])
            
            # Create binary target from rating
            # Investment grade (BBB- and above) = 1, Below investment grade = 0
            investment_grades = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-']
            df['is_investment_grade'] = df['Rating'].isin(investment_grades).astype(int)
            
            # Feature engineering for Enhanced Z-Score and RAD-KMV
            df['working_capital_to_assets'] = (df['Current Ratio'] - 1) / df['Current Ratio']  # Approximation
            df['retained_earnings_to_assets'] = df['ROE - Return On Equity'] / 100  # Approximation
            df['ebit_to_assets'] = df['EBIT Margin'] / 100  # EBIT/Assets approximation
            df['market_value_to_debt'] = 1 / (df['Debt/Equity Ratio'] + 0.01)  # Avoid division by zero
            df['sales_to_assets'] = df['Asset Turnover']
            
            # Calculate Enhanced Z-Score (simplified for CSV data)
            # Using available data to approximate enhanced components
            df['enhanced_z_x1'] = 0.4 * df['working_capital_to_assets'] + 0.3 * (df['Current Ratio'] / 2.0)
            df['enhanced_z_x2'] = 0.4 * df['retained_earnings_to_assets'] + 0.25 * np.tanh((df['Net Profit Margin'] - 5) / 10)
            df['enhanced_z_x3'] = df['ebit_to_assets'] * 0.9  # Risk adjustment approximation
            df['enhanced_z_x4'] = 0.4 * np.minimum(df['market_value_to_debt'], 3.0) + 0.25 * np.minimum(df['EBIT Margin'] / 30, 1.0)
            df['enhanced_z_x5'] = 0.4 * df['sales_to_assets'] + 0.2 * np.maximum(0, (df['Asset Turnover'] - 1.0))
            
            # Enhanced Z-Score calculation
            df['altman_z_score'] = (
                1.5 * df['enhanced_z_x1'] +
                1.6 * df['enhanced_z_x2'] +
                3.5 * df['enhanced_z_x3'] +
                0.8 * df['enhanced_z_x4'] +
                1.2 * df['enhanced_z_x5']
            )
            
            # Calculate RAD-KMV Distance-to-Default (simplified for CSV)
            df['volatility'] = 0.3  # Default volatility, could be enhanced with actual data
            df['leverage_ratio'] = df['Debt/Equity Ratio'] / (1 + df['Debt/Equity Ratio'])
            df['credit_quality'] = np.minimum(1.0, (df['ROE - Return On Equity'] * 0.004 + df['Operating Margin'] * 0.006) / 0.2)
            df['volatility_adjusted'] = df['volatility'] * (1.0 + (1 - df['credit_quality']) * 0.5)
            df['kmv_distance_to_default'] = np.log(1 / (df['leverage_ratio'] + 0.01)) / (df['volatility_adjusted'] * np.sqrt(1))
            
            # Apply regime adjustments for RAD-KMV
            df['health_score'] = (np.minimum(1.0, df['Current Ratio'] / 2.0) + np.minimum(1.0, df['ROE - Return On Equity'] / 15)) / 2
            df['regime_factor'] = np.where(df['health_score'] > 0.7, 1.1, 
                                         np.where(df['health_score'] < 0.3, 0.8, 1.0))
            df['kmv_distance_to_default'] = df['kmv_distance_to_default'] * df['regime_factor']
            
            return df
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            raise

    def train(self) -> Dict:
        """
        Train the EBM model using CSV data.
        Returns:
            Dictionary containing training results
        """
        try:
            # Load and preprocess data
            df = self._load_csv_data()
            
            # Define feature columns
            self.feature_columns = [
                'Current Ratio',
                'Debt/Equity Ratio', 
                'Gross Margin',
                'Operating Margin',
                'EBIT Margin',
                'EBITDA Margin',
                'Net Profit Margin',
                'Asset Turnover',
                'ROE - Return On Equity',
                'ROA - Return On Assets',
                'altman_z_score',
                'kmv_distance_to_default'
            ]
            
            # Prepare features and target
            X = df[self.feature_columns].fillna(0)  # Fill missing values with 0
            y = df['is_investment_grade']
            
            # Remove any remaining NaN or infinite values
            X = X.replace([np.inf, -np.inf], 0)
            X = X.fillna(0)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features for better performance
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train EBM model (single-threaded for Windows compatibility)
            self.model = ExplainableBoostingClassifier(
                random_state=42,
                learning_rate=0.01,
                max_bins=256,
                max_interaction_bins=32,
                interactions=10,
                n_jobs=1  # Single-threaded to avoid Windows multiprocessing issues
            )
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model and scaler
            import os
            model_dir = os.path.dirname(settings.random_forest_model_path)
            os.makedirs(model_dir, exist_ok=True)
            model_path = settings.random_forest_model_path.replace('random_forest', 'ebm')
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'feature_columns': self.feature_columns,
                    'scaler': self.scaler
                }, f)
            
            print(f"✅ EBM Model trained successfully!")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   Training samples: {len(X_train)}")
            print(f"   Test samples: {len(X_test)}")
            print(f"   Features: {len(self.feature_columns)}")
            
            return {
                'accuracy': accuracy,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': len(self.feature_columns),
                'model_path': model_path,
                'model_type': 'EBM'
            }
        except Exception as e:
            print(f"Error training EBM model: {e}")
            raise

    def _load_model(self):
        """Load the trained EBM model."""
        if self.model is None:
            try:
                # Load EBM model
                model_path = settings.random_forest_model_path.replace('random_forest', 'ebm')
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.feature_columns = model_data['feature_columns']
                    self.scaler = model_data.get('scaler')
            except FileNotFoundError:
                # If EBM model not found, raise an error
                raise ValueError("No trained EBM model found. Please train the model first.")
            # Removed legacy Random Forest fallback

    def _calculate_feature_impacts(self, features: np.ndarray, company_data: Dict) -> Dict:
        """
        Calculate specific point impacts of each feature on the credit score.
        Args:
            features: Input features array (scaled if scaler exists)
            company_data: Original company data
        Returns:
            Dictionary with detailed feature impacts
        """
        try:
            if not isinstance(self.model, ExplainableBoostingClassifier):
                return {"error": "Feature impacts only available for EBM models"}
                
            # Get prediction for current features
            current_prediction = self.model.predict_proba(features)[0][1] * 100
            
            # Calculate baseline (neutral) features for comparison
            neutral_features = np.zeros_like(features)
            if self.scaler:
                # Use scaled neutral values (mean = 0 for StandardScaler)
                neutral_features = np.zeros_like(features) # Already 0s, scaling doesn't change them
                
            baseline_prediction = self.model.predict_proba(neutral_features)[0][1] * 100
            
            # Calculate individual feature impacts by masking
            feature_impacts = {}
            for i, feature_name in enumerate(self.feature_columns):
                # Create feature array with all features at neutral except current one
                test_features = neutral_features.copy()
                test_features[0][i] = features[0][i]  # Only this feature is non-neutral
                test_prediction = self.model.predict_proba(test_features)[0][1] * 100
                impact = test_prediction - baseline_prediction
                feature_impacts[feature_name] = {
                    'value': float(features[0][i]),
                    'impact_points': float(impact),
                    'raw_feature_value': float(features[0][i])
                }
            
            return {
                'current_score': float(current_prediction),
                'baseline_score': float(baseline_prediction),
                'total_impact': float(current_prediction - baseline_prediction),
                'feature_impacts': feature_impacts
            }
        except Exception as e:
            print(f"Error calculating feature impacts: {e}")
            return {"error": str(e)}

    def _generate_numeric_explanations(self, company_data: Dict, feature_impacts: Dict, sector: Optional[str] = None) -> List[str]:
        """
        Generate human-readable numeric explanations for each feature impact.
        Args:
            company_data: Original company financial data
            feature_impacts: Feature impact analysis from _calculate_feature_impacts
            sector: Optional sector name for sector-specific benchmarks
        Returns:
            List of detailed explanations
        """
        explanations = []
        if 'feature_impacts' not in feature_impacts:
            return ["Error: Unable to generate feature explanations"]

        # Select appropriate benchmarks
        if sector and sector in SP500_SECTOR_BENCHMARKS:
            benchmarks = SP500_SECTOR_BENCHMARKS[sector]
        else:
            benchmarks = SP500_SECTOR_BENCHMARKS['General']

        # Map feature names to original values from company_data
        # This avoids recalculating Z-Score and KMV
        feature_mappings = {
            'Current Ratio': company_data.get('current_ratio', 0),
            'Debt/Equity Ratio': company_data.get('debt_to_equity', 0),
            'Gross Margin': company_data.get('operating_margin', 0) * 100, # Approximation
            'Operating Margin': company_data.get('operating_margin', 0) * 100,
            'EBIT Margin': company_data.get('operating_margin', 0) * 100, # Approximation
            'EBITDA Margin': company_data.get('operating_margin', 0) * 100, # Approximation
            'Net Profit Margin': company_data.get('net_margin', 0) * 100,
            'Asset Turnover': company_data.get('asset_turnover', 0),
            'ROE - Return On Equity': company_data.get('return_on_equity', 0) * 100,
            'ROA - Return On Assets': company_data.get('return_on_assets', 0) * 100,
        }

        for feature_name, impact_data in feature_impacts['feature_impacts'].items():
            impact_points = impact_data['impact_points']
            
            # Get original value from mapping or precomputed values
            if feature_name in feature_mappings:
                original_value = feature_mappings[feature_name]
            elif feature_name == 'altman_z_score':
                original_value = self._calculate_z_score(company_data)
            elif feature_name == 'kmv_distance_to_default':
                original_value = self._calculate_kmv(company_data)
            else:
                original_value = impact_data['value']

            # Determine performance level
            performance_level = "average"
            if feature_name in benchmarks:
                bench = benchmarks[feature_name]
                if feature_name == 'Debt/Equity Ratio':  # Lower is better
                    if original_value <= bench['excellent']:
                        performance_level = "excellent"
                    elif original_value <= bench['good']:
                        performance_level = "good"
                    elif original_value <= bench['fair']:
                        performance_level = "fair"
                    else:
                        performance_level = "poor"
                else:  # Higher is better
                    if original_value >= bench['excellent']:
                        performance_level = "excellent"
                    elif original_value >= bench['good']:
                        performance_level = "good"
                    elif original_value >= bench['fair']:
                        performance_level = "fair"
                    else:
                        performance_level = "poor"

            # Generate explanation
            impact_direction = "increase" if impact_points > 0 else "decrease"
            impact_magnitude = abs(impact_points)
            
            explanation = ""
            if feature_name == 'Current Ratio':
                explanation = f"Current Ratio was {original_value:.2f} which is {performance_level} and caused a {impact_direction} of {impact_magnitude:.1f} points in the credit score"
            elif feature_name == 'Debt/Equity Ratio':
                explanation = f"Debt-to-Equity Ratio was {original_value:.2f} which is {performance_level} and caused a {impact_direction} of {impact_magnitude:.1f} points in the credit score"
            elif 'Margin' in feature_name:
                explanation = f"{feature_name} was {original_value:.1f}% which is {performance_level} and caused a {impact_direction} of {impact_magnitude:.1f} points in the credit score"
            elif 'ROE' in feature_name or 'ROA' in feature_name:
                explanation = f"{feature_name} was {original_value:.1f}% which is {performance_level} and caused a {impact_direction} of {impact_magnitude:.1f} points in the credit score"
            elif feature_name == 'Asset Turnover':
                explanation = f"Asset Turnover was {original_value:.2f} which is {performance_level} and caused a {impact_direction} of {impact_magnitude:.1f} points in the credit score"
            elif feature_name == 'altman_z_score':
                explanation = f"Enhanced Z-Score was {original_value:.2f} which is {performance_level} and caused a {impact_direction} of {impact_magnitude:.1f} points in the credit score"
            elif feature_name == 'kmv_distance_to_default':
                explanation = f"RAD-KMV Distance-to-Default was {original_value:.2f} which is {performance_level} and caused a {impact_direction} of {impact_magnitude:.1f} points in the credit score"
            else:
                explanation = f"{feature_name} was {original_value:.2f} which is {performance_level} and caused a {impact_direction} of {impact_magnitude:.1f} points in the credit score"

            # Only include features with significant impact (> 0.5 points)
            if impact_magnitude > 0.5:
                explanations.append(explanation)

        # Sort by impact magnitude (descending)
        explanations.sort(key=lambda x: float(x.split('of ')[1].split(' points')[0]), reverse=True)
        
        return explanations

    def _score_to_grade(self, score: float) -> str:
        """
        Convert numeric score to credit grade.
        Args:
            score: Numeric credit score (0-100)
        Returns:
            Credit grade string
        """
        if score >= 95:
            return "AAA"
        elif score >= 90:
            return "AA"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "BBB"
        elif score >= 70:
            return "BB"
        elif score >= 60:
            return "B"
        elif score >= 50:
            return "CCC"
        elif score >= 40:
            return "CC"
        elif score >= 30:
            return "C"
        else:
            return "D"

    def predict(self, company_data: Dict, sector: Optional[str] = None) -> Dict:
        """
        Calculate structured credit score for raw company data.
        Args:
            company_data: Dictionary containing financial metrics
            sector: Optional sector name for sector-specific analysis
        Returns:
            Dictionary containing scores and explanations
        """
        self._load_model()
        try:
            # Calculate Z-Score and KMV once
            z_score = self._calculate_z_score(company_data)
            kmv_dd = self._calculate_kmv(company_data)
            
            # Prepare features for EBM using precomputed values
            features = self._prepare_features_for_ebm(company_data, precomputed_z_score=z_score, precomputed_kmv=kmv_dd)
            if self.scaler:
                features = self.scaler.transform(features)
            
            # Get prediction
            prediction_proba = self.model.predict_proba(features)[0]
            structured_score = prediction_proba[1] * 100
            
            # Calculate feature impacts for explanations
            feature_impacts = self._calculate_feature_impacts(features, company_data)
            
            # Generate numeric explanations using precomputed values
            explanations = []
            if 'error' not in feature_impacts:
                explanations = self._generate_numeric_explanations(company_data, feature_impacts, sector)
            
            # Determine credit grade and investment grade
            credit_grade = self._score_to_grade(structured_score)
            investment_grade = structured_score >= 70
            
            result = {
                'score': structured_score,
                'grade': credit_grade,
                'investment_grade': investment_grade,
                'explanations': explanations,
                'feature_impacts': feature_impacts if 'error' not in feature_impacts else None
            }
            
            # Include sector information if provided
            if sector:
                result['sector'] = sector
                
            return result
        except Exception as e:
            return {
                'error': f'Error calculating structured score: {str(e)}',
                'score': 0,
                'grade': 'Unknown',
                'investment_grade': False,
                'explanations': []
            }

    def get_structured_score(self, company_id: int, sector: Optional[str] = None) -> Dict:
        """
        Calculate structured credit score for a company.
        Args:
            company_id: Company ID
            sector: Optional sector name for sector-specific analysis
        Returns:
            Dictionary containing scores and explanations
        """
        self._load_model()
        db = get_db_session()
        try:
            # Get latest structured data
            latest_data = (
                db.query(StructuredData)
                .filter(StructuredData.company_id == company_id)
                .order_by(StructuredData.data_date.desc())
                .first()
            )
            if not latest_data:
                raise ValueError(f"No structured data found for company {company_id}")

            # Prepare company data dictionary
            company_data = {
                'current_ratio': latest_data.current_ratio or 0,
                'quick_ratio': latest_data.quick_ratio or 0,
                'debt_to_equity': latest_data.debt_to_equity or 0,
                'return_on_equity': latest_data.return_on_equity or 0,
                'return_on_assets': latest_data.return_on_assets or 0,
                'operating_margin': latest_data.operating_margin or 0,
                'net_margin': latest_data.net_margin or 0,
                'asset_turnover': latest_data.asset_turnover or 0,
                'inventory_turnover': latest_data.inventory_turnover or 0,
                'total_assets': latest_data.total_assets or 0,
                'total_liabilities': latest_data.total_liabilities or 0,
                'total_equity': latest_data.total_equity or 0,
                'market_cap': latest_data.market_cap or 0,
                'operating_income': latest_data.operating_income or 0,
                'volatility': latest_data.volatility or 0.3
            }

            # Calculate KMV and Z-Score once
            kmv_dd = self._calculate_kmv(company_data)
            z_score = self._calculate_z_score(company_data)

            # Update database with calculated values
            latest_data.kmv_distance_to_default = kmv_dd
            latest_data.altman_z_score = z_score
            db.commit()

            # Prepare features for EBM model using precomputed values
            features = self._prepare_features_for_ebm(company_data, precomputed_z_score=z_score, precomputed_kmv=kmv_dd)
            if self.scaler:
                features = self.scaler.transform(features)

            # Get model prediction
            prediction_proba = self.model.predict_proba(features)[0]
            structured_score = prediction_proba[1] * 100  # Probability of good credit * 100

            # Calculate detailed feature impacts for numeric explanations
            feature_impacts = self._calculate_feature_impacts(features, company_data)
            
            # Generate numeric explanations using precomputed values
            numeric_explanations = self._generate_numeric_explanations(company_data, feature_impacts, sector)

            # Removed legacy SHAP and Random Forest support
            
            result = {
                'structured_score': float(structured_score),
                'kmv_distance_to_default': float(kmv_dd),
                'altman_z_score': float(z_score),
                'shap_values': None, # Removed SHAP
                'feature_contributions': None, # Removed legacy SHAP
                'feature_names': self.feature_columns,
                'feature_values': features.flatten().tolist(),
                'company_data': company_data,
                'model_type': 'EBM', # Standardized to EBM
                'feature_impacts': feature_impacts,  # Detailed numeric impacts
                'numeric_explanations': numeric_explanations  # Human-readable explanations
            }
            
            # Include sector information if provided
            if sector:
                result['sector'] = sector
                
            return result
        except Exception as e:
            print(f"Error calculating structured score: {e}")
            raise
        finally:
            db.close()
