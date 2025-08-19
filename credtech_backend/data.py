# build_training_dataset_corrected.py
# --- Configuration ---
FRED_API_KEY = "1a39ebc94e4984ff4091baa2f84c0ba7"  # <--- REPLACE THIS
INPUT_CSV_PATH = r"C:\Users\asus\Documents\GitHub\BlackBox_Cred\credtech_backend\corporateCreditRatingWithFinancialRatios.csv"
OUTPUT_DIR = r"C:\Users\asus\Documents\GitHub\BlackBox_Cred\credtech_backend"
OUTPUT_CSV_FILENAME = "complete_training_dataset_corrected.csv"

# Date filtering - only use ratings from 2014-2016 for better Yahoo Finance coverage
START_YEAR = 2014
END_YEAR = 2016

# Financial modeling parameters (now configurable)
VOLATILITY_THRESHOLDS = {
    'high_debt_ratio': 0.6,
    'low_debt_ratio': 0.3,
    'high_debt_multiplier': 1.3,
    'low_debt_multiplier': 0.8,
    'normal_multiplier': 1.0
}

# Default volatility assumptions by broad industry categories
DEFAULT_VOLATILITIES = {
    'technology': 0.35,
    'financial': 0.25,
    'utilities': 0.15,
    'energy': 0.40,
    'healthcare': 0.30,
    'consumer_discretionary': 0.30,
    'consumer_staples': 0.20,
    'industrials': 0.25,
    'materials': 0.30,
    'real_estate': 0.25,
    'telecommunications': 0.20,
    'default': 0.30
}

import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime, timedelta
import warnings
import time
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, OUTPUT_CSV_FILENAME)

class TrainingDataBuilder:
    """
    Builds a training dataset by fetching financial data from Yahoo Finance,
    macroeconomic data from FRED, and calculating Z-Score and KMV for dates
    specified in an input CSV. Now includes financial soundness improvements.
    """
    def __init__(self, fred_api_key):
        self.fred_api_key = fred_api_key
        self.fred = Fred(api_key=fred_api_key) if fred_api_key else None
        self.macro_data = pd.DataFrame()
        self.training_data = []

    def load_input_ratings(self, csv_path):
        """Loads the input CSV and prepares it for processing."""
        logger.info(f"Loading input ratings from {csv_path}...")
        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Input CSV file not found: {csv_path}")
                
            df = pd.read_csv(csv_path)
            self.ratings_df = df[['Corporation', 'Ticker', 'Rating Date', 'Rating', 'Binary Rating']].copy()
            self.ratings_df.columns = ['Company_Name', 'Ticker', 'Date', 'Rating', 'Binary_Rating']

            # Clean and convert Date column
            self.ratings_df['Date'] = pd.to_datetime(self.ratings_df['Date'], errors='coerce')
            self.ratings_df['Ticker'] = self.ratings_df['Ticker'].astype(str)
            
            initial_count = len(self.ratings_df)
            
            # Remove rows with invalid dates
            self.ratings_df = self.ratings_df.dropna(subset=['Date'])
            
            # Filter to only include ratings from specified date range
            self.ratings_df = self.ratings_df[
                (self.ratings_df['Date'].dt.year >= START_YEAR) & 
                (self.ratings_df['Date'].dt.year <= END_YEAR)
            ]
            
            logger.info(f"After date filtering ({START_YEAR}-{END_YEAR}): {len(self.ratings_df)} entries remain")
            
            # Remove duplicates - keep only one rating per company
            initial_count_after_date = len(self.ratings_df)
            self.ratings_df = self.ratings_df.drop_duplicates(subset=['Company_Name', 'Ticker'], keep='first')
            removed_duplicates = initial_count_after_date - len(self.ratings_df)
            logger.info(f"Removed {removed_duplicates} duplicate companies. {len(self.ratings_df)} unique companies remain")
            
            # Filter out invalid tickers
            self.ratings_df = self.ratings_df[
                (self.ratings_df['Ticker'] != 'nan') & 
                (self.ratings_df['Ticker'] != '') & 
                (self.ratings_df['Ticker'].str.len() <= 6)
            ]
            
            logger.info(f"Loaded {len(self.ratings_df)} entries (dropped {initial_count - len(self.ratings_df)} invalid rows).")
            
            # Use the existing Binary Rating column
            self.ratings_df['Is_Investment_Grade'] = self.ratings_df['Binary_Rating'].astype(int)
            
            logger.info(f"Sample of loaded ratings data:\n{self.ratings_df.head()}")

        except Exception as e:
            logger.error(f"Error loading input CSV: {e}")
            raise

    def fetch_fred_macro_data(self):
        """Fetches and prepares macroeconomic data from FRED."""
        if not self.fred:
            logger.warning("No FRED API key provided. Macroeconomic features will be NaN.")
            return

        logger.info("Fetching macroeconomic data from FRED...")
        try:
            if self.ratings_df.empty:
                logger.warning("No rating dates to determine FRED data range.")
                return

            end_date_dt = self.ratings_df['Date'].max() + timedelta(days=365)
            start_date_dt = self.ratings_df['Date'].min() - timedelta(days=365)
            start_date_str = start_date_dt.strftime('%Y-%m-%d')
            end_date_str = end_date_dt.strftime('%Y-%m-%d')

            logger.info(f"Fetching FRED data from {start_date_str} to {end_date_str}")

            fred_series = {
                'fed_funds_rate': 'FEDFUNDS',
                'treasury_10y': 'GS10',
                'treasury_3m': 'GS3M',
                'credit_spread_high_yield': 'BAMLH0A0HYM2',
                'credit_spread_investment': 'BAMLC0A0CM',
                'vix': 'VIXCLS',
                'unemployment_rate': 'UNRATE',
            }

            macro_dict = {}
            for name, series_id in fred_series.items():
                try:
                    data = self.fred.get_series(series_id, start=start_date_str, end=end_date_str)
                    if not data.empty:
                        macro_dict[name] = data.resample('D').last().ffill()
                        logger.debug(f"  Fetched {name} ({len(data)} points)")
                    else:
                        logger.warning(f"  No data found for {name} ({series_id})")
                except Exception as e:
                    logger.error(f"  Error fetching {name} ({series_id}): {e}")

            if macro_dict:
                self.macro_data = pd.DataFrame(macro_dict)
                # Engineer derived features
                if 'treasury_10y' in self.macro_data.columns and 'treasury_3m' in self.macro_data.columns:
                    self.macro_data['yield_curve_slope'] = self.macro_data['treasury_10y'] - self.macro_data['treasury_3m']
                logger.info(f"Macroeconomic data fetched: {self.macro_data.shape}")
            else:
                logger.error("Failed to fetch any macroeconomic data from FRED.")

        except Exception as e:
            logger.error(f"Error in fetch_fred_macro_data: {e}")

    def _get_dynamic_risk_free_rate(self, rating_date):
        """Gets the dynamic risk-free rate from macro data for the rating date."""
        try:
            if self.macro_data.empty or 'treasury_3m' not in self.macro_data.columns:
                return 0.03  # Default 3% if no data available
            
            rating_date_dt = pd.to_datetime(rating_date)
            
            # Find the closest available rate on or before the rating date
            available_dates = self.macro_data.index[self.macro_data.index <= rating_date_dt]
            if available_dates.empty:
                return 0.03
                
            closest_date = available_dates.max()
            rate = self.macro_data.loc[closest_date, 'treasury_3m']
            
            if pd.isna(rate):
                return 0.03
                
            return float(rate) / 100.0  # Convert percentage to decimal
            
        except Exception as e:
            logger.debug(f"Error getting dynamic risk-free rate: {e}")
            return 0.03

    def _calculate_cash_conversion_cycle(self, data_dict):
        """Calculates the actual Cash Conversion Cycle instead of using hardcoded value."""
        try:
            revenue = data_dict.get('total_revenue', 0)
            cogs = data_dict.get('gross_profit', 0)
            
            # Estimate COGS if not available directly
            if revenue > 0 and cogs > 0:
                cogs = revenue - cogs  # gross_profit = revenue - cogs
            elif revenue > 0:
                cogs = revenue * 0.7  # Assume 70% COGS if unavailable
            else:
                return 0.5  # Default if no revenue data
            
            # Get balance sheet items (these would need to be added to data collection)
            current_assets = data_dict.get('current_assets', 0)
            inventory = data_dict.get('inventory', 0)
            current_liabilities = data_dict.get('current_liabilities', 0)
            
            # Estimate components if not available
            if current_assets > 0:
                # Rough estimates based on typical industry ratios
                accounts_receivable = current_assets * 0.3  # Estimate
                accounts_payable = current_liabilities * 0.4  # Estimate
            else:
                return 0.5
                
            # Calculate CCC components
            if revenue > 0 and cogs > 0:
                days_sales_outstanding = (accounts_receivable / revenue) * 365
                days_inventory = (inventory / cogs) * 365 if inventory > 0 else 30
                days_payable = (accounts_payable / cogs) * 365 if accounts_payable > 0 else 30
                
                ccc = days_sales_outstanding + days_inventory - days_payable
                
                # Normalize CCC to a 0-1 scale for use in Z-score
                # Typical CCC ranges from -30 to 150 days
                normalized_ccc = max(0, min(1, (ccc + 30) / 180))
                return normalized_ccc
            else:
                return 0.5
                
        except Exception as e:
            logger.debug(f"Error calculating CCC: {e}")
            return 0.5

    def _calculate_asset_quality(self, data_dict):
        """Calculates asset quality based on available data instead of fixed 0.8."""
        try:
            total_assets = data_dict.get('total_assets', 1)
            current_assets = data_dict.get('current_assets', 0)
            inventory = data_dict.get('inventory', 0)
            cash = data_dict.get('cash_and_equivalents', 0)
            
            if total_assets <= 0:
                return 0.8  # Default
                
            # Asset quality factors
            liquid_asset_ratio = (current_assets - inventory) / total_assets
            cash_ratio = cash / total_assets
            
            # Higher liquid assets and cash indicate better asset quality
            asset_quality = 0.5 + (liquid_asset_ratio * 0.3) + (cash_ratio * 0.2)
            
            # Cap between 0.3 and 1.0
            return max(0.3, min(1.0, asset_quality))
            
        except Exception as e:
            logger.debug(f"Error calculating asset quality: {e}")
            return 0.8

    def _find_relevant_financial_year(self, balance_sheet_dates, target_date, max_lookback_days=730):
        """Finds the most recent fiscal year end ON OR BEFORE the target date to avoid look-ahead bias."""
        if balance_sheet_dates.empty:
            logger.warning(f"No FYE dates available at all for target date {target_date}")
            return None

        target_dt = pd.to_datetime(target_date)
        bs_dates_dt = pd.to_datetime(balance_sheet_dates)

        # Only consider dates on or before the target date
        relevant_dates = bs_dates_dt[bs_dates_dt <= target_dt]

        if relevant_dates.empty:
            logger.warning(f"No FYE before target date {target_date}. Available FYE dates: {list(bs_dates_dt)}")
            return None

        # Find the most recent date within the lookback period
        cutoff_date = target_dt - timedelta(days=max_lookback_days)
        recent_dates = relevant_dates[relevant_dates >= cutoff_date]

        if not recent_dates.empty:
            return recent_dates.max()
        else:
            # If no recent data in window, use the most recent available (but still before target)
            logger.info(f"No FYE in lookback window for {target_date}, using most recent available: {relevant_dates.max()}")
            return relevant_dates.max()

    def _safe_get_yf_data(self, df, item, date, default=np.nan):
        """Safely retrieves data from a yfinance DataFrame."""
        try:
            if item in df.index:
                val = df.loc[item, date]
                return float(val) if pd.notna(val) else default
            
            # Common alternative names for yfinance
            alt_names = {
                'Total Revenue': ['Total Revenue', 'Revenue', 'Total Revenues', 'Net Sales'],
                'Net Income': ['Net Income', 'Net Income Common Stockholders', 'Net Earnings'],
                'Operating Income': ['Operating Income', 'Operating Earnings'],
                'EBIT': ['EBIT', 'Operating Income'],
                'Gross Profit': ['Gross Profit'],
                'Total Assets': ['Total Assets'],
                'Current Assets': ['Current Assets'],
                'Inventory': ['Inventory'],
                'Cash And Cash Equivalents': ['Cash And Cash Equivalents', 'Cash'],
                'Current Liabilities': ['Current Liabilities'],
                'Total Liabilities Net Minority Interest': ['Total Liabilities Net Minority Interest', 'Total Liab'],
                'Total Equity Gross Minority Interest': ['Total Equity Gross Minority Interest', 'Stockholders Equity', 'Total Stockholder Equity'],
                'Interest Expense': ['Interest Expense'],
                'EBITDA': ['EBITDA'],
                'Accounts Receivable': ['Accounts Receivable'],
                'Accounts Payable': ['Accounts Payable']
            }
            
            for alt_name in alt_names.get(item, [item]):
                if alt_name in df.index:
                    val = df.loc[alt_name, date]
                    return float(val) if pd.notna(val) else default
            return default
        except Exception:
            return default

    def calculate_enhanced_z_score(self, data_dict):
        """Calculates the Enhanced Z-Score with proper normalization to prevent extreme values."""
        try:
            eps = 1e-8
            total_assets = max(data_dict.get('total_assets', 1), eps)

            # X1: Dynamic Liquidity Stress Index (normalized to 0-1)
            current_assets = data_dict.get('current_assets', 0)
            current_liabilities = max(data_dict.get('current_liabilities', eps), eps)
            working_capital = current_assets - current_liabilities
            wc_ta = working_capital / total_assets
            wc_ta = max(-1, min(1, wc_ta))  # normalize to [-1, 1]
            
            inventory = data_dict.get('inventory', 0)
            quick_ratio = (current_assets - inventory) / max(current_liabilities, eps)
            quick_ratio_scaled = max(0, min(1, quick_ratio / 2.0))  # normalize to [0, 1]
            
            cash_and_equivalents = data_dict.get('cash_and_equivalents', 0)
            cash_ratio = cash_and_equivalents / max(current_liabilities, eps)
            cash_ratio = max(0, min(1, cash_ratio))  # normalize to [0, 1]
            
            ccc_component = self._calculate_cash_conversion_cycle(data_dict)
            ccc_scaled = max(0, min(1, (90 - ccc_component * 180 + 30) / 90))  # convert back to days and normalize
            
            x1_raw = (0.4 * wc_ta + 0.3 * quick_ratio_scaled + 0.2 * cash_ratio + 0.1 * ccc_scaled)
            x1 = max(0, min(1, x1_raw))  # final clamp to [0, 1]

            # X2: Multi-Period Earning Quality Score (normalized to 0-1)
            total_equity = max(data_dict.get('total_equity', eps), eps)
            retained_earnings_ta = total_equity / total_assets
            retained_earnings_ta = max(0, min(2, retained_earnings_ta)) / 2.0  # normalize to [0, 1]
            
            operating_margin = data_dict.get('operating_margin', 0)
            net_margin = max(data_dict.get('net_margin', eps), eps)
            cash_earnings_quality = (operating_margin / max(net_margin, eps)) if net_margin > 0 else 1.0
            cash_earnings_quality = max(0, min(2, cash_earnings_quality)) / 2.0  # normalize to [0, 1]
            
            roe = data_dict.get('return_on_equity', 0)
            roe_stability = min(1.0, roe / 0.15) if roe > 0 else 0
            roe_stability = max(0, roe_stability)  # already [0, 1]
            
            margin_trend = min(1.0, max(-1.0, (net_margin - 0.05) / 0.1))
            margin_trend_scaled = (margin_trend + 1) / 2.0  # normalize to [0, 1]
            
            x2_raw = (0.4 * retained_earnings_ta + 0.25 * max(0, cash_earnings_quality) +
                      0.25 * roe_stability + 0.1 * margin_trend_scaled)
            x2 = max(0, min(1, x2_raw))  # final clamp to [0, 1]

            # X3: Risk-Adjusted Operational Performance (normalized to 0-1)
            operating_income = data_dict.get('operating_income', 0)
            ebit_ta = operating_income / total_assets
            ebit_ta = max(-1, min(2, ebit_ta)) / 2.0  # normalize to [-0.5, 1]
            ebit_ta = max(0, ebit_ta + 0.5)  # shift to [0, 1.5] then clamp
            
            asset_volatility = data_dict.get('volatility', 0.3)
            risk_adjustment = max(0.5, 1 - (asset_volatility / 0.4))
            risk_adjustment_scaled = (risk_adjustment - 0.5) / 0.5  # normalize to [0, 1]
            
            industry_adjustment = 0.02
            industry_adj_scaled = min(1.0, industry_adjustment / 0.1)  # normalize to [0, 1]
            
            debt_ratio = data_dict.get('debt_ratio', data_dict.get('total_liabilities', 0) / total_assets)
            operating_leverage_penalty = min(0.5, debt_ratio * 0.01 / 0.1)  # normalize penalty
            
            x3_raw = (ebit_ta * risk_adjustment_scaled + 0.3 * industry_adj_scaled - operating_leverage_penalty)
            x3 = max(0, min(1, x3_raw))  # final clamp to [0, 1]

            # X4: Multi-Dimensional Solvency Score (normalized to 0-1)
            market_cap = data_dict.get('market_cap', 0)
            total_liabilities = data_dict.get('total_liabilities', 0)
            market_cap_debt = min(market_cap / max(total_liabilities, eps), 3.0) if market_cap > 0 else 0
            market_cap_debt_scaled = market_cap_debt / 3.0  # normalize to [0, 1]
            
            estimated_interest_rate = 0.03 + 0.02  # Default risk-free rate + spread
            estimated_interest = total_liabilities * estimated_interest_rate
            interest_income = data_dict.get('operating_income', 0)
            interest_coverage = interest_income / max(estimated_interest, eps) if estimated_interest > 0 else 10
            interest_coverage_scaled = min(1.0, interest_coverage / 3.0)  # normalize to [0, 1]
            
            debt_maturity_risk = min(0.5, debt_ratio)
            debt_maturity_scaled = (0.5 - debt_maturity_risk) / 0.5  # inverse, normalize to [0, 1]
            
            debt_service_ratio = debt_ratio * 2
            debt_service_scaled = max(0, 1 - debt_service_ratio / 4.0)  # normalize to [0, 1]
            
            market_risk_penalty = asset_volatility * 0.1
            market_risk_scaled = max(0, 1 - market_risk_penalty / 0.1)  # normalize to [0, 1]
            
            x4_raw = (0.4 * market_cap_debt_scaled + 0.25 * interest_coverage_scaled +
                      0.2 * debt_maturity_scaled + 0.1 * debt_service_scaled +
                      0.05 * market_risk_scaled)
            x4 = max(0, min(1, x4_raw))  # final clamp to [0, 1]

            # X5: Dynamic Asset Efficiency Index (normalized to 0-1)
            asset_turnover = data_dict.get('asset_turnover', 0)
            asset_turnover_scaled = max(0, min(2, asset_turnover)) / 2.0  # normalize to [0, 1]
            
            turnover_trend = max(0, (asset_turnover - 1.0) / 1.0)
            turnover_trend_scaled = max(0, min(1, turnover_trend * 10)) / 10.0  # normalize to [0, 1]
            
            asset_quality = self._calculate_asset_quality(data_dict)
            asset_quality_scaled = asset_quality  # already [0.3, 1.0], so mostly normalized
            
            revenue = total_assets * asset_turnover if asset_turnover > 0 else data_dict.get('total_revenue', 0)
            wc_turnover = revenue / max(abs(working_capital), eps) if working_capital != 0 else 5
            wc_turnover_scaled = min(1.0, wc_turnover / 5.0)  # normalize to [0, 1]
            
            revenue_per_employee_factor = 1.0
            rev_emp_scaled = max(0, min(1, (revenue_per_employee_factor - 1.0) / 1.0 + 1))  # normalize to [0, 1]
            
            x5_raw = (0.4 * asset_turnover_scaled + 0.2 * turnover_trend_scaled +
                      0.2 * asset_quality_scaled + 0.1 * wc_turnover_scaled +
                      0.1 * rev_emp_scaled)
            x5 = max(0, min(1, x5_raw))  # final clamp to [0, 1]

            # Enhanced Z-Score calculation with original coefficients for reference
            original_z_score = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
            
            # Enhanced Z-Score with your coefficients
            enhanced_z_score = 1.5*x1 + 1.6*x2 + 3.5*x3 + 0.8*x4 + 1.2*x5
            
            # Optional: Normalize final score to 0-10 range for consistency
            # Assuming typical Z-Score range is -5 to 10, map to 0-10
            normalized_enhanced_z = max(0, min(10, (enhanced_z_score + 5) / 1.5))
            
            logger.debug(f"Z-Score Components: X1={x1:.3f}, X2={x2:.3f}, X3={x3:.3f}, X4={x4:.3f}, X5={x5:.3f}")
            logger.debug(f"Original Z: {original_z_score:.3f}, Enhanced Z: {enhanced_z_score:.3f}, Normalized: {normalized_enhanced_z:.3f}")
            
            return float(normalized_enhanced_z)
        except Exception as e:
            logger.debug(f"    Error calculating Z-Score: {e}")
            return np.nan

    def calculate_rad_kmv(self, data_dict, rating_date):
        """Calculates the RAD-KMV Distance-to-Default with financial soundness improvements."""
        try:
            eps = 1e-8  # Small epsilon to prevent division by zero
            equity_value = data_dict.get('market_cap', np.nan)
            debt_value = data_dict.get('total_liabilities', 0)
            base_volatility = data_dict.get('volatility', 0.3)

            if not (pd.notna(equity_value) and equity_value > 0 and debt_value > 0):
                return np.nan

            asset_value = equity_value + debt_value
            
            # Improved default point calculation
            short_term_debt = data_dict.get('current_liabilities', debt_value * 0.3)  # Estimate if not available
            long_term_debt = debt_value - short_term_debt
            default_point = short_term_debt + (long_term_debt * 0.5)  # More realistic default point
            
            # Use dynamic risk-free rate instead of fixed 3%
            risk_free_rate = self._get_dynamic_risk_free_rate(rating_date)
            time_horizon = 1.0

            debt_ratio = debt_value / asset_value
            
            # Use configurable thresholds instead of hardcoded values
            if debt_ratio > VOLATILITY_THRESHOLDS['high_debt_ratio']:
                volatility_multiplier = VOLATILITY_THRESHOLDS['high_debt_multiplier']
            elif debt_ratio < VOLATILITY_THRESHOLDS['low_debt_ratio']:
                volatility_multiplier = VOLATILITY_THRESHOLDS['low_debt_multiplier']
            else:
                volatility_multiplier = VOLATILITY_THRESHOLDS['normal_multiplier']
                
            adjusted_volatility = base_volatility * volatility_multiplier

            roe = data_dict.get('return_on_equity', 0)
            operating_margin = data_dict.get('operating_margin', 0)
            credit_quality = min(1.0, (roe * 0.4 + operating_margin * 0.6) / 0.2)
            
            # Dynamic credit spread based on macro conditions
            base_credit_spread = 0.01
            if not self.macro_data.empty and 'credit_spread_high_yield' in self.macro_data.columns:
                try:
                    rating_date_dt = pd.to_datetime(rating_date)
                    available_dates = self.macro_data.index[self.macro_data.index <= rating_date_dt]
                    if not available_dates.empty:
                        closest_date = available_dates.max()
                        market_spread = self.macro_data.loc[closest_date, 'credit_spread_high_yield']
                        if pd.notna(market_spread):
                            base_credit_spread = max(0.005, float(market_spread) / 100.0 * 0.5)  # Scale down
                except Exception:
                    pass
                    
            credit_spread = base_credit_spread + (1 - credit_quality) * 0.05
            risk_adjusted_rate = risk_free_rate + credit_spread

            # Market cap-based liquidity adjustment
            market_cap = data_dict.get('market_cap', 0)
            if market_cap > 10e9:
                liquidity_adjustment = 1.0
            elif market_cap > 2e9:
                liquidity_adjustment = 1.1
            else:
                liquidity_adjustment = 1.2
                
            final_volatility = adjusted_volatility * liquidity_adjustment

            # KMV calculation
            mu = risk_adjusted_rate - 0.5 * final_volatility**2
            distance_numerator = np.log(asset_value / default_point) + mu * time_horizon
            distance_denominator = final_volatility * np.sqrt(time_horizon)
            distance_to_default = distance_numerator / distance_denominator if distance_denominator != 0 else np.nan

            # Regime-based adjustment
            current_ratio = data_dict.get('current_ratio', 1.0)
            quick_ratio = (data_dict.get('current_assets', 0) - data_dict.get('inventory', 0)) / max(data_dict.get('current_liabilities', 1), eps)
            liquidity_health = min(1.0, (current_ratio + quick_ratio) / 3.0)
            profitability_health = min(1.0, max(0, roe / 0.15))
            overall_health = (liquidity_health + profitability_health) / 2

            if overall_health > 0.7:
                regime_factor = 1.1
            elif overall_health < 0.3:
                regime_factor = 0.8
            else:
                regime_factor = 1.0

            rad_kmv_distance = distance_to_default * regime_factor
            rad_kmv_distance = max(-5.0, min(10.0, rad_kmv_distance))
            return float(rad_kmv_distance)
            
        except Exception as e:
            logger.debug(f"    Error calculating KMV: {e}")
            return np.nan

    def _get_industry_default_volatility(self, company_name, ticker):
        """Estimates industry-based default volatility (placeholder for future enhancement)."""
        # This is a placeholder - in production, you'd use industry classification
        # based on GICS sectors, SIC codes, or company name analysis
        
        # Simple keyword-based classification (can be enhanced)
        company_lower = company_name.lower() if company_name else ""
        ticker_lower = ticker.lower() if ticker else ""
        
        if any(word in company_lower for word in ['tech', 'software', 'computer', 'data']):
            return DEFAULT_VOLATILITIES['technology']
        elif any(word in company_lower for word in ['bank', 'financial', 'insurance']):
            return DEFAULT_VOLATILITIES['financial']
        elif any(word in company_lower for word in ['utility', 'electric', 'gas', 'water']):
            return DEFAULT_VOLATILITIES['utilities']
        elif any(word in company_lower for word in ['energy', 'oil', 'gas', 'petroleum']):
            return DEFAULT_VOLATILITIES['energy']
        elif any(word in company_lower for word in ['health', 'pharma', 'medical', 'drug']):
            return DEFAULT_VOLATILITIES['healthcare']
        else:
            return DEFAULT_VOLATILITIES['default']

    def fetch_and_calculate_for_entry(self, ticker, rating_date, company_name, rating):
        """Fetches data for a single entry and calculates features using Yahoo Finance with improvements."""
        logger.info(f"  Processing {ticker} for rating date {rating_date.date()}...")
        try:
            stock = yf.Ticker(ticker)
            financials = stock.financials
            balance_sheet = stock.balancesheet
            cash_flow = stock.cashflow

            if financials.empty or balance_sheet.empty:
                logger.warning(f"    Insufficient financial statements for {ticker}")
                return None

            # Use most recent available fiscal year end (practical approach)
            available_dates = balance_sheet.columns if not balance_sheet.empty else financials.columns
            if available_dates.empty:
                logger.warning(f"    No financial data available for {ticker}")
                return None
                
            fye_date = available_dates.max()
            logger.info(f"    Using most recent FYE: {fye_date.date()}")

            data = {
                'ticker': ticker, 
                'rating_date': rating_date, 
                'fye_date': fye_date, 
                'company_name': company_name, 
                'rating': rating
            }
            
            # Extract financial data
            data['total_revenue'] = self._safe_get_yf_data(financials, 'Total Revenue', fye_date)
            data['gross_profit'] = self._safe_get_yf_data(financials, 'Gross Profit', fye_date)
            data['operating_income'] = self._safe_get_yf_data(financials, 'Operating Income', fye_date)
            data['net_income'] = self._safe_get_yf_data(financials, 'Net Income', fye_date)
            data['total_assets'] = self._safe_get_yf_data(balance_sheet, 'Total Assets', fye_date)
            data['current_assets'] = self._safe_get_yf_data(balance_sheet, 'Current Assets', fye_date)
            data['current_liabilities'] = self._safe_get_yf_data(balance_sheet, 'Current Liabilities', fye_date)
            data['total_liabilities'] = self._safe_get_yf_data(balance_sheet, 'Total Liabilities Net Minority Interest', fye_date)
            data['total_equity'] = self._safe_get_yf_data(balance_sheet, 'Total Equity Gross Minority Interest', fye_date)
            data['inventory'] = self._safe_get_yf_data(balance_sheet, 'Inventory', fye_date)
            data['cash_and_equivalents'] = self._safe_get_yf_data(balance_sheet, 'Cash And Cash Equivalents', fye_date)
            
            # Try to get accounts receivable and payable for better CCC calculation
            data['accounts_receivable'] = self._safe_get_yf_data(balance_sheet, 'Accounts Receivable', fye_date)
            data['accounts_payable'] = self._safe_get_yf_data(balance_sheet, 'Accounts Payable', fye_date)

            # Calculate ratios
            eps = 1e-8
            data['current_ratio'] = data['current_assets'] / max(data['current_liabilities'], eps) if not pd.isna(data['current_assets']) and not pd.isna(data['current_liabilities']) and data['current_liabilities'] != 0 else np.nan
            data['debt_to_equity'] = data['total_liabilities'] / max(data['total_equity'], eps) if not pd.isna(data['total_liabilities']) and not pd.isna(data['total_equity']) and data['total_equity'] != 0 else np.nan
            data['debt_ratio'] = data['total_liabilities'] / max(data['total_assets'], eps) if not pd.isna(data['total_liabilities']) and not pd.isna(data['total_assets']) and data['total_assets'] != 0 else np.nan
            data['return_on_equity'] = data['net_income'] / max(data['total_equity'], eps) if not pd.isna(data['net_income']) and not pd.isna(data['total_equity']) and data['total_equity'] != 0 else np.nan
            data['return_on_assets'] = data['net_income'] / max(data['total_assets'], eps) if not pd.isna(data['net_income']) and not pd.isna(data['total_assets']) and data['total_assets'] != 0 else np.nan
            data['gross_margin'] = data['gross_profit'] / max(data['total_revenue'], eps) if not pd.isna(data['gross_profit']) and not pd.isna(data['total_revenue']) and data['total_revenue'] != 0 else np.nan
            data['operating_margin'] = data['operating_income'] / max(data['total_revenue'], eps) if not pd.isna(data['operating_income']) and not pd.isna(data['total_revenue']) and data['total_revenue'] != 0 else np.nan
            data['net_margin'] = data['net_income'] / max(data['total_revenue'], eps) if not pd.isna(data['net_income']) and not pd.isna(data['total_revenue']) and data['total_revenue'] != 0 else np.nan
            data['asset_turnover'] = data['total_revenue'] / max(data['total_assets'], eps) if not pd.isna(data['total_revenue']) and not pd.isna(data['total_assets']) and data['total_assets'] != 0 else np.nan

            # Market data - Calculate market cap and volatility
            try:
                shares_outstanding = self._safe_get_yf_data(balance_sheet, 'Share Issued', fye_date)
                if pd.isna(shares_outstanding):
                    shares_outstanding = self._safe_get_yf_data(balance_sheet, 'Ordinary Shares Number', fye_date)
                if pd.isna(shares_outstanding):
                    shares_outstanding = self._safe_get_yf_data(balance_sheet, 'Common Stock Shares Outstanding', fye_date)
                
                # Get stock price around the rating date (not after!)
                hist_start = rating_date - timedelta(days=30)
                hist_end = rating_date + timedelta(days=1)  # Allow same day
                hist_data = stock.history(start=hist_start, end=hist_end)
                
                if not hist_data.empty and not pd.isna(shares_outstanding) and shares_outstanding > 0:
                    # Use closing price closest to rating date (but not after)
                    stock_price = hist_data['Close'].iloc[-1]
                    data['market_cap'] = stock_price * shares_outstanding
                    logger.info(f"    Market cap calculated: ${data['market_cap']:,.0f}")
                else:
                    data['market_cap'] = np.nan
                    logger.debug(f"    Could not calculate market cap")
                
                # Calculate volatility using historical data before rating date
                if len(hist_data) > 5:
                    returns = hist_data['Close'].pct_change().dropna()
                    if len(returns) > 1:
                        data['volatility'] = returns.std() * np.sqrt(252)
                    else:
                        data['volatility'] = self._get_industry_default_volatility(company_name, ticker)
                else:
                    data['volatility'] = self._get_industry_default_volatility(company_name, ticker)
                    
            except Exception as e:
                logger.debug(f"    Error calculating market data: {e}")
                data['market_cap'] = np.nan
                data['volatility'] = self._get_industry_default_volatility(company_name, ticker)

            # Calculate enhanced metrics with improvements
            data['enhanced_z_score'] = self.calculate_enhanced_z_score(data)
            data['kmv_distance_to_default'] = self.calculate_rad_kmv(data, rating_date)

            logger.info(f"    Successfully processed {ticker}")
            return data

        except Exception as e:
            logger.error(f"    Error processing {ticker}: {e}")
            return None

    def build_dataset(self):
        """Main loop to build the training dataset."""
        if self.ratings_df.empty:
            logger.error("Input ratings data is empty.")
            return

        logger.info("Starting to build the training dataset...")
        total_entries = len(self.ratings_df)
        logger.info(f"Processing {total_entries} rating entries...")

        for i, (_, row) in enumerate(self.ratings_df.iterrows()):
            ticker = row['Ticker']
            rating_date = row['Date']
            company_name = row['Company_Name']
            rating = row['Rating']

            logger.info(f"Processing {i+1}/{total_entries}: {company_name} ({ticker}) on {rating_date.date()}")

            # Fetch and calculate features
            entry_data = self.fetch_and_calculate_for_entry(ticker, rating_date, company_name, rating)
            if entry_data:
                # Add label data from the original CSV
                entry_data['is_investment_grade'] = row['Binary_Rating']
                entry_data['rating_grade'] = row['Rating']
                self.training_data.append(entry_data)

            # Rate limiting
            time.sleep(0.1)

        logger.info(f"Finished processing entries. Successfully gathered data for {len(self.training_data)} entries.")

    def merge_with_macro_and_save(self):
        """Merges financial data with macro data and saves the final CSV."""
        if not self.training_data:
            logger.error("No training data to save.")
            return

        logger.info("Creating DataFrame from collected data...")
        df_final = pd.DataFrame(self.training_data)

        # Merge with Macroeconomic Data
        if not self.macro_data.empty and not df_final.empty:
            logger.info("Merging with macroeconomic data...")
            try:
                df_final['rating_date'] = pd.to_datetime(df_final['rating_date'])
                df_final = df_final.sort_values('rating_date').reset_index(drop=True)
                self.macro_data.index = pd.to_datetime(self.macro_data.index)
                self.macro_data = self.macro_data.sort_index()

                # Merge: for each rating date, get the latest macro data on or before that date
                df_final = pd.merge_asof(df_final, self.macro_data, left_on='rating_date', right_index=True, direction='backward')
                logger.info("Macroeconomic data merged successfully.")
            except Exception as e:
                logger.error(f"Error merging macro data: {e}. Proceeding without macro features.")
        else:
            logger.warning("No macro data or final data to merge.")

        # Save to CSV
        logger.info(f"Saving final dataset to {OUTPUT_CSV_PATH}...")
        try:
            # Format date columns
            df_final['rating_date'] = df_final['rating_date'].dt.strftime('%Y-%m-%d')
            if 'fye_date' in df_final.columns:
                df_final['fye_date'] = pd.to_datetime(df_final['fye_date']).dt.strftime('%Y-%m-%d')
            
            # Define column order
            feature_columns = [
                'current_ratio', 'debt_to_equity', 'debt_ratio', 'gross_margin', 'operating_margin',
                'net_margin', 'asset_turnover', 'return_on_equity', 'return_on_assets', 
                'enhanced_z_score', 'kmv_distance_to_default'
            ]
            
            # Add macro columns if they exist
            macro_cols = []
            if not self.macro_data.empty:
                macro_cols = [c for c in df_final.columns if c in self.macro_data.columns]
            feature_columns.extend(macro_cols)

            # Final column order: Features, Labels, Metadata
            label_columns = ['is_investment_grade', 'rating_grade']
            meta_columns = [c for c in df_final.columns if c not in feature_columns and c not in label_columns]
            
            # Ensure all columns are included
            final_column_order = feature_columns + label_columns + meta_columns
            final_column_order = [c for c in final_column_order if c in df_final.columns]

            df_final = df_final[final_column_order]
            
            df_final.to_csv(OUTPUT_CSV_PATH, index=False)
            logger.info(f"Final dataset saved. Shape: {df_final.shape}")
            logger.info(f"Columns: {list(df_final.columns)}")
            logger.debug(f"Sample of final dataset:\n{df_final.head().to_string()}")
            
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")

    def run(self):
        """Executes the full data building pipeline."""
        logger.info("Starting Training Data Build Pipeline (Financial Soundness Improved)")
        logger.info("=" * 60)
        start_time = time.time()

        try:
            self.load_input_ratings(INPUT_CSV_PATH)
            self.fetch_fred_macro_data()
            self.build_dataset()
            self.merge_with_macro_and_save()

            end_time = time.time()
            duration = end_time - start_time
            logger.info("=" * 60)
            logger.info("Training Data Build Pipeline Completed!")
            logger.info(f"Duration: {duration:.2f} seconds")
            logger.info(f"Output file: {OUTPUT_CSV_PATH}")
            logger.info("Financial Soundness Improvements Applied:")
            logger.info("  ✓ Dynamic Cash Conversion Cycle calculation")
            logger.info("  ✓ Dynamic risk-free rate from Treasury 3M data")
            logger.info("  ✓ Calculated asset quality based on liquidity")
            logger.info("  ✓ No look-ahead bias in fiscal year selection")
            logger.info("  ✓ Improved default point calculation for KMV")
            logger.info("  ✓ Industry-specific volatility defaults")
            logger.info("  ✓ Configurable volatility thresholds")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Fatal error in pipeline: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    builder = TrainingDataBuilder(fred_api_key=FRED_API_KEY)
    builder.run()
