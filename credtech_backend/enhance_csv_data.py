#!/usr/bin/env python3
"""
Enhanced Data Collector: Fetch missing financial data from Yahoo Finance
and add to the existing CSV for Enhanced Z-Score and RAD-KMV calculations.
"""

import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

def get_company_ticker_mapping():
    """
    Create a mapping from company names in CSV to Yahoo Finance tickers.
    This is a simplified mapping - in production, you'd have a comprehensive database.
    """
    ticker_mapping = {
        # Major companies that might be in the dataset
        'APPLE INC': 'AAPL',
        'MICROSOFT CORP': 'MSFT', 
        'AMAZON.COM INC': 'AMZN',
        'ALPHABET INC': 'GOOGL',
        'TESLA INC': 'TSLA',
        'META PLATFORMS INC': 'META',
        'BERKSHIRE HATHAWAY': 'BRK-B',
        'NVIDIA CORP': 'NVDA',
        'JOHNSON & JOHNSON': 'JNJ',
        'JPMORGAN CHASE & CO': 'JPM',
        'VISA INC': 'V',
        'PROCTER & GAMBLE': 'PG',
        'UNITEDHEALTH GROUP': 'UNH',
        'HOME DEPOT INC': 'HD',
        'MASTERCARD INC': 'MA',
        'BANK OF AMERICA CORP': 'BAC',
        'CHEVRON CORP': 'CVX',
        'ABBVIE INC': 'ABBV',
        'COCA-COLA CO': 'KO',
        'PEPSICO INC': 'PEP',
        'WALMART INC': 'WMT',
        'INTEL CORP': 'INTC',
        'CISCO SYSTEMS INC': 'CSCO',
        'PFIZER INC': 'PFE',
        'WALT DISNEY CO': 'DIS',
        'VERIZON COMMUNICATIONS': 'VZ',
        'AT&T INC': 'T',
        'EXXON MOBIL CORP': 'XOM',
        'NIKE INC': 'NKE',
        'SALESFORCE INC': 'CRM'
    }
    return ticker_mapping

def fetch_enhanced_data_for_ticker(ticker, period='1y'):
    """
    Fetch all required data for Enhanced Z-Score and RAD-KMV from Yahoo Finance.
    
    Args:
        ticker: Yahoo Finance ticker symbol
        period: Time period for data
        
    Returns:
        Dictionary with enhanced financial metrics
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get financial statements
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        info = stock.info
        
        # Get historical price data for volatility
        hist = stock.history(period=period)
        
        if hist.empty:
            return None
            
        # Calculate stock volatility (252-day annualized)
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 10 else 0.3
        
        # Extract latest financial data
        latest_data = {}
        
        # Basic financial metrics
        if not balance_sheet.empty and not income_stmt.empty:
            latest_bs = balance_sheet.iloc[:, 0]  # Most recent year
            latest_is = income_stmt.iloc[:, 0]   # Most recent year
            
            # Quick Ratio = (Current Assets - Inventory) / Current Liabilities
            current_assets = latest_bs.get('Current Assets', 0)
            inventory = latest_bs.get('Inventory', 0)
            current_liabilities = latest_bs.get('Current Liabilities', 1)
            latest_data['quick_ratio'] = (current_assets - inventory) / current_liabilities if current_liabilities > 0 else 0
            
            # Cash Ratio = Cash / Current Liabilities
            cash = latest_bs.get('Cash And Cash Equivalents', 0)
            latest_data['cash_ratio'] = cash / current_liabilities if current_liabilities > 0 else 0
            
            # Revenue (Total Revenue)
            latest_data['revenue'] = latest_is.get('Total Revenue', 0)
            
            # Operating Cash Flow
            if not cash_flow.empty:
                latest_cf = cash_flow.iloc[:, 0]
                latest_data['operating_cash_flow'] = latest_cf.get('Operating Cash Flow', 0)
            else:
                latest_data['operating_cash_flow'] = 0
                
            # Cash Conversion Cycle components
            # Days Sales Outstanding (DSO)
            accounts_receivable = latest_bs.get('Accounts Receivable', 0)
            revenue = latest_data['revenue']
            dso = (accounts_receivable * 365) / revenue if revenue > 0 else 30
            
            # Days Inventory Outstanding (DIO)
            cost_of_goods = latest_is.get('Cost Of Revenue', revenue * 0.7)  # Estimate if missing
            dio = (inventory * 365) / cost_of_goods if cost_of_goods > 0 else 30
            
            # Days Payable Outstanding (DPO)
            accounts_payable = latest_bs.get('Accounts Payable', 0)
            dpo = (accounts_payable * 365) / cost_of_goods if cost_of_goods > 0 else 20
            
            # Cash Conversion Cycle = DSO + DIO - DPO
            latest_data['cash_conversion_cycle'] = dso + dio - dpo
            
        # Market data
        latest_data['stock_volatility'] = volatility
        latest_data['beta'] = info.get('beta', 1.0)
        latest_data['market_cap'] = info.get('marketCap', 0)
        
        # Trading volume (average 30-day)
        if len(hist) >= 30:
            latest_data['avg_volume'] = hist['Volume'].tail(30).mean()
        else:
            latest_data['avg_volume'] = hist['Volume'].mean() if not hist.empty else 0
            
        # Market liquidity proxy (volume * price)
        latest_data['market_liquidity'] = latest_data['avg_volume'] * hist['Close'].iloc[-1] if not hist.empty else 0
        
        return latest_data
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def enhance_csv_with_yahoo_data(csv_file_path, output_file_path=None):
    """
    Enhance the existing CSV with Yahoo Finance data.
    
    Args:
        csv_file_path: Path to existing CSV file
        output_file_path: Path for enhanced CSV (if None, overwrites original)
    """
    # Load existing CSV
    print("📊 Loading existing CSV data...")
    df = pd.read_csv(csv_file_path)
    print(f"   Loaded {len(df)} companies")
    
    # Get ticker mapping
    ticker_mapping = get_company_ticker_mapping()
    
    # Initialize new columns
    new_columns = [
        'quick_ratio_yf', 'cash_ratio_yf', 'revenue_yf', 'operating_cash_flow_yf',
        'cash_conversion_cycle_yf', 'stock_volatility_yf', 'beta_yf', 
        'avg_volume_yf', 'market_liquidity_yf', 'data_source'
    ]
    
    for col in new_columns:
        df[col] = np.nan
    
    df['data_source'] = 'CSV_ONLY'  # Default value
    
    # Process companies with known tickers
    print("\n🔍 Fetching Yahoo Finance data...")
    successful_fetches = 0
    
    for idx, row in df.iterrows():
        company_name = str(row.get('Company', '')).upper().strip()
        
        # Try to find ticker
        ticker = None
        for name_pattern, stock_ticker in ticker_mapping.items():
            if name_pattern in company_name or company_name in name_pattern:
                ticker = stock_ticker
                break
        
        if ticker:
            print(f"   Fetching data for {company_name} ({ticker})...")
            
            # Fetch enhanced data
            enhanced_data = fetch_enhanced_data_for_ticker(ticker)
            
            if enhanced_data:
                # Update DataFrame
                df.loc[idx, 'quick_ratio_yf'] = enhanced_data.get('quick_ratio', np.nan)
                df.loc[idx, 'cash_ratio_yf'] = enhanced_data.get('cash_ratio', np.nan)
                df.loc[idx, 'revenue_yf'] = enhanced_data.get('revenue', np.nan)
                df.loc[idx, 'operating_cash_flow_yf'] = enhanced_data.get('operating_cash_flow', np.nan)
                df.loc[idx, 'cash_conversion_cycle_yf'] = enhanced_data.get('cash_conversion_cycle', np.nan)
                df.loc[idx, 'stock_volatility_yf'] = enhanced_data.get('stock_volatility', np.nan)
                df.loc[idx, 'beta_yf'] = enhanced_data.get('beta', np.nan)
                df.loc[idx, 'avg_volume_yf'] = enhanced_data.get('avg_volume', np.nan)
                df.loc[idx, 'market_liquidity_yf'] = enhanced_data.get('market_liquidity', np.nan)
                df.loc[idx, 'data_source'] = 'YAHOO_FINANCE'
                
                successful_fetches += 1
                print(f"     ✅ Success!")
            else:
                print(f"     ❌ Failed to fetch data")
            
            # Rate limiting
            time.sleep(0.1)  # Small delay to avoid overwhelming Yahoo Finance
        
        # Progress indicator
        if (idx + 1) % 10 == 0:
            print(f"   Processed {idx + 1}/{len(df)} companies...")
    
    print(f"\n📈 Enhanced data successfully fetched for {successful_fetches} companies")
    
    # Fill missing values with estimates for companies without Yahoo data
    print("\n🔧 Filling missing values with estimates...")
    
    # Quick ratio estimation (typically 70-80% of current ratio)
    df['quick_ratio_yf'].fillna(df['Current Ratio'] * 0.75, inplace=True)
    
    # Cash ratio estimation (typically 20-30% of current ratio)
    df['cash_ratio_yf'].fillna(df['Current Ratio'] * 0.25, inplace=True)
    
    # Cash conversion cycle estimation (varies by industry, default 45 days)
    df['cash_conversion_cycle_yf'].fillna(45, inplace=True)
    
    # Stock volatility estimation (default 30% annually)
    df['stock_volatility_yf'].fillna(0.30, inplace=True)
    
    # Beta estimation (default 1.0 - market average)
    df['beta_yf'].fillna(1.0, inplace=True)
    
    # Revenue estimation from asset turnover
    estimated_revenue = df['Asset Turnover'] * df.get('Total Assets', 0)
    df['revenue_yf'].fillna(estimated_revenue, inplace=True)
    
    # Operating cash flow estimation (typically 80-120% of net income)
    estimated_ocf = df['Net Profit Margin'] * df['revenue_yf'] * 0.9 / 100
    df['operating_cash_flow_yf'].fillna(estimated_ocf, inplace=True)
    
    # Market liquidity estimation (lower for companies without Yahoo data)
    df['market_liquidity_yf'].fillna(1000000, inplace=True)  # $1M default
    df['avg_volume_yf'].fillna(100000, inplace=True)  # 100K shares default
    
    # Save enhanced CSV
    output_path = output_file_path or csv_file_path.replace('.csv', '_enhanced.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\n💾 Enhanced CSV saved to: {output_path}")
    print(f"   Total companies: {len(df)}")
    print(f"   With Yahoo Finance data: {successful_fetches}")
    print(f"   With estimated data: {len(df) - successful_fetches}")
    
    return output_path

def main():
    """Main function to enhance the CSV with Yahoo Finance data."""
    csv_file_path = "corporateCreditRatingWithFinancialRatios.csv"
    
    try:
        enhanced_csv_path = enhance_csv_with_yahoo_data(csv_file_path)
        print(f"\n🎉 Enhancement complete! Enhanced file: {enhanced_csv_path}")
        
        # Display sample of enhanced data
        df = pd.read_csv(enhanced_csv_path)
        print(f"\n📋 Sample enhanced data:")
        enhanced_cols = ['Company', 'quick_ratio_yf', 'cash_ratio_yf', 'stock_volatility_yf', 'data_source']
        if all(col in df.columns for col in enhanced_cols):
            print(df[enhanced_cols].head())
        
    except Exception as e:
        print(f"❌ Error during enhancement: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
