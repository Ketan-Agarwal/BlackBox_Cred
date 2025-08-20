"""
main_pipeline.py

Main orchestrator for the modular credit scoring pipeline.
Now properly aligned with the architecture diagram for data collection and processing.
"""

from data_collection import collect_all_data
from data_processing import process_collected_data
from structured_analysis import compute_structured_score
from unstructured_analysis import compute_unstructured_score
from fusion_engine import fuse_scores
from explainability import explain_structured_score, explain_unstructured_score, explain_fusion, generate_comprehensive_explainability_report
import datetime
import logging

logger = logging.getLogger(__name__)

def run_credit_pipeline(company_name, ticker, rating_date, fred_api_key=None, days_back=7, max_articles=20):
    """
    Execute the complete credit pipeline following the architecture diagram:
    1. Data Collection (Yahoo Finance + FRED + News APIs)
    2. Data Processing (Clean & Feature Engineering) 
    3. Structured Analysis (Financial Ratios + Z-Score + KMV + EBM)
    4. Unstructured Analysis (News Sentiment + FinBERT)
    5. Dynamic Fusion (VIX-based weighting)
    6. Explainability (Interpretable results)
    """
    
    logger.info(f"=== STARTING CREDIT PIPELINE FOR {company_name} ===")
    
    # === STEP 1: DATA COLLECTION ===
    # This aligns with the "Data Collection Pipeline" section in architecture
    logger.info("Step 1: Data Collection from APIs...")
    collected_data = collect_all_data(
        company_name=company_name,
        ticker=ticker, 
        rating_date=rating_date,
        fred_api_key=fred_api_key
    )
    
    # === STEP 2: DATA PROCESSING ===  
    # This aligns with "Data Storage & Processing" section in architecture
    logger.info("Step 2: Data Processing & Feature Engineering...")
    processed_data = process_collected_data(collected_data)
    
    # === STEP 3: STRUCTURED ANALYSIS ===
    # This aligns with "Engine 1: Structured Analysis" in architecture
    logger.info("Step 3: Structured Analysis (Financial Ratios + Z-Score + KMV + EBM)...")
    
    # Use the combined structured features (financials + macro)
    struct_features = processed_data['combined_structured_features']
    structured_score, structured_assessment = compute_structured_score(struct_features)
    
    # === STEP 4: UNSTRUCTURED ANALYSIS ===
    # This aligns with "Engine 2: News Sentiment Analysis" in architecture
    logger.info("Step 4: Unstructured Analysis (News Sentiment + FinBERT)...")
    unstructured_score, unstructured_assessment = compute_unstructured_score(
        company_name, days_back=days_back, max_articles=max_articles
    )
    
    # === STEP 5: DYNAMIC FUSION ===
    # This aligns with "Dynamic Fusion & Results" section in architecture
    logger.info("Step 5: Dynamic Fusion with VIX-based weighting...")
    
    # Extract market conditions from processed macro features
    macro_features = processed_data.get('macro_features', {})
    market_conditions = {
        'vix': macro_features.get('vix', 20.0),
        'unemployment_rate': macro_features.get('unemployment_rate', 4.0), 
        'credit_spread': macro_features.get('credit_spread_high_yield', 2.0),
        'yield_curve_slope': macro_features.get('yield_curve_slope', 1.0),
        'economic_stress_index': _calculate_economic_stress_index(macro_features),
        'financial_conditions_index': _calculate_financial_conditions_index(macro_features),
        'regime': _determine_market_regime(macro_features)
    }
    
    fusion_result = fuse_scores(structured_assessment, unstructured_assessment, market_conditions)
    
    # === STEP 6: EXPLAINABILITY ===
    # This aligns with "Explainability & Audit" section in architecture
    logger.info("Step 6: Generating Explainability Reports...")
    
    struct_expl = explain_structured_score(struct_features, company_name=company_name)
    unstruct_expl = explain_unstructured_score(unstructured_assessment, company_name=company_name)
    fusion_expl = explain_fusion(fusion_result, structured_assessment, unstructured_assessment, company_name=company_name)
    
    # === STEP 7: FINAL OUTPUT ===
    final_score = fusion_result['fused_score']
    credit_grade = _score_to_credit_grade(final_score)
    
    logger.info(f"Pipeline complete! Final Score: {final_score:.1f}, Grade: {credit_grade}")
    
    # Generate comprehensive explainability report (teammate's format integrated)
    comprehensive_report = generate_comprehensive_explainability_report(
        company_name=company_name,
        final_score=final_score,
        credit_grade=credit_grade,
        structured_result=structured_assessment,
        unstructured_result=unstructured_assessment,
        fusion_result=fusion_result,
        market_conditions=market_conditions,
        structured_features=struct_features
    )
    
    return {
        'company': company_name,
        'ticker': ticker,
        'rating_date': rating_date,
        'structured_score': structured_score,
        'unstructured_score': unstructured_score,
        'final_score': final_score,
        'credit_grade': credit_grade,
        'explanations': {
            'structured': struct_expl,
            'unstructured': unstruct_expl,
            'fusion': fusion_expl
        },
        'comprehensive_report': comprehensive_report,  # Add the new comprehensive report
        'details': {
            'collected_data_summary': {
                'yahoo_finance': 'Available' if collected_data.get('yahoo_finance_data') is not None else 'Missing',
                'fred_macro': 'Available' if collected_data.get('fred_macro_data') is not None and not collected_data.get('fred_macro_data').empty else 'Missing', 
                'news_articles': len(collected_data.get('news_articles', []))
            },
            'processed_features': {
                'structured_features': len(struct_features),
                'macro_features': len(macro_features),
                'processed_news': len(processed_data.get('processed_news', []))
            },
            'market_conditions': market_conditions,
            'fusion_metadata': fusion_result
        }
    }

def _calculate_economic_stress_index(macro_features):
    """Calculate economic stress index from macro indicators"""
    try:
        vix = macro_features.get('vix', 20.0)
        unemployment = macro_features.get('unemployment_rate', 4.0)
        credit_spread = macro_features.get('credit_spread_high_yield', 2.0)
        
        # Simple stress index calculation
        stress_index = (vix - 15) * 2 + (unemployment - 3) * 10 + (credit_spread - 2) * 5
        return max(0, min(100, stress_index + 30))  # Normalize to 0-100
    except:
        return 30.0  # Default moderate stress

def _calculate_financial_conditions_index(macro_features):
    """Calculate financial conditions index from macro indicators"""
    try:
        fed_funds = macro_features.get('fed_funds_rate', 2.0)
        yield_curve = macro_features.get('yield_curve_slope', 1.0)
        
        # Simple financial conditions calculation  
        conditions_index = 50 - (fed_funds - 2) * 5 + yield_curve * 10
        return max(0, min(100, conditions_index))
    except:
        return 50.0  # Default neutral conditions

def _determine_market_regime(macro_features):
    """Determine market regime from macro indicators"""
    try:
        vix = macro_features.get('vix', 20.0)
        unemployment = macro_features.get('unemployment_rate', 4.0)
        
        if vix > 35 or unemployment > 8:
            return 'CRISIS'
        elif vix > 25 or unemployment > 6:
            return 'STRESS'
        else:
            return 'NORMAL'
    except:
        return 'NORMAL'

def _score_to_credit_grade(risk_score):
    """Convert risk score to credit grade (lower risk = higher grade)"""
    if risk_score <= 15:
        return 'AAA'
    elif risk_score <= 25:
        return 'AA'
    elif risk_score <= 35:
        return 'A'
    elif risk_score <= 45:
        return 'BBB'
    elif risk_score <= 55:
        return 'BB'
    elif risk_score <= 65:
        return 'B'
    elif risk_score <= 75:
        return 'CCC'
    else:
        return 'D'

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Test the pipeline
    fred_api_key = "1a39ebc94e4984ff4091baa2f84c0ba7"  # Your FRED API key
    
    result = run_credit_pipeline(
        company_name="Tesla Inc",
        ticker="TSLA",
        rating_date=datetime.date(2025, 8, 20),
        fred_api_key=fred_api_key,
        days_back=7,
        max_articles=10
    )
    
    print("\n" + "="*60)
    print("MODULAR CREDIT PIPELINE RESULTS")
    print("="*60)
    print(f"Company: {result['company']} ({result['ticker']})")
    print(f"Analysis Date: {result['rating_date']}")
    print(f"Structured Score: {result['structured_score']:.1f}/100")
    print(f"Unstructured Score: {result['unstructured_score']:.1f}/100") 
    print(f"Final Fused Score: {result['final_score']:.1f}/100")
    print(f"Credit Grade: {result['credit_grade']}")
    
    print(f"\n--- Data Collection Summary ---")
    summary = result['details']['collected_data_summary']
    print(f"Yahoo Finance: {summary['yahoo_finance']}")
    print(f"FRED Macro: {summary['fred_macro']}")
    print(f"News Articles: {summary['news_articles']}")
    
    print(f"\n--- Processing Summary ---")
    features = result['details']['processed_features']
    print(f"Structured Features: {features['structured_features']}")
    print(f"Macro Features: {features['macro_features']}")
    print(f"Processed News: {features['processed_news']}")
    
    print(f"\n--- Market Conditions ---") 
    mc = result['details']['market_conditions']
    print(f"VIX: {mc['vix']:.1f}")
    print(f"Market Regime: {mc['regime']}")
    print(f"Economic Stress: {mc['economic_stress_index']:.1f}")
    
    print("\n--- Sample Explanations ---")
    struct_exp = result['explanations']['structured']
    if isinstance(struct_exp, dict) and 'explanation_text' in struct_exp:
        lines = struct_exp['explanation_text'].split('\n')[:5]
        print("Structured Analysis:")
        for line in lines:
            if line.strip():
                print(f"  {line}")
    
    unstruct_exp = result['explanations']['unstructured']
    if isinstance(unstruct_exp, dict) and 'explanation' in unstruct_exp:
        lines = unstruct_exp['explanation'].split('\n')[:5]  
        print("\nUnstructured Analysis:")
        for line in lines:
            if line.strip():
                print(f"  {line}")
    
    print("\n" + "="*60)
    
    # Display the comprehensive explainability report
    if 'comprehensive_report' in result:
        print("\n" + "="*60)
        print("COMPREHENSIVE EXPLAINABILITY REPORT")
        print("="*60)
        print(result['comprehensive_report'])
        print("="*60)
