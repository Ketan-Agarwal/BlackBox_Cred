import json
import numpy as np
from collections import Counter
import datetime
import logging
from neon_db_interface import NeonDBInterface
from data_collection import collect_all_data, fetch_historical_news_data
from data_processing import process_collected_data
from structured_analysis import compute_structured_score
from unstructured_analysis import compute_unstructured_score, analyze_article_sentiment, analyze_risk_factors
from fusion_engine import fuse_scores
from explainability import explain_structured_score, explain_unstructured_score, explain_fusion, generate_comprehensive_explainability_report

logger = logging.getLogger(__name__)

def run_credit_pipeline_with_db_cache(company_name, ticker, fred_api_key=None, days_back=7, max_articles=20, db_interface=None):
    """
    Enhanced pipeline with DB cache and historical backfill logic.
    db_interface: object with methods get_latest_record(ticker), save_report(ticker, report_json, timestamp)
    
    PIPELINE FLOW:
    1. Check if company exists in DB with data less than 6 hours ago - if yes, return cached result
    2. If not, run full pipeline:
       - Fetch structured data (YF + FRED) and calculate score (same for all historical periods)
       - Fetch news articles for last 28 days once
       - Calculate unstructured scores for: today, 7 days ago, 14 days ago, 28 days ago
       - Generate explainability ONLY for today's scores
       - Add comprehensive explainability report to final JSON
    """
    today = datetime.date.today()
    
    # 1. Check DB for recent record (less than 6 hours ago)
    logger.info(f"🔍 Checking database cache for {ticker}...")
    if db_interface is not None:
        record = db_interface.get_latest_record(ticker)
        if record:
            record_time = record.get('timestamp')
            if record_time:
                record_dt = datetime.datetime.fromisoformat(record_time)
                hours_diff = (datetime.datetime.now() - record_dt).total_seconds() / 3600
                if hours_diff < 6:
                    logger.info(f"✅ Found recent cached data ({hours_diff:.1f} hours old) - returning cached result")
                    return record['report']
                else:
                    logger.info(f"⏰ Cached data is {hours_diff:.1f} hours old (>6h) - running fresh analysis")

    # 2. No recent cache found - run fresh analysis
    logger.info(f"� Running fresh credit analysis pipeline for {company_name}...")
    
    # === STEP 1: FETCH STRUCTURED DATA (SAME FOR ALL PERIODS) ===
    logger.info("📊 Step 1: Fetching structured data (Yahoo Finance + FRED)...")
    collected_data = collect_all_data(
        company_name=company_name,
        ticker=ticker, 
        rating_date=today,
        fred_api_key=fred_api_key,
        news_days_back=28  # We'll fetch 28 days of news once
    )
    
    # Process structured data
    processed_data = process_collected_data(collected_data)
    struct_features = processed_data['combined_structured_features']
    
    # Calculate structured score (same for all periods)
    structured_score, structured_assessment = compute_structured_score(struct_features)
    
    # Extract market conditions
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
    
    # === STEP 2: FETCH ALL NEWS FOR LAST 28 DAYS (ONE CALL ONLY) ===
    logger.info("📰 Step 2: Fetching news articles for last 28 days...")
    news_28d = collected_data.get('news_articles', [])
    logger.info(f"📊 Fetched {len(news_28d)} articles for last 28 days - will reuse for all analysis")

    # === STEP 3: CALCULATE SCORES FOR MULTIPLE TIME PERIODS ===
    logger.info("⏰ Step 3: Calculating scores for multiple time periods...")
    periods = [
        ("today", today),
        ("7_days_ago", today - datetime.timedelta(days=7)),
        ("14_days_ago", today - datetime.timedelta(days=14)),
        ("28_days_ago", today - datetime.timedelta(days=28)),
    ]

    historical_scores = {}
    today_results = None  # Store today's results for explainability

    def filter_articles_by_window(articles, start_date, end_date):
        """Filter articles by date window"""
        filtered = []
        for art in articles:
            pub = art.get('publishedAt') or art.get('published_at')
            if not pub:
                continue
            try:
                pub_date = datetime.datetime.fromisoformat(pub[:10]).date()
            except Exception:
                continue
            if start_date <= pub_date <= end_date:
                filtered.append(art)
        return filtered

    def compute_unstructured_from_articles(articles, company_name):
        """Compute unstructured score from pre-fetched articles"""
        if not articles:
            return 50.0, {
                'unstructured_score': 50.0,
                'confidence': 0.3,
                'articles_analyzed': 0,
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
                'method': 'No news data available'
            }
        
        sentiment_results = []
        all_risk_keywords = []

        for art in articles:
            sentiment = analyze_article_sentiment(
                art['title'],
                art.get('description', ''),
                art.get('content', '')
            )
            sentiment_results.append(sentiment)
            full_text = f"{art['title']} {art.get('description', '')} {art.get('content', '')}"
            risk_analysis = analyze_risk_factors(full_text)
            all_risk_keywords.extend(risk_analysis['keywords_found'])

        net_sentiments = [result['net_sentiment'] for result in sentiment_results]
        avg_net_sentiment = np.mean(net_sentiments) if net_sentiments else 0.0
        base_sentiment_score = 50 * (1 - avg_net_sentiment)
        risk_keyword_count = len(all_risk_keywords)
        risk_boost = min(risk_keyword_count * 3, 25)
        unstructured_score = max(0, min(100, base_sentiment_score + risk_boost))

        if sentiment_results:
            avg_confidence = np.mean([r['confidence'] for r in sentiment_results])
            article_confidence = min(len(articles) / 10, 1.0)
            confidence = (avg_confidence * 0.7 + article_confidence * 0.3)
        else:
            confidence = 0.3

        sentiment_dist = {
            'positive': sum(1 for r in sentiment_results if r['net_sentiment'] > 0.1),
            'neutral': sum(1 for r in sentiment_results if -0.1 <= r['net_sentiment'] <= 0.1),
            'negative': sum(1 for r in sentiment_results if r['net_sentiment'] < -0.1)
        }

        unstructured_assessment = {
            'unstructured_score': unstructured_score,
            'confidence': confidence,
            'articles_analyzed': len(articles),
            'sentiment_distribution': sentiment_dist,
            'risk_keywords': list(Counter(all_risk_keywords).most_common(5)),
            'base_sentiment_score': base_sentiment_score,
            'risk_boost': risk_boost,
            'sample_headlines': [art['title'] for art in articles[:3]],
            'method': 'FinBERT + Risk Keyword Analysis'
        }
        
        return unstructured_score, unstructured_assessment



    for label, hist_date in periods:
        # Calculate 7-day window for each historical period
        window_end = hist_date
        window_start = hist_date - datetime.timedelta(days=6) # 7-day window

        # Filter from the SAME 28-day dataset - NO ADDITIONAL API CALLS!
        period_articles = filter_articles_by_window(news_28d, window_start, window_end)
        logger.info(f"📅 {label}: {len(period_articles)} articles from {window_start} to {window_end}")

        # Compute unstructured score using filtered articles
        unstructured_score, unstructured_assessment = compute_unstructured_from_articles(
            period_articles, company_name
        )

        # Fetch VIX for this window (for historical accuracy)
        try:
            import yfinance as yf
            vix_hist = yf.Ticker("^VIX").history(start=window_start, end=window_end + datetime.timedelta(days=1))
            if not vix_hist.empty:
                vix_value = float(vix_hist['Close'].mean())
            else:
                vix_value = market_conditions.get('vix', 20.0)
        except Exception as e:
            logger.warning(f"Could not fetch historical VIX for {hist_date}: {e}")
            vix_value = market_conditions.get('vix', 20.0)

        macro_hist = market_conditions.copy()
        macro_hist['vix'] = vix_value
        macro_hist['regime'] = _determine_market_regime(macro_hist)

        # Create proper assessment formats for fusion using the fusion engine
        structured_assessment_for_fusion = {
            'risk_score': structured_score,
            'confidence': structured_assessment.get('confidence', 0.8),
            'details': structured_assessment
        }

        unstructured_assessment_for_fusion = {
            'risk_score': unstructured_score,
            'confidence': unstructured_assessment.get('confidence', 0.5),
            'details': unstructured_assessment
        }

        # Use the proper fusion engine
        fusion_result = fuse_scores(structured_assessment_for_fusion, unstructured_assessment_for_fusion, macro_hist)
        hist_final_score = fusion_result['fused_score']
        hist_credit_grade = _score_to_credit_grade(hist_final_score)

        weights = fusion_result.get('dynamic_weights', {})
        historical_scores[label] = {
            'date': hist_date.isoformat(),
            'structured_score': structured_score,
            'unstructured_score': unstructured_score,
            'final_score': hist_final_score,
            'credit_grade': hist_credit_grade,
            'fusion_result': fusion_result,
            'weights': weights
        }
        
        # Store today's results for explainability generation
        if label == "today":
            today_results = {
                'structured_score': structured_score,
                'unstructured_score': unstructured_score,
                'final_score': hist_final_score,
                'credit_grade': hist_credit_grade,
                'structured_assessment': structured_assessment,
                'unstructured_assessment': unstructured_assessment,
                'fusion_result': fusion_result,
                'market_conditions': macro_hist,
                'struct_features': struct_features
            }

    # Ensure today_results is set, else fallback to latest available period
    if today_results is None:
        # Fallback: use the most recent period if 'today' is missing
        if periods:
            last_label, _ = periods[0]
            today_results = historical_scores.get(last_label)
        if today_results is None:
            raise ValueError("No results available for today or any period to generate explainability report.")

    # === STEP 4: GENERATE EXPLAINABILITY ONLY FOR TODAY ===
    logger.info("🔍 Step 4: Generating explainability for today's scores...")
    struct_expl = explain_structured_score(struct_features, company_name=company_name)
    unstruct_expl = explain_unstructured_score(today_results['unstructured_assessment'], company_name=company_name)
    fusion_expl = explain_fusion(
        today_results['fusion_result'], 
        today_results['structured_assessment'], 
        today_results['unstructured_assessment'], 
        company_name=company_name
    )
    # Generate comprehensive explainability report
    logger.info("📝 Generating comprehensive explainability report for today...")
    comprehensive_report = generate_comprehensive_explainability_report(
        company_name=company_name,
        final_score=today_results['final_score'],
        credit_grade=today_results['credit_grade'],
        structured_result=today_results['structured_assessment'],
        unstructured_result=today_results['unstructured_assessment'],
        fusion_result=today_results['fusion_result'],
        market_conditions=today_results['market_conditions'],
        structured_features=struct_features
    )
    logger.info(f"✅ Explainability report generated ({len(comprehensive_report)} characters)")
    
    # Print the explainability report for clarity
    print("\n" + "="*80)
    print("EXPLAINABILITY REPORT FOR TODAY")
    print("="*80)
    print(comprehensive_report)
    print("="*80 + "\n")

    # === STEP 5: FINALIZE COMPREHENSIVE REPORT ===
    logger.info("📋 Step 5: Finalizing comprehensive report...")
    
    # Generate historical scores summary string
    historical_summary = "--- Historical Scores ---\n"
    for label, hist_data in historical_scores.items():
        date = hist_data['date']
        struct_score = hist_data['structured_score']
        unstruct_score = hist_data['unstructured_score']
        final_score = hist_data['final_score']
        grade = hist_data['credit_grade']
        weights = hist_data.get('weights', {})
        struct_weight = weights.get('structured_expert', 0.5)
        unstruct_weight = weights.get('news_sentiment_expert', 0.5)
        
        historical_summary += f"{label}: {date} | Structured: {struct_score:.1f} (w={struct_weight:.2f}) | Unstructured: {unstruct_score:.1f} (w={unstruct_weight:.2f}) | Fused: {final_score:.1f} | Grade: {grade}\n"
    
    report = {
        'company_info': {
            'company': f"{company_name} ({ticker})",
            'analysis_date': today.isoformat(),
            'structured_score': f"{today_results['structured_score']:.1f}/100",
            'unstructured_score': f"{today_results['unstructured_score']:.1f}/100", 
            'final_fused_score': f"{today_results['final_score']:.1f}/100",
            'credit_grade': today_results['credit_grade']
        },
        'explainability_report': comprehensive_report,
        'fusion_explanation': fusion_expl,
        'historical_scores_summary': historical_summary,
        'historical_scores_detailed': historical_scores
    }

    # Save to database if available
    if db_interface is not None:
        logger.info("💾 Saving report to database...")
        # Use full timestamp (date and time)
        now_timestamp = datetime.datetime.now().isoformat()
        db_interface.save_report(ticker, json.dumps(report, default=str), now_timestamp)
        logger.info(f"✅ Report saved to database successfully with timestamp {now_timestamp}")

    logger.info(f"🎉 Credit pipeline completed for {company_name}! Final Score: {today_results['final_score']:.1f}, Grade: {today_results['credit_grade']}")
    return report


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
        fred_api_key=fred_api_key,
        news_days_back=days_back
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
    news_28d = collected_data.get('news_articles', [])
    unstructured_score, unstructured_assessment = compute_unstructured_score(
        company_name, days_back=days_back, max_articles=max_articles, articles=news_28d
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
    
    # Create proper assessment formats for fusion using the fusion engine
    structured_assessment_for_fusion = {
        'risk_score': structured_score,
        'confidence': structured_assessment.get('confidence', 0.8),
        'details': structured_assessment
    }

    unstructured_assessment_for_fusion = {
        'risk_score': unstructured_score,
        'confidence': unstructured_assessment.get('confidence', 0.5),
        'details': unstructured_assessment
    }
    
    # Use the proper fusion engine
    fusion_result = fuse_scores(structured_assessment_for_fusion, unstructured_assessment_for_fusion, market_conditions)
    
    # === STEP 6: EXPLAINABILITY ===
    # This aligns with "Explainability & Audit" section in architecture
    logger.info("Step 6: Generating Explainability Reports...")
    
    struct_expl = explain_structured_score(struct_features, company_name=company_name)
    unstruct_expl = explain_unstructured_score(unstructured_assessment, company_name=company_name)
    fusion_expl = explain_fusion(fusion_result, structured_assessment_for_fusion, unstructured_assessment_for_fusion, company_name=company_name)
    
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
        'comprehensive_report': comprehensive_report,
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

def _score_to_credit_grade(final_score):
    """Convert score to S&P-like credit grade (higher score = higher grade)"""
    
    if final_score >= 95:
        return 'AAA'
    elif final_score >= 90:
        return 'AA+'
    elif final_score >= 85:
        return 'AA'
    elif final_score >= 80:
        return 'AA-'
    elif final_score >= 75:
        return 'A+'
    elif final_score >= 70:
        return 'A'
    elif final_score >= 65:
        return 'A-'
    elif final_score >= 60:
        return 'BBB+'
    elif final_score >= 55:
        return 'BBB'
    elif final_score >= 50:
        return 'BBB-'
    elif final_score >= 45:
        return 'BB+'
    elif final_score >= 40:
        return 'BB'
    elif final_score >= 35:
        return 'BB-'
    elif final_score >= 30:
        return 'B+'
    elif final_score >= 25:
        return 'B'
    elif final_score >= 20:
        return 'B-'
    elif final_score >= 15:
        return 'CCC+'
    elif final_score >= 10:
        return 'CCC'
    elif final_score >= 5:
        return 'CCC-'
    elif final_score >= 1:
        return 'CC'
    else:
        return 'D'


# Main entry point: use DB-enabled orchestrator
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    fred_api_key = "1a39ebc94e4984ff4091baa2f84c0ba7"  # Your FRED API key
    db = NeonDBInterface(dsn="postgresql://neondb_owner:npg_CTOegl5r6oXV@ep-billowing-pine-adshdcaw-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require")  # Replace with your Neon DSN or set NEON_DSN env var
    result = run_credit_pipeline_with_db_cache(
        company_name="Apple Inc",
        ticker="AAPL",
        fred_api_key=fred_api_key,
        days_back=7,
        max_articles=20,
        db_interface=db
    )
    print("\n" + "="*60)
    print("CREDIT PIPELINE WITH DB CACHE RESULTS")
    print("="*60)
    print(f"Company: {result['company_info']['company']}")
    print(f"Analysis Date: {result['company_info']['analysis_date']}")
    print(f"Structured Score: {result['company_info']['structured_score']}")
    print(f"Unstructured Score: {result['company_info']['unstructured_score']}")
    print(f"Final Fused Score: {result['company_info']['final_fused_score']}")
    print(f"Credit Grade: {result['company_info']['credit_grade']}\n")
    print(f"--- Historical Scores ---")
    for label, hist in result.get('historical_scores_detailed', {}).items():
        weights = hist.get('weights', {})
        struct_w = weights.get('structured_expert', 0)
        unstruct_w = weights.get('news_sentiment_expert', 0)
        print(f"{label}: {hist['date']} | Structured: {hist['structured_score']:.1f} (w={struct_w:.2f}) | Unstructured: {hist['unstructured_score']:.1f} (w={unstruct_w:.2f}) | Fused: {hist['final_score']:.1f} | Grade: {hist['credit_grade']}")
    print("\n" + "="*60)
    db.close()