import datetime

def generate_explainability_report(company_name, final_score, alpha, threshold, ebm_output, headlines, sentiments, sentiment_scores):
    """
    Generate explainability report for credit scoring.

    Parameters:
    -----------
    final_score : float
        Final computed credit score.
    alpha : float
        Volatility factor (market stability indicator).
    threshold : float
        Threshold to decide if market is volatile or not.
    ebm_output : list of dict
        Each dict contains {"feature": str, "value": float, "contribution": float, "interpretation": str}.
    headlines : list of str
        News headlines.
    sentiments : list of str
        Sentiments corresponding to each headline (Positive/Negative/Neutral).
    sentiment_scores : list of float
        Scores corresponding to sentiment intensity.

    Returns:
    --------
    str : full formatted report
    """

    lines = []
    lines.append(f"CREDIT RISK EXPLAINABILITY REPORT for {company_name}")
    lines.append("="*60)
    lines.append("")

    # 1. Final Score
    lines.append(f"Final Credit Score")
    lines.append(f"   Overall Credit Score: {final_score}")
    lines.append("")

    # 2. Data Sources
    lines.append("Data Sources Used in Calculation")
    lines.append("   - Financial Ratios")
    lines.append("   - News Sentiment")
    lines.append("   - Volatility Factor")
    lines.append("")

    # 3. Market Volatility Influence
    lines.append("Market Volatility Influence")
    if alpha > threshold:
        lines.append(f"  → Market was LESS volatile.")
        lines.append("   → Financial fundamentals had higher influence on the score than the News sentiments.")
    else:
        lines.append("   → Market was MORE volatile.")
        lines.append("   → News sentiment had stronger influence on the score than Financial fundamentals.")
    lines.append("")

    # 4. Top Contributing Financial Factors
    lines.append("Top Contributing Financial Factors")
    # sort by abs contribution
    ebm_sorted = sorted(ebm_output, key=lambda x: abs(x['contribution']), reverse=True)[:5]
    for i, feat in enumerate(ebm_sorted, start=1):
        adjusted_contrib = feat['contribution'] * alpha
        lines.append(f"   {feat['feature']} (Value: {feat['value']:.4f})")
        lines.append(f"      Interpretation: {feat['interpretation']}")
        lines.append(f"      Contribution: {adjusted_contrib:+.4f}")
        lines.append("")
    
    lines.append("Impact of Global Sentiments on the Score")

    # 5. Pick Top 3 by Sentiment score
    headline_data = list(zip(headlines, sentiments, sentiment_scores))
    top_headlines = sorted(headline_data, key=lambda x: abs(x[2]), reverse=True)[:3]

    # global severity from alpha 
    if alpha > threshold:
        global_context = "Due to high market volatility, the impact of news was amplified.\n"
        global_emphasis = " strongly"
    else:
        global_context = "In relatively stable market conditions, the impact of news was muted.\n"
        global_emphasis = " slightly"

    lines.append(f"   {global_context}")

    for i, (headline, sentiment, score) in enumerate(top_headlines, start=1):
        # impact direction
        if sentiment.lower() == "positive":
            impact_text = "increased the credit score"
        elif sentiment.lower() == "negative":
            impact_text = "reduced the credit score"
        else:
            impact_text = "had a neutral effect on the credit score"
        
        # local severity (based on score)
        if abs(score) > 0.8:
            local_severity = "significantly (High Local Severity)"
        elif abs(score) > 0.5:
            local_severity = "moderately (Moderate Local Severity)"
        else:
            local_severity = "slightly (Low Local Severity)"
        
        # combine global + local
        if(sentiment.lower() != "neutral"):
            lines.append(
                f"   -The headline \"{headline}\" had a {sentiment.lower()} impact on the score, "
                f"which {impact_text}{global_emphasis} {local_severity}.\n"
            )
        else:
            lines.append(
                f"   -The headline \"{headline}\" had a little-to-no impact on the score, "
                f"which {impact_text}{global_emphasis} {local_severity}.\n"
            )


    # 6. Final Narrative Summary
    lines.append("Final Narrative Summary")
    if alpha > threshold:
        lines.append("   Market conditions were stable. Score was mainly driven by financial ratios.")
    else:
        lines.append("   Market was volatile. Recently surfaced headlines had stronger influence on the score.")
    lines.append("   Key strengths: Positive top financial drivers, supportive headlines.")
    lines.append("   Key risks: Negative contributions from some features, adverse headlines.")
    lines.append("")
    
    # timestamp
    lines.append(f"Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return "\n".join(lines)


# Example Usage
if __name__ == "__main__":
    final_score = "AAA"
    alpha = 0.65
    threshold = 0.7
    
    ebm_output = [
        {"feature": "Debt_to_equity", "value": 1.2, "contribution": 0.15, "interpretation": "good capital structure with moderate leverage"},
        {"feature": "Current_ratio", "value": 1.8, "contribution": 0.10, "interpretation": "good liquidity providing reasonable safety buffer"},
        {"feature": "Return_on_equity", "value": 0.12, "contribution": 0.08, "interpretation": "acceptable profitability within industry norms"},
        {"feature": "Volatility", "value": 0.25, "contribution": -0.12, "interpretation": "moderate market risk typical for established companies"},
        {"feature": "Net_margin", "value": 0.15, "contribution": 0.05, "interpretation": "strong profit margins indicating competitive advantage"},
    ]
    
    headlines = [
        "Company reports record quarterly earnings",
        "Regulatory probe into company practices",
        "Industry outlook remains stable"
    ]
    sentiments = ["Positive", "Negative", "Neutral"]
    sentiment_scores = [0.9, -0.85, 0.3]
    
    report = generate_explainability_report("company name", final_score, alpha, threshold, ebm_output, headlines, sentiments, sentiment_scores)
    print(report)
