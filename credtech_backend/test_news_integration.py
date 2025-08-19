"""
test_news_integration.py

Complete integration test showing how news_unstructured_score.py and news_explainability.py work together.
This demonstrates the full pipeline from news fetching to detailed explanations.
"""

import sys
import traceback
from news_explainability import get_company_news_explanation, explain_news_assessment
from news_unstructured_score import get_news_risk_assessment


def test_individual_components():
    """Test each component separately"""
    print("🧪 TESTING INDIVIDUAL COMPONENTS")
    print("="*50)
    
    company = "Apple"
    
    # Test 1: News risk assessment only
    print(f"\n1️⃣ Testing news risk assessment for {company}...")
    try:
        news_result = get_news_risk_assessment(company, days_back=5, max_articles=10)
        print(f"✅ News assessment successful")
        print(f"   Risk Score: {news_result.get('risk_score', 'N/A')}")
        print(f"   Confidence: {news_result.get('confidence', 'N/A'):.1%}")
        print(f"   Articles: {news_result.get('detailed_analysis', {}).get('articles_analyzed', 0)}")
        
        # Test 2: Explanation generation
        print(f"\n2️⃣ Testing explanation generation...")
        explanation_result = explain_news_assessment(news_result)
        print(f"✅ Explanation generation successful")
        print(f"   Risk Level: {explanation_result.get('risk_level', 'N/A')}")
        print(f"   Insights: {len(explanation_result.get('actionable_insights', []))} generated")
        
        return news_result, explanation_result
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        traceback.print_exc()
        return None, None


def test_integrated_pipeline():
    """Test the integrated pipeline"""
    print("\n🔗 TESTING INTEGRATED PIPELINE")
    print("="*50)
    
    test_companies = ["Apple", "Tesla", "Microsoft"]
    
    for company in test_companies:
        print(f"\n🏢 Testing {company}...")
        try:
            # Use the integrated function
            complete_result = get_company_news_explanation(company, days_back=7, max_articles=15)
            
            if 'error' not in complete_result:
                news_data = complete_result['news_assessment']
                explanation_data = complete_result['explanation_analysis']
                
                print(f"✅ Integration successful for {company}")
                print(f"   📊 Risk Score: {news_data.get('risk_score', 0):.1f}/100")
                print(f"   🎯 Confidence: {news_data.get('confidence', 0):.1%}")
                print(f"   📰 Articles: {news_data.get('detailed_analysis', {}).get('articles_analyzed', 0)}")
                print(f"   💡 Insights: {len(explanation_data.get('actionable_insights', []))}")
                
                # Show first insight
                insights = explanation_data.get('actionable_insights', [])
                if insights:
                    print(f"   📝 Key Insight: {insights[0]}")
                
            else:
                print(f"❌ Integration failed for {company}: {complete_result.get('error')}")
                
        except Exception as e:
            print(f"❌ Error testing {company}: {e}")
        
        print("-" * 30)


def test_detailed_explanation():
    """Test detailed explanation output"""
    print("\n📋 TESTING DETAILED EXPLANATION OUTPUT")
    print("="*50)
    
    try:
        # Get a complete analysis
        result = get_company_news_explanation("Tesla", days_back=7, max_articles=10)
        
        if 'error' not in result:
            explanation_data = result['explanation_analysis']
            
            print("📄 FULL EXPLANATION:")
            print(explanation_data.get('explanation', 'No explanation available'))
            
            print(f"\n📊 SENTIMENT ANALYSIS:")
            sentiment = explanation_data.get('sentiment_analysis', {})
            print(f"   Overall: {sentiment.get('overall_sentiment', 'N/A')}")
            print(f"   Impact: {sentiment.get('impact', 'N/A')}")
            
            print(f"\n⚠️ RISK FACTORS:")
            risk_factors = explanation_data.get('risk_factor_analysis', {})
            concerns = risk_factors.get('primary_concerns', [])
            for concern in concerns:
                print(f"   • {concern.get('name', 'N/A')}: {concern.get('mentions', 0)} mentions")
            
            print(f"\n💡 ACTIONABLE INSIGHTS:")
            insights = explanation_data.get('actionable_insights', [])
            for insight in insights:
                print(f"   • {insight}")
                
        else:
            print(f"❌ Could not generate detailed explanation: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Detailed explanation test failed: {e}")
        traceback.print_exc()


def main():
    """Run all integration tests"""
    print("🚀 NEWS SENTIMENT ANALYSIS - INTEGRATION TESTS")
    print("="*60)
    
    # Test individual components
    news_result, explanation_result = test_individual_components()
    
    # Test integrated pipeline
    test_integrated_pipeline()
    
    # Test detailed explanation
    test_detailed_explanation()
    
    print("\n✅ ALL INTEGRATION TESTS COMPLETED")
    print("="*60)
    
    # Usage examples
    print("\n📚 USAGE EXAMPLES:")
    print("="*30)
    print("# Method 1: Use integrated function")
    print("from news_explainability import get_company_news_explanation")
    print("result = get_company_news_explanation('Apple', days_back=7, max_articles=15)")
    print("print(result['explanation_analysis']['explanation'])")
    print()
    print("# Method 2: Use components separately")
    print("from news_unstructured_score import get_news_risk_assessment")
    print("from news_explainability import explain_news_assessment")
    print("news_data = get_news_risk_assessment('Apple')")
    print("explanation = explain_news_assessment(news_data)")
    print("print(explanation['explanation'])")


if __name__ == "__main__":
    main()
