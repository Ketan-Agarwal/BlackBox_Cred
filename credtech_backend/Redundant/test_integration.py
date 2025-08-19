#!/usr/bin/env python3
"""
Integration test for the complete explainability system.
"""
import sys
import os
import json

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.structured_service import StructuredModelService
from app.services.explainability_service import ExplainabilityService

def test_complete_system():
    """Test the complete explainability system integration."""
    print("🧪 Testing complete explainability system integration...")
    
    try:
        # Initialize services
        structured_service = StructuredModelService()
        explainability_service = ExplainabilityService()
        
        # Sample company data
        sample_company_data = {
            'current_ratio': 1.07,
            'debt_to_equity': 1.69,
            'return_on_equity': 0.175,
            'return_on_assets': 0.125,
            'operating_margin': 0.30,
            'net_margin': 0.25,
            'asset_turnover': 1.12,
            'inventory_turnover': 15.5,
            'total_assets': 365000000000,
            'total_liabilities': 290000000000,
            'total_equity': 75000000000,
            'market_cap': 3000000000000,
            'operating_income': 115000000000,
            'volatility': 0.25
        }
        
        print("📊 Testing complete scoring and explanation flow...")
        
        # Get structured credit score
        print("1. Getting structured credit score...")
        structured_result = structured_service.predict(sample_company_data)
        
        print(f"   Structured Score: {structured_result['score']:.2f}/100")
        print(f"   Credit Grade: {structured_result['grade']}")
        print(f"   Investment Grade: {structured_result['investment_grade']}")
        
        # Check if explanations are included
        if 'explanations' in structured_result:
            print(f"   ✅ Explanations included in structured result")
            
            # Display explanations
            explanations = structured_result['explanations']
            print(f"\n📝 Structured Explanations ({len(explanations)} total):")
            for i, explanation in enumerate(explanations[:5], 1):  # Show first 5
                print(f"   {i}. {explanation}")
            if len(explanations) > 5:
                print(f"   ... and {len(explanations) - 5} more explanations")
        else:
            print(f"   ⚠️  No explanations in structured result")
        
        # Test explainability service
        print(f"\n2. Testing explainability service...")
        
        # Create explanation request
        explanation_request = {
            'structured_score': structured_result['score'],
            'unstructured_score': 85.0,  # Mock unstructured score
            'overall_score': (structured_result['score'] + 85.0) / 2,
            'structured_explanations': structured_result.get('explanations', []),
            'unstructured_explanations': [
                "Positive sentiment in recent financial reports increased score by 8 points",
                "Management commentary shows confidence, adding 5 points to rating"
            ]
        }
        
        comprehensive_explanation = explainability_service.generate_comprehensive_explanation(explanation_request)
        
        print(f"   ✅ Comprehensive explanation generated")
        print(f"   Overall Rating: {comprehensive_explanation['overall_rating']}")
        print(f"   Investment Grade: {comprehensive_explanation['investment_grade']}")
        
        # Display structured explanations
        print(f"\n📊 Structured Analysis:")
        for explanation in comprehensive_explanation['structured_explanations'][:3]:
            print(f"   • {explanation}")
        
        # Display unstructured explanations
        print(f"\n📄 Unstructured Analysis:")
        for explanation in comprehensive_explanation['unstructured_explanations']:
            print(f"   • {explanation}")
        
        # Display summary
        print(f"\n📋 Executive Summary:")
        print(f"   {comprehensive_explanation['summary']}")
        
        print(f"\n🎉 Complete system integration test successful!")
        
    except Exception as e:
        print(f"❌ Error during integration test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_complete_system()
