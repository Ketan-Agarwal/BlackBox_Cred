#!/usr/bin/env python3
"""
Script to test the EBM model with detailed numeric explanations.
"""
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.structured_service import StructuredModelService

def test_numeric_explanations():
    """Test the EBM model with detailed numeric explanations."""
    print("🧪 Testing EBM model with detailed numeric explanations...")
    
    try:
        # Initialize the service
        service = StructuredModelService()
        
        # Sample company data (Apple-like strong financial profile)
        sample_company_data = {
            'current_ratio': 1.07,
            'debt_to_equity': 1.69,
            'return_on_equity': 0.175,  # 17.5%
            'return_on_assets': 0.125,  # 12.5%
            'operating_margin': 0.30,   # 30%
            'net_margin': 0.25,         # 25%
            'asset_turnover': 1.12,
            'inventory_turnover': 15.5,
            'total_assets': 365000000000,
            'total_liabilities': 290000000000,
            'total_equity': 75000000000,
            'market_cap': 3000000000000,
            'operating_income': 115000000000,
            'volatility': 0.25
        }
        
        print("\n📊 Sample Company Financial Data:")
        print(f"   Current Ratio: {sample_company_data['current_ratio']}")
        print(f"   Debt/Equity: {sample_company_data['debt_to_equity']}")
        print(f"   ROE: {sample_company_data['return_on_equity']*100:.1f}%")
        print(f"   ROA: {sample_company_data['return_on_assets']*100:.1f}%")
        print(f"   Operating Margin: {sample_company_data['operating_margin']*100:.1f}%")
        print(f"   Net Margin: {sample_company_data['net_margin']*100:.1f}%")
        
        # Load the model
        service._load_model()
        
        # Prepare features for EBM
        features = service._prepare_features_for_ebm(sample_company_data)
        if service.scaler:
            features = service.scaler.transform(features)
        
        # Get prediction
        prediction_proba = service.model.predict_proba(features)[0]
        structured_score = prediction_proba[1] * 100
        
        print(f"\n🎯 Model Prediction:")
        print(f"   Structured Credit Score: {structured_score:.2f}/100")
        print(f"   Investment Grade Probability: {prediction_proba[1]:.4f}")
        
        # Test detailed feature impacts
        print(f"\n🔍 Calculating detailed feature impacts...")
        feature_impacts = service._calculate_feature_impacts(features, sample_company_data)
        
        if 'error' in feature_impacts:
            print(f"   Error: {feature_impacts['error']}")
        else:
            print(f"   Current Score: {feature_impacts['current_score']:.2f}")
            print(f"   Baseline Score: {feature_impacts['baseline_score']:.2f}")
            print(f"   Total Impact: {feature_impacts['total_impact']:.2f}")
            
            print(f"\n📋 Individual Feature Impacts:")
            for feature_name, impact_data in feature_impacts['feature_impacts'].items():
                impact_points = impact_data['impact_points']
                direction = "increase" if impact_points > 0 else "decrease"
                print(f"   {feature_name}: {direction} of {abs(impact_points):.2f} points")
        
        # Test numeric explanations
        print(f"\n📝 Human-Readable Numeric Explanations:")
        numeric_explanations = service._generate_numeric_explanations(sample_company_data, feature_impacts)
        
        for i, explanation in enumerate(numeric_explanations, 1):
            print(f"   {i}. {explanation}")
        
        print(f"\n🎉 Numeric explanations test completed successfully!")
        
        # Test with a weaker company profile
        print(f"\n" + "="*60)
        print(f"🧪 Testing with a weaker company profile...")
        
        weak_company_data = {
            'current_ratio': 0.8,   # Poor liquidity
            'debt_to_equity': 3.5,  # High debt
            'return_on_equity': 0.05,  # 5% ROE - poor
            'return_on_assets': 0.02,  # 2% ROA - poor
            'operating_margin': 0.05,  # 5% - poor
            'net_margin': 0.02,     # 2% - poor
            'asset_turnover': 0.5,  # Low efficiency
            'inventory_turnover': 3.0,
            'total_assets': 100000000000,
            'total_liabilities': 90000000000,
            'total_equity': 10000000000,
            'market_cap': 50000000000,
            'operating_income': 5000000000,
            'volatility': 0.45
        }
        
        print(f"\n📊 Weak Company Financial Data:")
        print(f"   Current Ratio: {weak_company_data['current_ratio']}")
        print(f"   Debt/Equity: {weak_company_data['debt_to_equity']}")
        print(f"   ROE: {weak_company_data['return_on_equity']*100:.1f}%")
        print(f"   ROA: {weak_company_data['return_on_assets']*100:.1f}%")
        print(f"   Operating Margin: {weak_company_data['operating_margin']*100:.1f}%")
        print(f"   Net Margin: {weak_company_data['net_margin']*100:.1f}%")
        
        # Prepare features for weak company
        weak_features = service._prepare_features_for_ebm(weak_company_data)
        if service.scaler:
            weak_features = service.scaler.transform(weak_features)
        
        # Get prediction for weak company
        weak_prediction_proba = service.model.predict_proba(weak_features)[0]
        weak_structured_score = weak_prediction_proba[1] * 100
        
        print(f"\n🎯 Weak Company Prediction:")
        print(f"   Structured Credit Score: {weak_structured_score:.2f}/100")
        print(f"   Investment Grade Probability: {weak_prediction_proba[1]:.4f}")
        
        # Get feature impacts for weak company
        weak_feature_impacts = service._calculate_feature_impacts(weak_features, weak_company_data)
        weak_numeric_explanations = service._generate_numeric_explanations(weak_company_data, weak_feature_impacts)
        
        print(f"\n📝 Weak Company Numeric Explanations:")
        for i, explanation in enumerate(weak_numeric_explanations, 1):
            print(f"   {i}. {explanation}")
        
        print(f"\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_numeric_explanations()
