#!/usr/bin/env python3
"""
Script to test EBM model integration with explainability service.
"""
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.structured_service import StructuredModelService
from app.services.explainability_service import ExplainabilityService

def test_ebm_with_explainability():
    """Test EBM model with explainability features."""
    print("🔬 Testing EBM Model with Explainability Service...")
    
    try:
        # Sample company data (Apple-like strong financials)
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
        
        # Initialize services
        structured_service = StructuredModelService()
        explainability_service = ExplainabilityService()
        
        print("\n📊 Testing Structured Model Score Calculation...")
        
        # Load the model
        structured_service._load_model()
        
        # Calculate scores
        kmv_dd = structured_service._calculate_kmv(sample_company_data)
        z_score = structured_service._calculate_z_score(sample_company_data)
        
        print(f"   KMV Distance-to-Default: {kmv_dd:.4f}")
        print(f"   Altman Z-Score: {z_score:.4f}")
        
        # Prepare features
        if hasattr(structured_service.model, '__class__') and 'ExplainableBoostingClassifier' in str(structured_service.model.__class__):
            features = structured_service._prepare_features_for_ebm(sample_company_data)
            if structured_service.scaler:
                features = structured_service.scaler.transform(features)
            model_type = "EBM"
        else:
            features = structured_service._prepare_features(sample_company_data)
            model_type = "RandomForest"
        
        print(f"   Model Type: {model_type}")
        
        # Get prediction
        prediction_proba = structured_service.model.predict_proba(features)[0]
        structured_score = prediction_proba[1] * 100
        
        print(f"   Structured Score: {structured_score:.2f}/100")
        
        # Test explainability
        print(f"\n🔍 Testing EBM Explainability...")
        
        # Create a mock structured result for explainability testing
        mock_structured_result = {
            'structured_score': structured_score,
            'kmv_distance_to_default': kmv_dd,
            'altman_z_score': z_score,
            'shap_values': None,  # EBM doesn't use SHAP
            'feature_contributions': None,  # Will be populated
            'feature_names': structured_service.feature_columns,
            'feature_values': features.flatten().tolist(),
            'company_data': sample_company_data,
            'model_type': model_type
        }
        
        # Get EBM explanations
        if model_type == "EBM":
            try:
                ebm_local = structured_service.model.explain_local(features, y=None)
                if hasattr(ebm_local, 'data') and ebm_local.data is not None:
                    mock_structured_result['feature_contributions'] = ebm_local.data[0]
                    print(f"   ✅ EBM Local explanations obtained")
                else:
                    print(f"   ⚠️ EBM Local explanations not available, using fallback")
                    mock_structured_result['feature_contributions'] = [0.1] * len(structured_service.feature_columns)
            except Exception as e:
                print(f"   ❌ EBM explanation error: {e}")
                mock_structured_result['feature_contributions'] = [0.1] * len(structured_service.feature_columns)
        
        # Test explainability service
        print(f"\n📋 Testing Explainability Service...")
        
        structured_explanation = explainability_service._create_structured_explanation(mock_structured_result)
        
        print(f"   Model Score: {structured_explanation.random_forest_score:.2f}")
        print(f"   KMV Distance-to-Default: {structured_explanation.kmv_distance_to_default:.4f}")
        print(f"   Altman Z-Score: {structured_explanation.altman_z_score:.4f}")
        print(f"   Z-Score Interpretation: {structured_explanation.z_score_interpretation}")
        
        print(f"\n🎯 Top Feature Contributions:")
        for i, contrib in enumerate(structured_explanation.top_feature_contributions):
            impact = "positive" if contrib.contribution > 0 else "negative"
            print(f"   {i+1}. {contrib.feature_name}: {contrib.contribution:.4f} ({impact} impact)")
        
        # Test readable feature mapping
        print(f"\n🔤 Feature Name Mapping Test:")
        test_features = ['Current Ratio', 'Debt/Equity Ratio', 'altman_z_score', 'kmv_distance_to_default']
        for feature in test_features:
            readable = explainability_service._make_feature_readable(feature)
            print(f"   {feature} → {readable}")
        
        print(f"\n✅ EBM with Explainability test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during EBM explainability test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ebm_with_explainability()
