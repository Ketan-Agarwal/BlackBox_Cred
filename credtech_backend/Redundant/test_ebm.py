#!/usr/bin/env python3
"""
Script to test the trained EBM model with sample data.
"""
import sys
import os
import numpy as np

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.structured_service import StructuredModelService

def test_model_prediction():
    """Test the trained EBM model with sample company data."""
    print("🧪 Testing EBM model prediction and explainability...")
    
    try:
        # Initialize the service
        service = StructuredModelService()
        
        # Sample company data (similar to Apple's strong financial profile)
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
        
        # Calculate KMV and Z-Score
        kmv_dd = service._calculate_kmv(sample_company_data)
        z_score = service._calculate_z_score(sample_company_data)
        
        print(f"\n🔍 Calculated Financial Metrics:")
        print(f"   KMV Distance-to-Default: {kmv_dd:.4f}")
        print(f"   Altman Z-Score: {z_score:.4f}")
        
        # Load the model
        service._load_model()
        
        # Prepare features for EBM
        features = service._prepare_features_for_ebm(sample_company_data)
        if service.scaler:
            features = service.scaler.transform(features)
        
        print(f"\n🎯 Model Input Features:")
        for i, feature_name in enumerate(service.feature_columns):
            print(f"   {feature_name}: {features[0][i]:.4f}")
        
        # Get prediction
        prediction_proba = service.model.predict_proba(features)[0]
        structured_score = prediction_proba[1] * 100
        
        print(f"\n✅ Model Prediction:")
        print(f"   Investment Grade Probability: {prediction_proba[1]:.4f}")
        print(f"   Structured Credit Score: {structured_score:.2f}/100")
        
        # Grade mapping (simplified)
        if structured_score >= 90:
            grade = "AAA"
        elif structured_score >= 85:
            grade = "AA+"
        elif structured_score >= 80:
            grade = "AA"
        elif structured_score >= 75:
            grade = "A+"
        elif structured_score >= 70:
            grade = "A"
        elif structured_score >= 65:
            grade = "A-"
        elif structured_score >= 60:
            grade = "BBB+"
        elif structured_score >= 55:
            grade = "BBB"
        elif structured_score >= 50:
            grade = "BBB-"
        else:
            grade = "Below Investment Grade"
            
        print(f"   Credit Grade: {grade}")
        
        # Test explainability
        try:
            print(f"\n🔬 Model Explainability (EBM):")
            ebm_global = service.model.explain_global()
            print(f"   Global explanations available: {ebm_global is not None}")
            
            ebm_local = service.model.explain_local(features, y=None)
            print(f"   Local explanations available: {ebm_local is not None}")
            
            if hasattr(ebm_local, 'data') and ebm_local.data is not None:
                print(f"   Feature contributions shape: {np.array(ebm_local.data[0]).shape}")
                print(f"   Top 3 most important features:")
                feature_contributions = ebm_local.data[0]
                for i in range(min(3, len(feature_contributions))):
                    feature_name = service.feature_columns[i] if i < len(service.feature_columns) else f"Feature_{i}"
                    contribution = feature_contributions[i]
                    print(f"     {feature_name}: {contribution:.4f}")
            
        except Exception as e:
            print(f"   Explainability error: {e}")
        
        print(f"\n🎉 Model test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during model test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_prediction()
