
"""
structured_analysis.py

Structured risk analysis using trained EBM model - simplified implementation.
"""

import logging
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def compute_structured_score(processed_features: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Compute structured score using trained EBM model.
    Just takes classifier output and scales to 0-100.
    Returns: (structured_score, assessment_dict)
    """
    try:
        logger.info("Computing structured score...")
        # Load the model
        MODEL_PATH = r"C:\\Users\\asus\\Documents\\GitHub\\BlackBox_Cred\\BlackBox_Backend\\model\\ebm_model_struct_score.pkl"
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        ebm_model = model_data['model']
        scaler = model_data['scaler']
        feature_columns = model_data['feature_columns']
        # Prepare features
        feature_df = pd.DataFrame([processed_features])
        available_features = [col for col in feature_columns if col in feature_df.columns]
        feature_df = feature_df[available_features].fillna(0).replace([np.inf, -np.inf], 0)
        # Scale and predict
        features_scaled = scaler.transform(feature_df)
        prediction_proba = ebm_model.predict_proba(features_scaled)[0]
        # Take classifier output and scale to 0-100
        # prediction_proba[0] = probability of Non-Investment Grade 
        structured_score = prediction_proba[1]*100
        
        # Get feature contributions for explainability
        feature_contributions = {}
        try:
            # Try to get local explanations from EBM
            local_explanation = ebm_model.explain_local(features_scaled)
            local_data = local_explanation.data()
            
            if hasattr(local_data, 'scores') and len(local_data.scores) > 0:
                # Get feature contributions for the first instance
                scores = local_data.scores[0]  # First instance
                for i, feature_name in enumerate(available_features):
                    if i < len(scores):
                        feature_contributions[feature_name] = scores[i]
                    else:
                        feature_contributions[feature_name] = 0.0
            else:
                # Fallback: use feature values as proxy contributions
                logger.warning("Could not get EBM local explanations, using feature values as proxy")
                for feature_name in available_features:
                    feature_value = feature_df.iloc[0][feature_name]
                    # Normalize feature contribution based on value magnitude
                    feature_contributions[feature_name] = float(feature_value) * 0.01
        
        except Exception as e:
            logger.warning(f"Could not extract feature contributions: {e}")
            # Fallback: create dummy contributions
            for feature_name in available_features:
                feature_value = feature_df.iloc[0][feature_name]
                feature_contributions[feature_name] = float(feature_value) * 0.01
        
        logger.info(f"Structured Score: {structured_score:.1f}")
        assessment = {
            'structured_score': structured_score,
            'features_used': len(available_features),
            'feature_contributions': feature_contributions
        }
        return structured_score, assessment
    except Exception as e:
        logger.error(f"Error in structured analysis: {e}")
        assessment = {
            'structured_score': 50.0,
            'error': str(e)
        }
        return 50.0, assessment
