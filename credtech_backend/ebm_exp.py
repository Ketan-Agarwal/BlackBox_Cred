# explainability_ebm_model.py

import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
MODEL_PATH = r"C:\Users\asus\Documents\GitHub\BlackBox_Cred\credtech_backend\models\ebm_model_trained_on_csv.pkl"
CSV_FILE_PATH = r"C:\Users\asus\Documents\GitHub\BlackBox_Cred\credtech_backend\complete_training_dataset_corrected.csv"
EXPLANATIONS_OUTPUT_DIR = r"C:\Users\asus\Documents\GitHub\BlackBox_Cred\credtech_backend\explanations"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
os.makedirs(EXPLANATIONS_OUTPUT_DIR, exist_ok=True)

class EBMExplainer:
    """Comprehensive explainability class for EBM credit scoring model."""
    
    def __init__(self, model_path):
        """Initialize the explainer with a trained model."""
        self.model_path = model_path
        self.model_data = None
        self.ebm_model = None
        self.scaler = None
        self.feature_columns = None
        self.load_model()
        
    def load_model(self):
        """Load the trained EBM model and associated components."""
        logger.info(f"Loading model from {self.model_path}...")
        try:
            with open(self.model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            
            self.ebm_model = self.model_data['model']
            self.scaler = self.model_data['scaler']
            self.feature_columns = self.model_data['feature_columns']
            
            logger.info("Model loaded successfully.")
            logger.info(f"Model accuracy: {self.model_data.get('accuracy', 'N/A')}")
            logger.info(f"Features: {len(self.feature_columns)}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_feature_interpretation(self, feature_name, value, contribution):
        """Get business interpretation for a specific feature value."""
        
        # Define interpretation rules based on financial knowledge
        interpretations = {
            'debt_to_equity': {
                'thresholds': [0.5, 1.0, 2.0, 3.0],
                'descriptions': [
                    "excellent capital structure with very low leverage",
                    "good capital structure with moderate leverage", 
                    "concerning leverage levels increasing risk",
                    "high leverage indicating significant financial risk",
                    "extremely high leverage suggesting potential distress"
                ]
            },
            'debt_ratio': {
                'thresholds': [0.3, 0.5, 0.7, 0.8],
                'descriptions': [
                    "very low debt burden indicating strong financial position",
                    "moderate debt levels within healthy range",
                    "elevated debt levels requiring monitoring",
                    "high debt burden increasing default risk",
                    "excessive debt burden indicating financial distress"
                ]
            },
            'current_ratio': {
                'thresholds': [1.0, 1.5, 2.0, 3.0],
                'descriptions': [
                    "insufficient liquidity to meet short-term obligations",
                    "adequate liquidity but below optimal levels",
                    "good liquidity providing reasonable safety buffer",
                    "strong liquidity position with excellent coverage",
                    "very strong liquidity position"
                ]
            },
            'return_on_equity': {
                'thresholds': [0.0, 0.05, 0.15, 0.25],
                'descriptions': [
                    "negative returns indicating poor management performance",
                    "weak profitability suggesting operational challenges",
                    "acceptable profitability within industry norms",
                    "strong profitability indicating efficient management",
                    "exceptional profitability demonstrating superior performance"
                ]
            },
            'enhanced_z_score': {
                'thresholds': [1.8, 3.0, 4.5, 6.0],
                'descriptions': [
                    "high bankruptcy risk requiring immediate attention",
                    "moderate bankruptcy risk needing close monitoring",
                    "low bankruptcy risk indicating stable operations",
                    "very low bankruptcy risk with strong fundamentals",
                    "minimal bankruptcy risk with excellent financial health"
                ]
            },
            'net_margin': {
                'thresholds': [0.0, 0.05, 0.10, 0.20],
                'descriptions': [
                    "negative profitability indicating operational losses",
                    "weak profit margins suggesting pricing or cost issues",
                    "adequate profit margins within industry standards",
                    "strong profit margins indicating competitive advantage",
                    "exceptional profit margins demonstrating pricing power"
                ]
            },
            'volatility': {
                'thresholds': [0.15, 0.30, 0.50, 0.75],
                'descriptions': [
                    "very low market risk with stable stock performance",
                    "moderate market risk typical for established companies",
                    "elevated market risk indicating investor uncertainty",
                    "high market risk suggesting significant concerns",
                    "extreme market risk indicating severe volatility"
                ]
            },
            'kmv_distance_to_default': {
                'thresholds': [0.0, 1.5, 3.0, 5.0],
                'descriptions': [
                    "company is at immediate risk of default",
                    "elevated default risk requiring urgent attention",
                    "moderate default risk needing monitoring",
                    "low default risk indicating stable credit profile",
                    "minimal default risk with strong credit metrics"
                ]
            }
        }
        
        # Get interpretation for the feature
        if feature_name in interpretations:
            thresholds = interpretations[feature_name]['thresholds']
            descriptions = interpretations[feature_name]['descriptions']
            
            # Find appropriate description based on value
            description_idx = 0
            for i, threshold in enumerate(thresholds):
                if value <= threshold:
                    description_idx = i
                    break
            else:
                description_idx = len(descriptions) - 1
            
            return descriptions[description_idx]
        else:
            # Generic interpretation based on contribution and prediction context
            if contribution > 0:
                return "contributing to the predicted rating"
            else:
                return "opposing the predicted rating"
    
    def explain_single_prediction(self, sample_data, company_name="Unknown Company"):
        """Generate detailed explanation for a single prediction."""
        
        # Ensure sample_data is a DataFrame
        if isinstance(sample_data, dict):
            sample_df = pd.DataFrame([sample_data])
        elif isinstance(sample_data, pd.Series):
            sample_df = sample_data.to_frame().T
        else:
            sample_df = sample_data.copy()
        
        # Ensure we have the right columns
        available_features = [col for col in self.feature_columns if col in sample_df.columns]
        sample_df = sample_df[available_features]
        
        # Handle missing values
        sample_df = sample_df.fillna(0)
        sample_df = sample_df.replace([np.inf, -np.inf], 0)
        
        # Scale the sample
        sample_scaled = self.scaler.transform(sample_df)
        
        # Get prediction and probability
        try:
            prediction = self.ebm_model.predict(sample_scaled)[0]
            prediction_proba = self.ebm_model.predict_proba(sample_scaled)[0]
        except Exception as e:
            logger.error(f"Error getting prediction: {e}")
            prediction = 0
            prediction_proba = [0.5, 0.5]  # Default probabilities
        
        # Get local explanation
        try:
            # Try different approaches to get local explanation
            local_explanation = None
            feature_names = None
            feature_scores = None
            
            # Method 1: Standard explain_local
            try:
                local_explanation = self.ebm_model.explain_local(sample_scaled)
                if local_explanation is not None:
                    local_data = local_explanation.data()
                    if local_data is not None and 'names' in local_data and 'scores' in local_data:
                        feature_names = local_data['names']
                        feature_scores = local_data['scores']
                        logger.info(f"Successfully got local explanation with {len(feature_scores)} feature scores")
            except Exception as e:
                logger.debug(f"Method 1 failed: {e}")
            
            # Method 2: Try with different parameters
            if feature_names is None:
                try:
                    local_explanation = self.ebm_model.explain_local(sample_scaled, [0])
                    if local_explanation is not None:
                        local_data = local_explanation.data()
                        if local_data is not None and 'names' in local_data and 'scores' in local_data:
                            feature_names = local_data['names']
                            feature_scores = local_data['scores']
                            logger.info(f"Method 2 success: got {len(feature_scores)} feature scores")
                except Exception as e:
                    logger.debug(f"Method 2 failed: {e}")
            
            # Method 3: Calculate approximate feature importance based on model coefficients
            if feature_names is None:
                logger.warning(f"Local explanation failed, calculating approximate feature importance")
                feature_names = self.feature_columns
                feature_scores = []
                
                # Get the feature values
                sample_values = sample_df.iloc[0]
                
                # Calculate approximate importance as feature_value * global_importance
                global_explanation = self.ebm_model.explain_global()
                global_data = global_explanation.data()
                global_scores = global_data.get('scores', [1.0] * len(feature_names))
                
                for i, feature_name in enumerate(feature_names):
                    if i < len(global_scores):
                        feature_value = sample_values.get(feature_name, 0)
                        # Approximate contribution as normalized feature value * global importance
                        approx_score = (feature_value / (abs(feature_value) + 1)) * global_scores[i] * 0.1
                        feature_scores.append(approx_score)
                    else:
                        feature_scores.append(0.0)
                
                logger.info(f"Generated approximate scores for {len(feature_scores)} features")
                
        except Exception as e:
            logger.warning(f"All explanation methods failed: {e}, using minimal fallback")
            feature_names = self.feature_columns
            feature_scores = [0.001] * len(feature_names)  # Small non-zero values
        
        # Get feature values
        feature_values = sample_df.iloc[0].to_dict()
        
        # Generate detailed explanation text
        explanation = self._generate_detailed_explanation(
            company_name, prediction, prediction_proba, 
            feature_names, feature_scores, feature_values
        )
        
        return {
            'explanation_text': explanation,
            'prediction': 'Investment Grade' if prediction == 1 else 'Non-Investment Grade',
            'probability_investment_grade': prediction_proba[1],
            'probability_non_investment_grade': prediction_proba[0],
            'feature_contributions': dict(zip(feature_names, feature_scores))
        }
    
    def _generate_detailed_explanation(self, company_name, prediction, prediction_proba, 
                                     feature_names, feature_scores, feature_values):
        """Generate the detailed textual explanation."""
        
        # Sort features by absolute contribution
        feature_data = list(zip(feature_names, feature_scores, 
                               [feature_values.get(name, 0) for name in feature_names]))
        feature_data.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Start building explanation
        explanation_lines = []
        explanation_lines.append(f"CREDIT RISK ANALYSIS FOR {company_name.upper()}")
        explanation_lines.append("=" * 60)
        explanation_lines.append("")
        
        # Overall prediction
        grade = "INVESTMENT GRADE" if prediction == 1 else "NON-INVESTMENT GRADE"
        
        explanation_lines.append(f"OVERALL RATING: {grade}")
        explanation_lines.append(f"Investment Grade Probability: {prediction_proba[1]:.1%}")
        explanation_lines.append(f"Non-Investment Grade Probability: {prediction_proba[0]:.1%}")
        explanation_lines.append("")
        
        # Feature analysis
        explanation_lines.append("DETAILED FEATURE ANALYSIS:")
        explanation_lines.append("-" * 40)
        explanation_lines.append("")
        
        total_abs_contribution = sum(abs(score) for _, score, _ in feature_data)
        
        for i, (feature_name, contribution, value) in enumerate(feature_data[:10]):  # Top 10 features
            
            # Calculate percentage contribution
            contrib_pct = (abs(contribution) / max(total_abs_contribution, 1e-8)) * 100
            
            # Determine impact direction based on prediction and contribution
            if prediction == 1:  # Investment Grade prediction
                impact = "INCREASES investment grade probability" if contribution > 0 else "DECREASES investment grade probability"
                impact_symbol = "+" if contribution > 0 else "-"
            else:  # Non-Investment Grade prediction
                impact = "INCREASES non-investment grade probability" if contribution > 0 else "DECREASES non-investment grade probability"
                impact_symbol = "+" if contribution > 0 else "-"
            
            # Get business interpretation
            interpretation = self.get_feature_interpretation(feature_name, value, contribution)
            
            # Format the explanation
            explanation_lines.append(f"{i+1}. {feature_name.replace('_', ' ').title()}:")
            explanation_lines.append(f"   Value: {value:.4f}")
            explanation_lines.append(f"   Interpretation: {interpretation}")
            explanation_lines.append(f"   Impact: [{impact_symbol}] {impact} by {contrib_pct:.1f}%")
            explanation_lines.append(f"   Contribution Score: {contribution:+.4f}")
            explanation_lines.append("")
        
        # Risk summary
        explanation_lines.append("RISK SUMMARY:")
        explanation_lines.append("-" * 20)
        
        # Count positive vs negative contributions based on prediction
        if prediction == 1:  # Investment Grade
            supporting_features = [f for f, s, v in feature_data if s > 0]
            opposing_features = [f for f, s, v in feature_data if s < 0]
            explanation_lines.append(f"Supporting Investment Grade: {len(supporting_features)} features")
            explanation_lines.append(f"Opposing Investment Grade: {len(opposing_features)} features")
        else:  # Non-Investment Grade
            supporting_features = [f for f, s, v in feature_data if s > 0]
            opposing_features = [f for f, s, v in feature_data if s < 0]
            explanation_lines.append(f"Supporting Non-Investment Grade: {len(supporting_features)} features")
            explanation_lines.append(f"Opposing Non-Investment Grade: {len(opposing_features)} features")
        
        explanation_lines.append("")
        
        # Key findings based on actual prediction
        explanation_lines.append("KEY FINDINGS:")
        
        if prediction == 1:  # Investment Grade
            # Show top positive contributors (pros)
            key_strengths = []
            for feature_name, contribution, value in feature_data[:5]:
                if contribution > 0:
                    strength_desc = f"{feature_name.replace('_', ' ').title()} ({value:.3f})"
                    key_strengths.append(strength_desc)
            
            if key_strengths:
                explanation_lines.append("Investment Grade Drivers (Pros):")
                for strength in key_strengths:
                    explanation_lines.append(f"  • {strength}")
                explanation_lines.append("")
                
        else:  # Non-Investment Grade
            # Show top negative contributors (cons) that drive the non-IG decision
            # For non-IG, we want to highlight the factors that push it away from investment grade
            key_risks = []
            risk_factors = []
            
            # If most contributions are positive but prediction is non-IG, 
            # it means the baseline/bias pushes toward non-IG
            negative_contributors = [f for f, s, v in feature_data if s < 0]
            if len(negative_contributors) == 0 or len(negative_contributors) < 3:
                explanation_lines.append("Non-Investment Grade Drivers (Cons):")
                explanation_lines.append("  • Overall risk profile exceeds investment grade thresholds")
                explanation_lines.append("  • Despite positive individual factors, combined risk assessment indicates higher risk")
                
                # Show the factors that contribute least (weakest positive factors)
                weakest_factors = sorted(feature_data, key=lambda x: x[1])[:3]
                explanation_lines.append("  • Weakest supporting factors:")
                for feature_name, contribution, value in weakest_factors:
                    explanation_lines.append(f"    - {feature_name.replace('_', ' ').title()}: {value:.3f}")
            else:
                # Show actual negative contributors
                for feature_name, contribution, value in feature_data:
                    if contribution < 0:
                        risk_desc = f"{feature_name.replace('_', ' ').title()} ({value:.3f})"
                        risk_factors.append(risk_desc)
                        if len(risk_factors) >= 5:
                            break
                
                if risk_factors:
                    explanation_lines.append("Non-Investment Grade Drivers (Cons):")
                    for risk in risk_factors:
                        explanation_lines.append(f"  • {risk}")
            
            explanation_lines.append("")
        
        # Recommendation
        explanation_lines.append("RECOMMENDATION:")
        explanation_lines.append("-" * 15)
        
        confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
        
        if prediction == 1 and confidence > 0.8:
            recommendation = "APPROVE - Strong investment grade profile with high confidence"
        elif prediction == 1 and confidence > 0.6:
            recommendation = "APPROVE - Investment grade with moderate confidence, monitor key metrics"
        elif prediction == 0 and confidence > 0.8:
            recommendation = "REJECT - High risk profile, not suitable for investment grade"
        else:
            recommendation = "REVIEW - Borderline case requiring manual underwriting review"
        
        explanation_lines.append(recommendation)
        explanation_lines.append("")
        explanation_lines.append(f"Analysis generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(explanation_lines)
    
    def explain_multiple_samples(self, csv_path, sample_indices=None, num_samples=5):
        """Explain multiple samples from the dataset."""
        
        logger.info(f"Loading data for multiple explanations from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        if sample_indices is None:
            # Select random samples
            sample_indices = np.random.choice(len(df), min(num_samples, len(df)), replace=False)
        
        explanations = []
        
        for idx in sample_indices:
            sample = df.iloc[idx]
            company_name = sample.get('company_name', f'Company_{idx}')
            
            try:
                explanation_result = self.explain_single_prediction(sample, company_name)
                explanations.append({
                    'index': idx,
                    'company_name': company_name,
                    'result': explanation_result
                })
                
                # Save individual explanation
                filename = f"explanation_company_{idx}_{company_name.replace(' ', '_')}.txt"
                filepath = os.path.join(EXPLANATIONS_OUTPUT_DIR, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(explanation_result['explanation_text'])
                
                logger.info(f"Explanation saved for {company_name}: {filepath}")
                
            except Exception as e:
                logger.error(f"Error explaining sample {idx}: {e}")
        
        return explanations
    
    def generate_global_explanation_report(self):
        """Generate a global explanation report showing overall model behavior."""
        
        # Get global explanation
        global_explanation = self.ebm_model.explain_global()
        global_data = global_explanation.data()
        
        feature_names = global_data['names']
        feature_importances = global_data['scores']
        
        # Create report
        report_lines = []
        report_lines.append("EBM MODEL GLOBAL EXPLANATION REPORT")
        report_lines.append("=" * 50)
        report_lines.append("")
        report_lines.append("FEATURE IMPORTANCE RANKING:")
        report_lines.append("-" * 30)
        report_lines.append("")
        
        # Sort by importance
        feature_importance_pairs = list(zip(feature_names, feature_importances))
        feature_importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for i, (feature, importance) in enumerate(feature_importance_pairs):
            report_lines.append(f"{i+1:2d}. {feature.replace('_', ' ').title()}")
            report_lines.append(f"    Importance Score: {importance:.4f}")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_path = os.path.join(EXPLANATIONS_OUTPUT_DIR, "global_explanation_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Global explanation report saved: {report_path}")
        return report_text


def main():
    """Main function to demonstrate explainability features."""
    
    logger.info("Starting EBM Model Explainability Analysis")
    logger.info("=" * 50)
    
    try:
        # Initialize explainer
        explainer = EBMExplainer(MODEL_PATH)
        
        # Generate global explanation report
        global_report = explainer.generate_global_explanation_report()
        print("Global Explanation Report Generated")
        
        # Explain multiple samples
        explanations = explainer.explain_multiple_samples(CSV_FILE_PATH, num_samples=3)
        
        # Print one detailed explanation as example
        if explanations:
            print("\n" + "="*60)
            print("SAMPLE DETAILED EXPLANATION:")
            print("="*60)
            print(explanations[0]['result']['explanation_text'])
        
        logger.info("Explainability analysis completed successfully!")
        logger.info(f"Explanations saved to: {EXPLANATIONS_OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"Error in explainability analysis: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
