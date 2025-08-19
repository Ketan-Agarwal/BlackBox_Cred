# train_ebm_model_updated.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from interpret.glassbox import ExplainableBoostingClassifier
import pickle
import os
from datetime import datetime
import logging

# --- Configuration ---
# *** UPDATE THIS PATH TO YOUR ACTUAL CSV FILE LOCATION ***
CSV_FILE_PATH = r"C:\Users\asus\Documents\GitHub\BlackBox_Cred\credtech_backend\complete_training_dataset_corrected.csv"

# Output directory for the trained model
MODEL_OUTPUT_DIR = r"C:\Users\asus\Documents\GitHub\BlackBox_Cred\credtech_backend\models"
MODEL_FILENAME = "ebm_model_trained_on_csv.pkl" # Changed filename to avoid conflicts

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure output directory exists
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
MODEL_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, MODEL_FILENAME)

# --- Feature Definitions ---
# These column names are derived from analyzing your complete_training_dataset.csv
# The order doesn't strictly matter for EBM, but listing them explicitly is good practice.
FEATURE_COLUMNS = [
    
    # === CORE CREDIT RISK SCORES (Must-Have) ===
    "enhanced_z_score",                    # Best single bankruptcy predictor
    "kmv_distance_to_default",             # Market-based default probability
    
    # === LEVERAGE & CAPITAL STRUCTURE ===
    "debt_ratio",                          # Total leverage ratio
    "debt_to_equity",                      # Capital structure risk
    
    # === LIQUIDITY & WORKING CAPITAL ===
    "current_ratio",                       # Short-term liquidity buffer
    "cash_and_equivalents",                # Immediate liquidity
    "accounts_receivable",                 # Collection risk indicator
    "accounts_payable",                    # Payment timing flexibility
    "inventory",                           # Working capital component
    
    # === PROFITABILITY & EFFICIENCY ===
    "gross_margin",                        # Pricing power indicator
    "operating_margin",                    # Core business profitability
    "net_margin",                          # Bottom-line efficiency
    "return_on_equity",                    # Management effectiveness
    "return_on_assets",                    # Asset utilization efficiency
    "asset_turnover",                      # Asset efficiency ratio
    
    # === RAW FINANCIAL DATA (Income Statement) ===
    "total_revenue",                       # Business scale indicator
    "gross_profit",                        # Revenue quality
    "operating_income",                    # Operational strength
    "net_income",                          # Final profitability
    
    # === RAW FINANCIAL DATA (Balance Sheet) ===
    "total_assets",                        # Company size
    "current_assets",                      # Liquid resources
    "total_equity",                        # Capital base strength
    "current_liabilities",                 # Short-term obligations
    "total_liabilities",                   # Total debt burden
    
    # === MARKET RISK & SENTIMENT ===
    "market_cap",                          # Size/systemic importance
    "volatility",                          # Market-based risk signal
    "vix",                                 # Market-wide fear gauge
    
    # === MACROECONOMIC ENVIRONMENT ===
    "fed_funds_rate",                      # Base borrowing environment
    "treasury_10y",                        # Long-term risk-free rate
    "treasury_3m",                         # Short-term risk-free rate
    "credit_spread_high_yield",            # Credit market stress
    "credit_spread_investment",            # Investment grade spreads
    "unemployment_rate",                   # Economic health indicator
    "yield_curve_slope"                    # Economic expectations
]



# The target column name in your CSV
TARGET_COLUMN = 'is_investment_grade' # 1 for Investment Grade, 0 for Non-Investment Grade

def load_and_preprocess_data(csv_path):
    """Loads the CSV and prepares it for training."""
    logger.info(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Data loaded. Shape: {df.shape}")

        # --- Data Inspection (Optional, helpful for debugging) ---
        # logger.debug(f"First few rows:\n{df.head()}")
        # logger.debug(f"Column names:\n{list(df.columns)}")
        # logger.debug(f"Data types:\n{df.dtypes}")

        # Verify target column exists
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in CSV.")

        
        # --- Data Preprocessing ---
        # 1. Select feature columns. Handle potential missing macro columns gracefully.
        available_feature_columns = [col for col in FEATURE_COLUMNS if col in df.columns]
        missing_feature_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_feature_columns:
            logger.warning(f"Some feature columns not found in CSV (will be ignored): {missing_feature_columns}")
        
        feature_df = df[available_feature_columns]

        # 2. Handle missing values: Fill NaNs with 0. 
        #    (Financial data sometimes uses 0 for missing, or imputation could be used)
        feature_df = feature_df.fillna(0)
        
        # 3. Handle infinite values (e.g., from division by zero)
        feature_df = feature_df.replace([np.inf, -np.inf], 0)
        
        # 4. Extract target variable
        y = df[TARGET_COLUMN].astype(int) # Ensure target is integer (0 or 1)

        # 5. Final check for consistency
        if len(feature_df) != len(y):
            raise ValueError("Feature matrix and target vector have different lengths after processing.")

        logger.info("Data preprocessing complete.")
        logger.info(f"Final feature matrix shape: {feature_df.shape}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        
        return feature_df, y

    except Exception as e:
        logger.error(f"Error loading or preprocessing data: {e}")
        raise

def train_and_evaluate_model(X, y):
    """Trains the EBM model and evaluates it."""
    logger.info("Splitting data into train and test sets...")
    # Use stratification to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
    )
    logger.info(f"Train set size: {X_train.shape[0]}")
    logger.info(f"Test set size: {X_test.shape[0]}")

    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Feature scaling complete.")

    logger.info("Training Explainable Boosting Machine (EBM) model...")
    # Configure EBM
    ebm_model = ExplainableBoostingClassifier(
        random_state=42,
        learning_rate=0.01,
        max_bins=256,
        max_interaction_bins=32,
        interactions=30, # Allows EBM to find interactions between features
        n_jobs=1 # Use single core to avoid Windows multiprocessing issues
    )
    
    # Fit the model
    ebm_model.fit(X_train_scaled, y_train)
    logger.info("EBM model training complete.")

    logger.info("Evaluating model on test set...")
    # Make predictions
    y_pred = ebm_model.predict(X_test_scaled)
    # y_pred_proba = ebm_model.predict_proba(X_test_scaled)[:, 1] # Probability of class 1 (Investment Grade)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test Accuracy: {accuracy:.4f}")

    logger.info("Classification Report:")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Non-Investment Grade', 'Investment Grade'])}")

    logger.info("Confusion Matrix:")
    logger.info(f"\n{confusion_matrix(y_test, y_pred)}")

    # Package results
    results = {
        'model': ebm_model,
        'scaler': scaler,
        'feature_columns': X.columns.tolist(), # Store the actual columns used
        'accuracy': accuracy,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test,
        'y_pred': y_pred,
        # 'y_pred_proba': y_pred_proba # Not strictly needed for saving, but useful for analysis
    }

    return results

def save_model(results, output_path):
    """Saves the trained model, scaler, and metadata."""
    logger.info(f"Saving model to {output_path}...")
    try:
        model_data = {
            'model': results['model'],
            'scaler': results['scaler'],
            'feature_columns': results['feature_columns'], # Use the actual columns from preprocessing
            'training_date': datetime.now().isoformat(),
            'accuracy': results['accuracy']
        }
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info("Model saved successfully.")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def main():
    """Main execution function."""
    logger.info("Starting EBM Model Training Pipeline (Updated for CSV)")
    logger.info("=" * 60)

    try:
        # 1. Load and preprocess data
        X, y = load_and_preprocess_data(CSV_FILE_PATH)

        # 2. Train and evaluate the model
        results = train_and_evaluate_model(X, y)

        # 3. Save the trained model
        save_model(results, MODEL_OUTPUT_PATH)

        logger.info("=" * 60)
        logger.info("EBM Model Training Pipeline Completed Successfully!")
        logger.info(f"Model saved to: {MODEL_OUTPUT_PATH}")
        logger.info(f"Final Test Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Features used: {len(results['feature_columns'])}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Fatal error in training pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()