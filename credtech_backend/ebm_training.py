# ebm_training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
MODEL_FILENAME = "ebm_model_trained_on_csv.pkl"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure output directory exists
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
MODEL_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, MODEL_FILENAME)

# --- Feature Definitions ---
FEATURE_COLUMNS = [
    # === CORE CREDIT RISK SCORES (Must-Have) ===
    "enhanced_z_score",
    "kmv_distance_to_default",

    # === LEVERAGE & CAPITAL STRUCTURE ===
    "debt_ratio",
    "debt_to_equity",

    # === LIQUIDITY & WORKING CAPITAL ===
    "current_ratio",
    "cash_and_equivalents",
    "accounts_receivable",
    "accounts_payable",
    "inventory",

    # === PROFITABILITY & EFFICIENCY ===
    "gross_margin",
    "operating_margin",
    "net_margin",
    "return_on_equity",
    "return_on_assets",
    "asset_turnover",

    # === RAW FINANCIAL DATA (Income Statement) ===
    "total_revenue",
    "gross_profit",
    "operating_income",
    "net_income",

    # === RAW FINANCIAL DATA (Balance Sheet) ===
    "total_assets",
    "current_assets",
    "total_equity",
    "current_liabilities",
    "total_liabilities",

    # === MARKET RISK & SENTIMENT ===
    "market_cap",
    "volatility",
    "vix",

    # === MACROECONOMIC ENVIRONMENT ===
    "fed_funds_rate",
    "treasury_10y",
    "treasury_3m",
    "credit_spread_high_yield",
    "credit_spread_investment",
    "unemployment_rate",
    "yield_curve_slope"
]

# The target column name in your CSV
TARGET_COLUMN = 'is_investment_grade'

# --- Hyperparameter Tuning Configuration ---
# Simplified grid for initial stability and speed
# Explicitly setting classifier__n_jobs=1 to prevent internal EBM parallelism issues
HYPERPARAMETER_GRID = {
    'classifier__learning_rate': [0.01, 0.015], # Default 0.015 for classification
    'classifier__max_leaves': [2, 3], # Default 2. Try 3 based on guidance.
    'classifier__smoothing_rounds': [75, 150], # Default 75 for classification
    'classifier__interactions': ["3x", "4x"], # Default "3x". Try slightly higher.
    'classifier__inner_bags': [0, 10], # Default 0. Try 10 (less than 20).
    # Start with fewer params to ensure stability. Add interactions/inner_bags later if this works.
}

N_CV_FOLDS = 3
SCORING_METRIC = 'accuracy' # Or 'f1', 'roc_auc'

def load_and_preprocess_data(csv_path):
    """Loads the CSV and prepares it for training."""
    logger.info(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Data loaded. Shape: {df.shape}")

        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in CSV.")

        # --- Data Preprocessing ---
        available_feature_columns = [col for col in FEATURE_COLUMNS if col in df.columns]
        missing_feature_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_feature_columns:
            logger.warning(f"Some feature columns not found in CSV (will be ignored): {missing_feature_columns}")

        feature_df = df[available_feature_columns]
        feature_df = feature_df.fillna(0)
        feature_df = feature_df.replace([np.inf, -np.inf], 0)
        y = df[TARGET_COLUMN].astype(int)

        if len(feature_df) != len(y):
            raise ValueError("Feature matrix and target vector have different lengths after processing.")

        logger.info("Data preprocessing complete.")
        logger.info(f"Final feature matrix shape: {feature_df.shape}")
        logger.info(f"Target distribution:\n{y.value_counts()}")

        return feature_df, y

    except Exception as e:
        logger.error(f"Error loading or preprocessing data: {e}")
        raise

def get_ebm_pipeline():
    """Creates a scikit-learn Pipeline with a StandardScaler and an ExplainableBoostingClassifier."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', ExplainableBoostingClassifier(
            random_state=42,
            # CRITICAL: Explicitly disable internal EBM parallelism to avoid '_posixsubprocess' error on Windows
            n_jobs=1, # Add this line
            # Potentially adjust other parameters that control fitting time if needed
            # early_stopping_rounds=50, # Reduce if needed for faster tuning
            # max_rounds=1000, # Reduce if needed for faster tuning (ensure it's enough)
        ))
    ])
    return pipeline


def perform_hyperparameter_tuning(X_train, y_train):
    """Performs hyperparameter tuning using GridSearchCV."""
    logger.info("Starting hyperparameter tuning with GridSearchCV...")

    pipeline = get_ebm_pipeline()

    # Define GridSearchCV with error_score='raise' for better debugging
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=HYPERPARAMETER_GRID,
        cv=N_CV_FOLDS,
        scoring=SCORING_METRIC,
        # CRITICAL: Use n_jobs=1 for GridSearchCV itself to avoid Windows multiprocessing issues
        n_jobs=1, # Keep this as 1
        verbose=2,
        # CRITICAL: Add error_score='raise' to see the actual error from failed fits
        error_score='raise' # Change this line
    )

    logger.info(f"Fitting GridSearchCV with grid: {HYPERPARAMETER_GRID}")
    try:
        grid_search.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"An error occurred during GridSearchCV fitting: {e}")
        logger.error("This might be due to an issue with one of the hyperparameter combinations.")
        logger.error("Consider simplifying the HYPERPARAMETER_GRID or checking data types.")
        raise # Re-raise the exception to stop execution

    logger.info("Hyperparameter tuning complete.")
    logger.info(f"Best cross-validation {SCORING_METRIC}: {grid_search.best_score_:.4f}")
    logger.info(f"Best parameters: {grid_search.best_params_}")

    return grid_search.best_estimator_

def train_and_evaluate_model(X, y):
    """Splits data, tunes hyperparameters, trains the best EBM model, and evaluates it."""
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
    )
    logger.info(f"Train set size: {X_train.shape[0]}")
    logger.info(f"Test set size: {X_test.shape[0]}")

    # --- Hyperparameter Tuning ---
    best_pipeline = perform_hyperparameter_tuning(X_train, y_train)

    best_scaler = best_pipeline.named_steps['scaler']
    best_ebm_model = best_pipeline.named_steps['classifier']

    logger.info("Evaluating the best model on the test set...")
    y_pred = best_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test Accuracy (Best Model): {accuracy:.4f}")

    logger.info("Classification Report (Best Model):")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Non-Investment Grade', 'Investment Grade'])}")

    logger.info("Confusion Matrix (Best Model):")
    logger.info(f"\n{confusion_matrix(y_test, y_pred)}")

    results = {
        'model': best_ebm_model,
        'scaler': best_scaler,
        'pipeline': best_pipeline,
        'feature_columns': X.columns.tolist(),
        'accuracy': accuracy,
        'X_test': X_test,
        'X_test_scaled': best_scaler.transform(X_test),
        'y_test': y_test,
        'y_pred': y_pred,
    }

    return results

def save_model(results, output_path):
    """Saves the trained model, scaler, and metadata."""
    logger.info(f"Saving model to {output_path}...")
    try:
        model_data = {
            'model': results['model'],
            'scaler': results['scaler'],
            'feature_columns': results['feature_columns'],
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
    logger.info("Starting EBM Model Training Pipeline with Hyperparameter Tuning (Updated for CSV)")
    logger.info("=" * 70)

    try:
        X, y = load_and_preprocess_data(CSV_FILE_PATH)
        results = train_and_evaluate_model(X, y)
        save_model(results, MODEL_OUTPUT_PATH)

        logger.info("=" * 70)
        logger.info("EBM Model Training Pipeline with Hyperparameter Tuning Completed Successfully!")
        logger.info(f"Model saved to: {MODEL_OUTPUT_PATH}")
        logger.info(f"Final Test Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Features used: {len(results['feature_columns'])}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Fatal error in training pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()