"""
Structured model service implementing KMV, Z-Score, and Random Forest models.
"""
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from sqlalchemy.orm import Session
from scipy.optimize import fsolve
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import shap

from app.core.config import settings
from app.db.session import get_db_session
from app.db.models import Company, StructuredData, CreditScoreHistory


class StructuredModelService:
    """Service for structured financial data analysis and scoring."""
    
    def __init__(self):
        self.model: Optional[RandomForestClassifier] = None
        self.feature_columns: List[str] = []
        self.shap_explainer: Optional[shap.TreeExplainer] = None
        
    def _calculate_kmv(self, company_data: Dict) -> float:
        """
        Calculate KMV Distance-to-Default using Merton model.
        
        Args:
            company_data: Dictionary containing financial data
            
        Returns:
            Distance-to-Default value
        """
        try:
            # Extract required data
            equity_value = company_data.get('market_cap', 0)
            debt_value = company_data.get('total_liabilities', 0)
            volatility = company_data.get('volatility', 0.3)  # Default to 30% if not available
            risk_free_rate = 0.03  # Assume 3% risk-free rate
            time_horizon = 1.0  # 1 year
            
            if equity_value <= 0 or debt_value <= 0:
                return 0.0
                
            # Asset value is equity + debt
            asset_value = equity_value + debt_value
            
            # Default point (debt face value)
            default_point = debt_value
            
            # Calculate d1 and d2 from Black-Scholes-Merton
            d1 = (np.log(asset_value / default_point) + 
                  (risk_free_rate + 0.5 * volatility**2) * time_horizon) / (volatility * np.sqrt(time_horizon))
            
            d2 = d1 - volatility * np.sqrt(time_horizon)
            
            # Distance-to-Default is essentially d2
            distance_to_default = d2
            
            return float(distance_to_default)
            
        except Exception as e:
            print(f"Error calculating KMV: {e}")
            return 0.0
    
    def _calculate_z_score(self, company_data: Dict) -> float:
        """
        Calculate Altman Z-Score for bankruptcy prediction.
        
        Args:
            company_data: Dictionary containing financial ratios
            
        Returns:
            Altman Z-Score value
        """
        try:
            # Altman Z-Score formula for public companies:
            # Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
            
            # X1 = Working Capital / Total Assets
            current_assets = company_data.get('total_assets', 0) * company_data.get('current_ratio', 1.0)
            current_liabilities = current_assets / max(company_data.get('current_ratio', 1.0), 0.1)
            working_capital = current_assets - current_liabilities
            x1 = working_capital / max(company_data.get('total_assets', 1), 1)
            
            # X2 = Retained Earnings / Total Assets (approximated using equity ratio)
            x2 = company_data.get('total_equity', 0) / max(company_data.get('total_assets', 1), 1)
            
            # X3 = EBIT / Total Assets (using operating income)
            x3 = company_data.get('operating_income', 0) / max(company_data.get('total_assets', 1), 1)
            
            # X4 = Market Value of Equity / Book Value of Total Debt
            x4 = company_data.get('market_cap', 0) / max(company_data.get('total_liabilities', 1), 1)
            
            # X5 = Sales / Total Assets (asset turnover)
            x5 = company_data.get('asset_turnover', 0)
            
            # Calculate Z-Score
            z_score = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
            
            return float(z_score)
            
        except Exception as e:
            print(f"Error calculating Z-Score: {e}")
            return 0.0
    
    def _prepare_features(self, company_data: Dict) -> np.ndarray:
        """
        Prepare feature vector for Random Forest model.
        
        Args:
            company_data: Dictionary containing all financial data
            
        Returns:
            Feature vector as numpy array
        """
        # Calculate KMV and Z-Score
        kmv_dd = self._calculate_kmv(company_data)
        z_score = self._calculate_z_score(company_data)
        
        # Define feature order (must match training order)
        features = [
            company_data.get('current_ratio', 0),
            company_data.get('quick_ratio', 0),
            company_data.get('debt_to_equity', 0),
            company_data.get('return_on_equity', 0),
            company_data.get('return_on_assets', 0),
            company_data.get('operating_margin', 0),
            company_data.get('net_margin', 0),
            company_data.get('asset_turnover', 0),
            company_data.get('inventory_turnover', 0),
            kmv_dd,  # KMV Distance-to-Default
            z_score,  # Altman Z-Score
        ]
        
        return np.array(features).reshape(1, -1)
    
    def train(self) -> Dict:
        """
        Train the Random Forest model using historical data.
        
        Returns:
            Training results and metrics
        """
        db = get_db_session()
        
        try:
            # Load historical structured data
            query = db.query(StructuredData).join(Company).all()
            
            if len(query) < 10:
                raise ValueError("Insufficient training data. Need at least 10 records.")
            
            # Prepare training data
            X = []
            y = []
            
            for record in query:
                # Convert record to dictionary
                company_data = {
                    'current_ratio': record.current_ratio or 0,
                    'quick_ratio': record.quick_ratio or 0,
                    'debt_to_equity': record.debt_to_equity or 0,
                    'return_on_equity': record.return_on_equity or 0,
                    'return_on_assets': record.return_on_assets or 0,
                    'operating_margin': record.operating_margin or 0,
                    'net_margin': record.net_margin or 0,
                    'asset_turnover': record.asset_turnover or 0,
                    'inventory_turnover': record.inventory_turnover or 0,
                    'total_assets': record.total_assets or 0,
                    'total_liabilities': record.total_liabilities or 0,
                    'total_equity': record.total_equity or 0,
                    'market_cap': record.market_cap or 0,
                    'operating_income': record.operating_income or 0,
                    'volatility': record.volatility or 0.3
                }
                
                # Prepare features
                features = self._prepare_features(company_data).flatten()
                X.append(features)
                
                # Create binary label based on financial health (simplified)
                # You would typically have actual default labels
                z_score = self._calculate_z_score(company_data)
                label = 1 if z_score > 2.6 else 0  # Simplified labeling
                y.append(label)
            
            X = np.array(X)
            y = np.array(y)
            
            # Define feature names
            self.feature_columns = [
                'current_ratio', 'quick_ratio', 'debt_to_equity', 'return_on_equity',
                'return_on_assets', 'operating_margin', 'net_margin', 'asset_turnover',
                'inventory_turnover', 'kmv_distance_to_default', 'altman_z_score'
            ]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train Random Forest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Initialize SHAP explainer
            self.shap_explainer = shap.TreeExplainer(self.model)
            
            # Save model
            import os
            os.makedirs(os.path.dirname(settings.random_forest_model_path), exist_ok=True)
            with open(settings.random_forest_model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'feature_columns': self.feature_columns,
                    'shap_explainer': self.shap_explainer
                }, f)
            
            return {
                'accuracy': accuracy,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': len(self.feature_columns),
                'model_path': settings.random_forest_model_path
            }
            
        except Exception as e:
            print(f"Error training model: {e}")
            raise
        finally:
            db.close()
    
    def _load_model(self):
        """Load the trained Random Forest model."""
        if self.model is None:
            try:
                with open(settings.random_forest_model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.feature_columns = model_data['feature_columns']
                    self.shap_explainer = model_data.get('shap_explainer')
            except FileNotFoundError:
                raise ValueError("Model not found. Please train the model first.")
    
    def get_structured_score(self, company_id: int) -> Dict:
        """
        Calculate structured credit score for a company.
        
        Args:
            company_id: Company ID
            
        Returns:
            Dictionary containing scores and explanations
        """
        self._load_model()
        db = get_db_session()
        
        try:
            # Get latest structured data
            latest_data = (
                db.query(StructuredData)
                .filter(StructuredData.company_id == company_id)
                .order_by(StructuredData.data_date.desc())
                .first()
            )
            
            if not latest_data:
                raise ValueError(f"No structured data found for company {company_id}")
            
            # Prepare company data dictionary
            company_data = {
                'current_ratio': latest_data.current_ratio or 0,
                'quick_ratio': latest_data.quick_ratio or 0,
                'debt_to_equity': latest_data.debt_to_equity or 0,
                'return_on_equity': latest_data.return_on_equity or 0,
                'return_on_assets': latest_data.return_on_assets or 0,
                'operating_margin': latest_data.operating_margin or 0,
                'net_margin': latest_data.net_margin or 0,
                'asset_turnover': latest_data.asset_turnover or 0,
                'inventory_turnover': latest_data.inventory_turnover or 0,
                'total_assets': latest_data.total_assets or 0,
                'total_liabilities': latest_data.total_liabilities or 0,
                'total_equity': latest_data.total_equity or 0,
                'market_cap': latest_data.market_cap or 0,
                'operating_income': latest_data.operating_income or 0,
                'volatility': latest_data.volatility or 0.3
            }
            
            # Calculate KMV and Z-Score
            kmv_dd = self._calculate_kmv(company_data)
            z_score = self._calculate_z_score(company_data)
            
            # Update database with calculated values
            latest_data.kmv_distance_to_default = kmv_dd
            latest_data.altman_z_score = z_score
            db.commit()
            
            # Prepare features for model
            features = self._prepare_features(company_data)
            
            # Get model prediction
            prediction_proba = self.model.predict_proba(features)[0]
            structured_score = prediction_proba[1] * 100  # Probability of good credit * 100
            
            # Get SHAP values for explanation
            shap_values = None
            if self.shap_explainer:
                shap_values = self.shap_explainer.shap_values(features)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification, take positive class
            
            return {
                'structured_score': float(structured_score),
                'kmv_distance_to_default': float(kmv_dd),
                'altman_z_score': float(z_score),
                'shap_values': shap_values.flatten().tolist() if shap_values is not None else None,
                'feature_names': self.feature_columns,
                'feature_values': features.flatten().tolist(),
                'company_data': company_data
            }
            
        except Exception as e:
            print(f"Error calculating structured score: {e}")
            raise
        finally:
            db.close()
