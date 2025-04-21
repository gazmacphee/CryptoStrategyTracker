"""
Model Ensemble System for Cryptocurrency Price Prediction

This module provides an ensemble ML approach that combines multiple prediction models:
1. Gradient Boosting Regression (from original implementation)
2. Random Forest Regression
3. Linear Regression (as a baseline)

By combining these models, we aim to create more robust predictions
that work well across different market conditions.
"""

import os
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import time
import json

# ML libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Local imports
from database import get_historical_data, get_indicators
from utils import timeframe_to_seconds
from ml_prediction import MLPredictor  # Import the base predictor class

class EnsemblePredictor(MLPredictor):
    """
    Ensemble-based Machine Learning Predictor for cryptocurrency price movements
    that combines multiple models for more robust predictions
    """
    
    def __init__(self, symbol, interval, prediction_horizon=24, feature_window=30):
        """
        Initialize the ensemble predictor
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Time interval (e.g., '1h', '4h', '1d')
            prediction_horizon: How many periods ahead to predict
            feature_window: How many historical periods to use for features
        """
        # Initialize parent class
        super().__init__(symbol, interval, prediction_horizon, feature_window)
        
        # Override model paths for ensemble
        self.model_path = os.path.join('models', f"{symbol}_{interval}_ensemble_model.joblib")
        self.metadata_path = os.path.join('models', f"{symbol}_{interval}_ensemble_metadata.json")
        
        # Create model-specific paths
        self.gbr_model_path = os.path.join('models', f"{symbol}_{interval}_gbr_model.joblib")
        self.rf_model_path = os.path.join('models', f"{symbol}_{interval}_rf_model.joblib")
        self.lr_model_path = os.path.join('models', f"{symbol}_{interval}_lr_model.joblib")
        
        # Initialize sub-models
        self.gbr_model = None  # Gradient Boosting Regressor
        self.rf_model = None   # Random Forest Regressor
        self.lr_model = None   # Linear Regression (baseline)
        
        # Ensemble weights
        self.weights = [0.5, 0.3, 0.2]  # Default weights: 50% GBR, 30% RF, 20% LR
        
        # Track sub-model metrics
        self.submodel_metrics = {}
    
    def _create_model(self):
        """
        Create an ensemble of machine learning models
        
        Returns:
            Initialized ML model ensemble
        """
        # Create sub-models
        self.gbr_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        
        self.lr_model = LinearRegression()
        
        # Create ensemble using VotingRegressor
        ensemble = VotingRegressor([
            ('gbr', self.gbr_model),
            ('rf', self.rf_model),
            ('lr', self.lr_model)
        ], weights=self.weights)
        
        return ensemble
    
    def train_model(self, lookback_days=90, retrain=False):
        """
        Train the ensemble prediction model
        
        Args:
            lookback_days: How many days of historical data to use
            retrain: Whether to retrain an existing model or create a new one
            
        Returns:
            Boolean indicating success
        """
        # Check if models already exist and we don't want to retrain
        if (os.path.exists(self.model_path) and 
            os.path.exists(self.gbr_model_path) and 
            os.path.exists(self.rf_model_path) and 
            os.path.exists(self.lr_model_path) and 
            not retrain):
            logging.info(f"Ensemble model for {self.symbol}/{self.interval} already exists. Loading model.")
            self.load_model()
            return True
            
        # Get historical data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        # Get price data
        df = get_historical_data(self.symbol, self.interval, start_time, end_time)
        
        # Get indicators data
        indicators_df = get_indicators(self.symbol, self.interval, start_time, end_time)
        
        # Merge price data with indicators
        if not indicators_df.empty:
            df = pd.merge(df, indicators_df, on='timestamp', how='left')
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        if X is None or y is None:
            logging.error(f"Failed to prepare data for {self.symbol}/{self.interval}")
            return False
            
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Feature scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create models if not already created
        if self.model is None:
            self.model = self._create_model()
        
        # Train ensemble model
        try:
            # Train the ensemble model
            self.model.fit(X_train_scaled, y_train)
            
            # Train individual models to get their separate metrics
            logging.info(f"Training individual models for {self.symbol}/{self.interval}")
            
            # Extract individual models from the ensemble
            gbr = self.model.named_estimators_['gbr']
            rf = self.model.named_estimators_['rf']
            lr = self.model.named_estimators_['lr']
            
            # Calculate metrics for ensemble
            y_pred_ensemble = self.model.predict(X_test_scaled)
            self.metrics = self._calculate_metrics(y_test, y_pred_ensemble)
            self.metrics['model_type'] = 'ensemble'
            
            # Calculate metrics for each sub-model
            y_pred_gbr = gbr.predict(X_test_scaled)
            y_pred_rf = rf.predict(X_test_scaled)
            y_pred_lr = lr.predict(X_test_scaled)
            
            self.submodel_metrics['gbr'] = self._calculate_metrics(y_test, y_pred_gbr)
            self.submodel_metrics['gbr']['model_type'] = 'gradient_boosting'
            
            self.submodel_metrics['rf'] = self._calculate_metrics(y_test, y_pred_rf)
            self.submodel_metrics['rf']['model_type'] = 'random_forest'
            
            self.submodel_metrics['lr'] = self._calculate_metrics(y_test, y_pred_lr)
            self.submodel_metrics['lr']['model_type'] = 'linear_regression'
            
            # Save feature importance for supported models
            if hasattr(gbr, 'feature_importances_'):
                self.submodel_metrics['gbr']['feature_importance'] = dict(zip(
                    self.feature_names, gbr.feature_importances_
                ))
            
            if hasattr(rf, 'feature_importances_'):
                self.submodel_metrics['rf']['feature_importance'] = dict(zip(
                    self.feature_names, rf.feature_importances_
                ))
            
            if hasattr(lr, 'coef_'):
                # For linear regression, use normalized coefficients
                coefs = lr.coef_
                abs_coefs = np.abs(coefs)
                normalized_coefs = abs_coefs / np.sum(abs_coefs)
                self.submodel_metrics['lr']['feature_importance'] = dict(zip(
                    self.feature_names, normalized_coefs
                ))
            
            # Calculate optimal ensemble weights based on performance
            self._optimize_ensemble_weights()
            
            # Save models
            self.save_model()
            
            # Save individual models
            joblib.dump(gbr, self.gbr_model_path)
            joblib.dump(rf, self.rf_model_path)
            joblib.dump(lr, self.lr_model_path)
            
            logging.info(f"Successfully trained ensemble model for {self.symbol}/{self.interval}")
            return True
            
        except Exception as e:
            logging.error(f"Error training ensemble model for {self.symbol}/{self.interval}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    def _optimize_ensemble_weights(self):
        """
        Optimize ensemble weights based on sub-model performance
        """
        # If we have direction accuracy for all models, use that as the weight basis
        if all(('direction_accuracy' in self.submodel_metrics[m] for m in ['gbr', 'rf', 'lr'])):
            # Get direction accuracy for each model
            gbr_acc = self.submodel_metrics['gbr']['direction_accuracy']
            rf_acc = self.submodel_metrics['rf']['direction_accuracy']
            lr_acc = self.submodel_metrics['lr']['direction_accuracy']
            
            # Calculate weights proportional to accuracy
            total_acc = gbr_acc + rf_acc + lr_acc
            gbr_weight = gbr_acc / total_acc
            rf_weight = rf_acc / total_acc
            lr_weight = lr_acc / total_acc
            
            # Set weights with a minimum threshold of 0.1
            self.weights = [
                max(0.1, gbr_weight),
                max(0.1, rf_weight),
                max(0.1, lr_weight)
            ]
            
            # Normalize weights to sum to 1
            weight_sum = sum(self.weights)
            self.weights = [w / weight_sum for w in self.weights]
            
            logging.info(f"Optimized ensemble weights for {self.symbol}/{self.interval}: {self.weights}")
        else:
            # Fallback to default weights
            self.weights = [0.5, 0.3, 0.2]
            logging.info(f"Using default ensemble weights for {self.symbol}/{self.interval}: {self.weights}")
    
    def save_model(self):
        """
        Save the trained ensemble model to disk
        
        Returns:
            Boolean indicating success
        """
        try:
            # Save ensemble model
            joblib.dump(self.model, self.model_path)
            
            # Save metadata (feature names, scaler, metrics, weights)
            metadata = {
                'feature_names': self.feature_names,
                'metrics': self.metrics,
                'submodel_metrics': self.submodel_metrics,
                'weights': self.weights,
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'interval': self.interval,
                'prediction_horizon': self.prediction_horizon,
                'feature_window': self.feature_window,
                'model_type': 'ensemble'
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f)
                
            # Save scaler
            scaler_path = os.path.join('models', f"{self.symbol}_{self.interval}_ensemble_scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            
            return True
            
        except Exception as e:
            logging.error(f"Error saving ensemble model for {self.symbol}/{self.interval}: {e}")
            return False
    
    def load_model(self):
        """
        Load a trained ensemble model from disk
        
        Returns:
            Boolean indicating success
        """
        try:
            # Load ensemble model
            self.model = joblib.load(self.model_path)
            
            # Load metadata
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
                
            self.feature_names = metadata['feature_names']
            self.metrics = metadata['metrics']
            self.submodel_metrics = metadata.get('submodel_metrics', {})
            self.weights = metadata.get('weights', [0.5, 0.3, 0.2])
            self.prediction_horizon = metadata.get('prediction_horizon', self.prediction_horizon)
            self.feature_window = metadata.get('feature_window', self.feature_window)
            
            # Load scaler
            scaler_path = os.path.join('models', f"{self.symbol}_{self.interval}_ensemble_scaler.joblib")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            # Try to load individual models if possible
            if os.path.exists(self.gbr_model_path):
                self.gbr_model = joblib.load(self.gbr_model_path)
            
            if os.path.exists(self.rf_model_path):
                self.rf_model = joblib.load(self.rf_model_path)
                
            if os.path.exists(self.lr_model_path):
                self.lr_model = joblib.load(self.lr_model_path)
                
            return True
            
        except Exception as e:
            logging.error(f"Error loading ensemble model for {self.symbol}/{self.interval}: {e}")
            return False
    
    def predict_with_submodels(self):
        """
        Make predictions using each individual sub-model
        
        Returns:
            Dictionary with predictions from each model and the ensemble
        """
        if not all([self.gbr_model, self.rf_model, self.lr_model]):
            logging.error(f"Sub-models not available for {self.symbol}/{self.interval}")
            return None
            
        # Get most recent data for prediction
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)  # Need enough historical data for features
        
        # Get price data
        df = get_historical_data(self.symbol, self.interval, start_time, end_time)
        
        # Get indicators data
        indicators_df = get_indicators(self.symbol, self.interval, start_time, end_time)
        
        # Merge price data with indicators
        if not indicators_df.empty:
            df = pd.merge(df, indicators_df, on='timestamp', how='left')
        
        if df.empty:
            logging.error(f"No recent data available for {self.symbol}/{self.interval}")
            return None
            
        # Prepare features
        X, _ = self.prepare_data(df)
        
        if X is None or X.empty:
            logging.error(f"Failed to prepare features for prediction")
            return None
            
        # Use only the most recent data point for prediction
        X_recent = X.iloc[-1:].copy()
        
        # Scale features
        if self.scaler is not None:
            X_recent_scaled = self.scaler.transform(X_recent)
            
            # Get most recent price
            latest_price = df['close'].iloc[-1]
            
            # Make predictions with each model
            gbr_change = self.gbr_model.predict(X_recent_scaled)[0]
            rf_change = self.rf_model.predict(X_recent_scaled)[0]
            lr_change = self.lr_model.predict(X_recent_scaled)[0]
            
            # Calculate predicted prices
            gbr_price = latest_price * (1 + gbr_change)
            rf_price = latest_price * (1 + rf_change)
            lr_price = latest_price * (1 + lr_change)
            
            # Make ensemble prediction
            ensemble_change = self.model.predict(X_recent_scaled)[0]
            ensemble_price = latest_price * (1 + ensemble_change)
            
            # Calculate confidence scores
            gbr_confidence = self._get_prediction_confidence(gbr_change, model_type='gbr')
            rf_confidence = self._get_prediction_confidence(rf_change, model_type='rf')
            lr_confidence = self._get_prediction_confidence(lr_change, model_type='lr')
            ensemble_confidence = self._get_prediction_confidence(ensemble_change)
            
            # Return all predictions
            return {
                'latest_price': float(latest_price),
                'timestamp': df['timestamp'].iloc[-1],
                'ensemble': {
                    'predicted_change': float(ensemble_change),
                    'predicted_price': float(ensemble_price),
                    'confidence': float(ensemble_confidence),
                    'weight': 1.0  # The ensemble is the final prediction
                },
                'gbr': {
                    'predicted_change': float(gbr_change),
                    'predicted_price': float(gbr_price),
                    'confidence': float(gbr_confidence),
                    'weight': float(self.weights[0])
                },
                'rf': {
                    'predicted_change': float(rf_change),
                    'predicted_price': float(rf_price),
                    'confidence': float(rf_confidence),
                    'weight': float(self.weights[1])
                },
                'lr': {
                    'predicted_change': float(lr_change),
                    'predicted_price': float(lr_price),
                    'confidence': float(lr_confidence),
                    'weight': float(self.weights[2])
                }
            }
            
        else:
            logging.error(f"Scaler not available for {self.symbol}/{self.interval}")
            return None
    
    def _get_prediction_confidence(self, predicted_movement, model_type='ensemble'):
        """
        Calculate confidence score for prediction from a specific model
        
        Args:
            predicted_movement: Predicted price movement
            model_type: Which model to get confidence for ('ensemble', 'gbr', 'rf', 'lr')
            
        Returns:
            Confidence score (0-1)
        """
        # Get appropriate metrics based on model type
        if model_type != 'ensemble' and model_type in self.submodel_metrics:
            metrics = self.submodel_metrics[model_type]
        else:
            metrics = self.metrics
        
        # Base confidence on model metrics if available
        if 'direction_accuracy' in metrics:
            base_confidence = metrics['direction_accuracy']
        else:
            base_confidence = 0.6  # Default if no metrics available
            
        # Adjust confidence based on magnitude of predicted movement
        # Small movements are less certain than larger ones
        magnitude_factor = min(abs(predicted_movement) * 10, 1.0)
        
        # Final confidence is a combination of model accuracy and signal strength
        confidence = base_confidence * 0.7 + magnitude_factor * 0.3
        
        # Ensure confidence is between 0 and 1
        return min(max(confidence, 0.0), 1.0)


# Function to train ensemble models for multiple symbol/interval combinations
def train_ensemble_models(symbols, intervals, lookback_days=90, retrain=False):
    """
    Train ensemble models for all symbol/interval combinations
    
    Args:
        symbols: List of symbols to train models for
        intervals: List of intervals to train models for
        lookback_days: How many days of historical data to use
        retrain: Whether to retrain existing models
        
    Returns:
        Dictionary with training results
    """
    results = {}
    
    for symbol in symbols:
        results[symbol] = {}
        
        for interval in intervals:
            logging.info(f"Training ensemble model for {symbol}/{interval}")
            
            try:
                # Initialize and train ensemble model
                predictor = EnsemblePredictor(symbol, interval)
                success = predictor.train_model(lookback_days=lookback_days, retrain=retrain)
                
                if success:
                    # Get metrics
                    results[symbol][interval] = {
                        'status': 'success',
                        'metrics': predictor.metrics,
                        'submodel_metrics': predictor.submodel_metrics,
                        'weights': predictor.weights
                    }
                else:
                    results[symbol][interval] = {
                        'status': 'failed',
                        'error': 'Training failed'
                    }
            except Exception as e:
                logging.error(f"Error training ensemble model for {symbol}/{interval}: {e}")
                results[symbol][interval] = {
                    'status': 'error',
                    'error': str(e)
                }
                
    return results