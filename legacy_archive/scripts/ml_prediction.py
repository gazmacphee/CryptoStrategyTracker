"""
Machine Learning Module for Cryptocurrency Price Prediction

This module provides functionality to:
1. Prepare data from the database for machine learning
2. Train prediction models using various algorithms
3. Evaluate model performance
4. Make predictions on future price movements
5. Continuously update models with new data
"""

import os
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import time
import json

# ML libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Local imports
from database import get_historical_data, get_indicators
from utils import timeframe_to_seconds

class MLPredictor:
    """
    Machine Learning Predictor for cryptocurrency price movements
    """
    
    def __init__(self, symbol, interval, prediction_horizon=24, feature_window=30):
        """
        Initialize the ML predictor
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Time interval (e.g., '1h', '4h', '1d')
            prediction_horizon: How many periods ahead to predict
            feature_window: How many historical periods to use for features
        """
        self.symbol = symbol
        self.interval = interval
        self.prediction_horizon = prediction_horizon
        self.feature_window = feature_window
        self.model = None
        self.feature_names = None
        self.scaler = None
        self.metrics = {}
        self.model_path = os.path.join('models', f"{symbol}_{interval}_model.joblib")
        self.metadata_path = os.path.join('models', f"{symbol}_{interval}_metadata.json")
        
        # Ensure model directory exists
        os.makedirs('models', exist_ok=True)
    
    def prepare_data(self, df, target_column='close'):
        """
        Prepare data for machine learning
        
        Args:
            df: DataFrame with historical data
            target_column: Column to predict (default: 'close')
            
        Returns:
            X: Features DataFrame
            y: Target Series
        """
        if df.empty:
            logging.warning(f"Empty dataframe provided for {self.symbol}/{self.interval}")
            return None, None
            
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Create lagged features for OHLCV data
        for col in ['open', 'high', 'low', 'close', 'volume']:
            for i in range(1, self.feature_window + 1):
                df[f'{col}_lag_{i}'] = df[col].shift(i)
        
        # Calculate price changes for different periods
        for period in [1, 3, 5, 10, 20]:
            df[f'price_change_{period}'] = df['close'].pct_change(periods=period)
        
        # Calculate moving averages
        for window in [7, 14, 30, 50, 100]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            
        # Add technical indicators like RSI and MACD if available
        if 'rsi' in df.columns:
            for i in range(1, 5):
                df[f'rsi_lag_{i}'] = df['rsi'].shift(i)
                
        if 'macd' in df.columns:
            for i in range(1, 5):
                df[f'macd_lag_{i}'] = df['macd'].shift(i)
                
        if 'macd_signal' in df.columns:
            for i in range(1, 5):
                df[f'macd_signal_lag_{i}'] = df['macd_signal'].shift(i)
                
        if 'bb_lower' in df.columns and 'bb_upper' in df.columns:
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
            for i in range(1, 5):
                df[f'bb_width_lag_{i}'] = df['bb_width'].shift(i)
        
        # Create target variable - future price change
        df['target'] = df[target_column].pct_change(periods=self.prediction_horizon).shift(-self.prediction_horizon)
        
        # Drop rows with NaN (due to lagging/leading operations)
        df = df.dropna()
        
        if df.empty:
            logging.warning(f"After feature creation, dataframe is empty for {self.symbol}/{self.interval}")
            return None, None
        
        # Exclude timestamp, target_column, and other non-feature columns
        exclude_cols = ['timestamp', 'target', target_column, 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Create feature matrix and target vector
        X = df[feature_cols]
        y = df['target']
        
        return X, y
    
    def train_model(self, lookback_days=90, retrain=False):
        """
        Train the prediction model
        
        Args:
            lookback_days: How many days of historical data to use
            retrain: Whether to retrain an existing model or create a new one
            
        Returns:
            Boolean indicating success
        """
        # Check if model already exists and we don't want to retrain
        if os.path.exists(self.model_path) and not retrain:
            logging.info(f"Model for {self.symbol}/{self.interval} already exists. Loading model.")
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
        
        # Create model
        self.model = self._create_model()
        
        # Train model
        try:
            self.model.fit(X_train_scaled, y_train)
            
            # Calculate metrics
            y_pred = self.model.predict(X_test_scaled)
            self.metrics = self._calculate_metrics(y_test, y_pred)
            
            # Store feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                feature_importances = self.model.feature_importances_
                self.metrics['feature_importance'] = dict(zip(self.feature_names, feature_importances))
                
            # Save model
            self.save_model()
            
            logging.info(f"Successfully trained model for {self.symbol}/{self.interval}")
            return True
            
        except Exception as e:
            logging.error(f"Error training model for {self.symbol}/{self.interval}: {e}")
            return False
    
    def _create_model(self):
        """
        Create a machine learning model
        
        Returns:
            Initialized ML model
        """
        # Use Gradient Boosting Regressor for price prediction - good balance of performance and accuracy
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        return model
    
    def _calculate_metrics(self, y_true, y_pred):
        """
        Calculate performance metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'timestamp': datetime.now().isoformat()
        }
        
        # Direction accuracy - how often we correctly predict up/down
        direction_correct = np.sum((y_true > 0) == (y_pred > 0)) / len(y_true)
        metrics['direction_accuracy'] = direction_correct
        
        return metrics
    
    def predict_future(self, periods=1):
        """
        Make predictions for future price movements
        
        Args:
            periods: Number of future periods to predict
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            if os.path.exists(self.model_path):
                self.load_model()
            else:
                logging.error(f"No trained model found for {self.symbol}/{self.interval}")
                return pd.DataFrame()
                
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
            return pd.DataFrame()
            
        # Prepare features (but ignore the target since we're predicting)
        X, _ = self.prepare_data(df)
        
        if X is None or X.empty:
            logging.error(f"Failed to prepare features for prediction")
            return pd.DataFrame()
            
        # Use only the most recent data point for prediction
        X_recent = X.iloc[-1:].copy()
        
        # Scale features
        if self.scaler is not None:
            X_recent_scaled = self.scaler.transform(X_recent)
            
            # Make prediction
            predicted_change = self.model.predict(X_recent_scaled)[0]
        else:
            logging.error(f"Scaler not available for {self.symbol}/{self.interval}")
            return pd.DataFrame()
            
        # Get most recent price
        latest_price = df['close'].iloc[-1]
        latest_time = df['timestamp'].iloc[-1]
        
        # Calculate predicted price
        predicted_price = latest_price * (1 + predicted_change)
        
        # Calculate confidence score
        confidence = self._get_prediction_confidence(predicted_change)
        
        # Prepare result dataframe
        time_delta = self._get_period_delta()
        prediction_data = []
        
        for i in range(1, periods + 1):
            predicted_time = latest_time + (time_delta * i)
            prediction_data.append({
                'timestamp': predicted_time,
                'predicted_price': predicted_price if i == 1 else None,  # Only include first prediction
                'predicted_change': predicted_change if i == 1 else None,
                'confidence': confidence if i == 1 else None,
                'is_prediction': True
            })
            
        predictions_df = pd.DataFrame(prediction_data)
        
        # Add historical data
        historical_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        historical_df['is_prediction'] = False
        
        # Combine historical and prediction data
        result_df = pd.concat([historical_df, predictions_df], ignore_index=True)
        result_df = result_df.sort_values('timestamp')
        
        return result_df
    
    def _get_period_delta(self):
        """
        Get timedelta for one period based on interval
        
        Returns:
            timedelta object
        """
        # Convert interval to seconds
        seconds = timeframe_to_seconds(self.interval)
        return timedelta(seconds=seconds)
    
    def _get_prediction_confidence(self, predicted_movement):
        """
        Calculate confidence score for prediction
        
        Args:
            predicted_movement: Predicted price movement
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence on model metrics if available
        if 'direction_accuracy' in self.metrics:
            base_confidence = self.metrics['direction_accuracy']
        else:
            base_confidence = 0.6  # Default if no metrics available
            
        # Adjust confidence based on magnitude of predicted movement
        # Small movements are less certain than larger ones
        magnitude_factor = min(abs(predicted_movement) * 10, 1.0)
        
        # Final confidence is a combination of model accuracy and signal strength
        confidence = base_confidence * 0.7 + magnitude_factor * 0.3
        
        # Ensure confidence is between 0 and 1
        return min(max(confidence, 0.0), 1.0)
    
    def save_model(self):
        """
        Save the trained model to disk
        
        Returns:
            Boolean indicating success
        """
        try:
            # Save model
            joblib.dump(self.model, self.model_path)
            
            # Save metadata (feature names, scaler, metrics)
            metadata = {
                'feature_names': self.feature_names,
                'metrics': self.metrics,
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'interval': self.interval,
                'prediction_horizon': self.prediction_horizon,
                'feature_window': self.feature_window
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f)
                
            # Save scaler
            scaler_path = os.path.join('models', f"{self.symbol}_{self.interval}_scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            
            return True
            
        except Exception as e:
            logging.error(f"Error saving model for {self.symbol}/{self.interval}: {e}")
            return False
    
    def load_model(self):
        """
        Load a trained model from disk
        
        Returns:
            Boolean indicating success
        """
        try:
            # Load model
            self.model = joblib.load(self.model_path)
            
            # Load metadata
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
                
            self.feature_names = metadata['feature_names']
            self.metrics = metadata['metrics']
            self.prediction_horizon = metadata.get('prediction_horizon', self.prediction_horizon)
            self.feature_window = metadata.get('feature_window', self.feature_window)
            
            # Load scaler
            scaler_path = os.path.join('models', f"{self.symbol}_{self.interval}_scaler.joblib")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                
            return True
            
        except Exception as e:
            logging.error(f"Error loading model for {self.symbol}/{self.interval}: {e}")
            return False


def train_all_models(symbols, intervals, lookback_days=90, retrain=False):
    """
    Train models for all symbol/interval combinations
    
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
            logging.info(f"Training model for {symbol}/{interval}")
            
            try:
                # Initialize and train model
                predictor = MLPredictor(symbol, interval)
                success = predictor.train_model(lookback_days=lookback_days, retrain=retrain)
                
                if success:
                    # Get metrics
                    results[symbol][interval] = {
                        'status': 'success',
                        'metrics': predictor.metrics
                    }
                else:
                    results[symbol][interval] = {
                        'status': 'failed',
                        'error': 'Training failed'
                    }
            except Exception as e:
                logging.error(f"Error training model for {symbol}/{interval}: {e}")
                results[symbol][interval] = {
                    'status': 'error',
                    'error': str(e)
                }
                
    return results


def predict_for_all(symbols, intervals, periods=1):
    """
    Make predictions for all symbol/interval combinations
    
    Args:
        symbols: List of symbols to make predictions for
        intervals: List of intervals to make predictions for
        periods: Number of periods to predict
        
    Returns:
        Dictionary with prediction results
    """
    predictions = {}
    
    for symbol in symbols:
        predictions[symbol] = {}
        
        for interval in intervals:
            logging.info(f"Making predictions for {symbol}/{interval}")
            
            try:
                # Initialize predictor and load model
                predictor = MLPredictor(symbol, interval)
                
                # Check if model exists
                if os.path.exists(predictor.model_path):
                    # Make prediction
                    pred_df = predictor.predict_future(periods=periods)
                    
                    if not pred_df.empty:
                        # Get the prediction row
                        pred_row = pred_df[pred_df['is_prediction'] == True].iloc[0]
                        
                        predictions[symbol][interval] = {
                            'status': 'success',
                            'timestamp': pred_row['timestamp'].isoformat(),
                            'predicted_price': float(pred_row['predicted_price']),
                            'predicted_change': float(pred_row['predicted_change']),
                            'confidence': float(pred_row['confidence']),
                            'direction': 'up' if pred_row['predicted_change'] > 0 else 'down'
                        }
                    else:
                        predictions[symbol][interval] = {
                            'status': 'failed',
                            'error': 'No predictions generated'
                        }
                else:
                    predictions[symbol][interval] = {
                        'status': 'no_model',
                        'error': 'No trained model available'
                    }
            except Exception as e:
                logging.error(f"Error making predictions for {symbol}/{interval}: {e}")
                predictions[symbol][interval] = {
                    'status': 'error',
                    'error': str(e)
                }
                
    return predictions


def continuous_learning_cycle(symbols, intervals, retraining_interval_hours=24):
    """
    Run continuous learning cycle that periodically retrains models with new data
    
    Args:
        symbols: List of symbols to train models for
        intervals: List of intervals to train models for
        retraining_interval_hours: How often to retrain models (in hours)
        
    Note: This function runs indefinitely until interrupted
    """
    while True:
        logging.info(f"Starting continuous learning cycle")
        
        # Train all models with retraining enabled
        train_all_models(symbols, intervals, retrain=True)
        
        # Log completion
        logging.info(f"Continuous learning cycle completed. Next update in {retraining_interval_hours} hours")
        
        # Sleep until next update
        sleep_seconds = retraining_interval_hours * 3600
        time.sleep(sleep_seconds)