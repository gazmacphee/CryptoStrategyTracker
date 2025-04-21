"""
Machine Learning Module for Cryptocurrency Price Prediction

This module provides functionality to:
1. Prepare data from the database for machine learning
2. Train prediction models using various algorithms
3. Evaluate model performance
4. Make predictions on future price movements
5. Continuously update models with new data
"""

import numpy as np
import pandas as pd
import logging
import pickle
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
import joblib

# Import local modules
from database import get_db_connection, get_historical_data
import indicators

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='ml_prediction.log',
                    filemode='a')

# Create models directory if it doesn't exist
MODELS_DIR = 'models'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

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
        self.scaler = MinMaxScaler()
        self.feature_columns = []
        self.model_filename = f"{MODELS_DIR}/{symbol}_{interval}_predictor.joblib"
        self.metrics_history = []
        
        # Try to load existing model
        if os.path.exists(self.model_filename):
            try:
                self.load_model()
                logging.info(f"Loaded existing model for {symbol}/{interval}")
            except Exception as e:
                logging.error(f"Error loading model for {symbol}/{interval}: {e}")
                self.model = None
    
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
        # Ensure data is sorted by timestamp
        df = df.sort_values('timestamp')
        
        # Calculate technical indicators if not present
        if 'bb_upper' not in df.columns:
            df = indicators.add_bollinger_bands(df)
        if 'rsi' not in df.columns:
            df = indicators.add_rsi(df)
        if 'macd' not in df.columns:
            df = indicators.add_macd(df)
        if 'ema_9' not in df.columns:
            df = indicators.add_ema(df)
            
        # Calculate price changes
        df['price_change'] = df[target_column].pct_change()
        df['price_change_1d'] = df[target_column].pct_change(periods=24 if self.interval == '1h' else 
                                                           (48 if self.interval == '30m' else 96))
        df['volatility'] = df[target_column].rolling(window=self.feature_window).std()
        
        # Create target: future price movement
        df[f'future_{target_column}'] = df[target_column].shift(-self.prediction_horizon)
        df[f'target_movement'] = df[f'future_{target_column}'] / df[target_column] - 1.0
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Feature selection
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'ema_9', 'ema_21', 'price_change', 'price_change_1d', 'volatility'
        ]
        
        # Create feature matrix and target vector
        X = df[self.feature_columns].values
        y = df['target_movement'].values
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Create sequences for time series forecasting
        X_sequences, y_sequences = [], []
        for i in range(len(df) - self.feature_window):
            X_sequences.append(X[i:i+self.feature_window])
            y_sequences.append(y[i+self.feature_window])
            
        return np.array(X_sequences), np.array(y_sequences)
    
    def train_model(self, lookback_days=90, retrain=False):
        """
        Train the prediction model
        
        Args:
            lookback_days: How many days of historical data to use
            retrain: Whether to retrain an existing model or create a new one
            
        Returns:
            Boolean indicating success
        """
        try:
            # Get historical data
            end_time = datetime.now()
            start_time = end_time - timedelta(days=lookback_days)
            df = get_historical_data(self.symbol, self.interval, start_time, end_time)
            
            if df.empty:
                logging.error(f"No data available for {self.symbol}/{self.interval}")
                return False
                
            logging.info(f"Training model for {self.symbol}/{self.interval} with {len(df)} data points")
            
            # Prepare data
            X, y = self.prepare_data(df)
            
            if len(X) < 100:  # Require at least 100 samples
                logging.warning(f"Insufficient data for {self.symbol}/{self.interval}: {len(X)} samples")
                return False
                
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Create new model or use existing one
            if self.model is None or retrain:
                self.model = self._create_model()
                
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred)
            self.metrics_history.append({
                'timestamp': datetime.now(),
                'metrics': metrics
            })
            
            # Save model
            self.save_model()
            
            logging.info(f"Model training completed for {self.symbol}/{self.interval}. Metrics: {metrics}")
            return True
            
        except Exception as e:
            logging.error(f"Error training model for {self.symbol}/{self.interval}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    def _create_model(self):
        """
        Create a machine learning model
        
        Returns:
            Initialized ML model
        """
        # For sequence data, reshape is needed for non-RNN models
        return MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=1000,
            shuffle=False,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    
    def _calculate_metrics(self, y_true, y_pred):
        """
        Calculate performance metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with performance metrics
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'correct_direction': np.mean((y_true > 0) == (y_pred > 0))
        }
    
    def predict_future(self, periods=1):
        """
        Make predictions for future price movements
        
        Args:
            periods: Number of future periods to predict
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            logging.error(f"No model available for {self.symbol}/{self.interval}")
            return None
            
        try:
            # Get recent data
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)  # Get enough data for features
            df = get_historical_data(self.symbol, self.interval, start_time, end_time)
            
            if df.empty:
                logging.error(f"No recent data available for {self.symbol}/{self.interval}")
                return None
                
            # Prepare data - only need the most recent window
            df = df.sort_values('timestamp')
            
            # Calculate technical indicators if not present
            if 'bb_upper' not in df.columns:
                df = indicators.add_bollinger_bands(df)
            if 'rsi' not in df.columns:
                df = indicators.add_rsi(df)
            if 'macd' not in df.columns:
                df = indicators.add_macd(df)
            if 'ema_9' not in df.columns:
                df = indicators.add_ema(df)
                
            # Calculate price changes
            df['price_change'] = df['close'].pct_change()
            df['price_change_1d'] = df['close'].pct_change(periods=24 if self.interval == '1h' else 
                                                         (48 if self.interval == '30m' else 96))
            df['volatility'] = df['close'].rolling(window=self.feature_window).std()
            
            # Drop rows with NaN values
            df = df.dropna()
            
            if len(df) < self.feature_window:
                logging.error(f"Insufficient recent data for {self.symbol}/{self.interval}")
                return None
                
            # Extract features
            features = df[self.feature_columns].values
            
            # Scale features
            scaled_features = self.scaler.transform(features)
            
            # Get the most recent window
            recent_window = scaled_features[-self.feature_window:].reshape(1, self.feature_window, -1)
            
            # Make prediction
            predicted_movement = self.model.predict(recent_window)[0]
            
            # Get the most recent close price
            last_price = df['close'].iloc[-1]
            predicted_price = last_price * (1 + predicted_movement)
            
            # Create result DataFrame
            last_timestamp = df['timestamp'].iloc[-1]
            period_delta = self._get_period_delta()
            
            prediction_df = pd.DataFrame({
                'timestamp': [last_timestamp + (i+1)*period_delta for i in range(periods)],
                'predicted_movement': [predicted_movement] * periods,
                'predicted_price': [predicted_price] * periods,
                'confidence': [self._get_prediction_confidence(predicted_movement)] * periods
            })
            
            logging.info(f"Made prediction for {self.symbol}/{self.interval}: {predicted_movement:.2%}")
            return prediction_df
            
        except Exception as e:
            logging.error(f"Error making prediction for {self.symbol}/{self.interval}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None
    
    def _get_period_delta(self):
        """
        Get timedelta for one period based on interval
        
        Returns:
            timedelta object
        """
        if self.interval == '1m':
            return timedelta(minutes=1)
        elif self.interval == '5m':
            return timedelta(minutes=5)
        elif self.interval == '15m':
            return timedelta(minutes=15)
        elif self.interval == '30m':
            return timedelta(minutes=30)
        elif self.interval == '1h':
            return timedelta(hours=1)
        elif self.interval == '4h':
            return timedelta(hours=4)
        elif self.interval == '1d':
            return timedelta(days=1)
        else:
            return timedelta(hours=1)  # Default
    
    def _get_prediction_confidence(self, predicted_movement):
        """
        Calculate confidence score for prediction
        
        Args:
            predicted_movement: Predicted price movement
            
        Returns:
            Confidence score (0-1)
        """
        # Simple heuristic - higher absolute movement means lower confidence
        return max(0.5, 1.0 - min(1.0, abs(predicted_movement) * 5))
    
    def save_model(self):
        """
        Save the trained model to disk
        
        Returns:
            Boolean indicating success
        """
        try:
            if self.model is not None:
                joblib.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_columns': self.feature_columns,
                    'metrics_history': self.metrics_history,
                    'updated_at': datetime.now().isoformat()
                }, self.model_filename)
                logging.info(f"Model saved to {self.model_filename}")
                return True
            return False
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
            if os.path.exists(self.model_filename):
                model_data = joblib.load(self.model_filename)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_columns = model_data['feature_columns']
                self.metrics_history = model_data.get('metrics_history', [])
                logging.info(f"Model loaded from {self.model_filename}")
                return True
            return False
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
            predictor = MLPredictor(symbol, interval)
            success = predictor.train_model(lookback_days, retrain)
            results[symbol][interval] = success
            
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
    results = {}
    
    for symbol in symbols:
        results[symbol] = {}
        for interval in intervals:
            logging.info(f"Making prediction for {symbol}/{interval}")
            predictor = MLPredictor(symbol, interval)
            
            if not os.path.exists(predictor.model_filename):
                # Train model if it doesn't exist
                success = predictor.train_model()
                if not success:
                    results[symbol][interval] = None
                    continue
                    
            predictions = predictor.predict_future(periods)
            results[symbol][interval] = predictions
            
    return results

def continuous_learning_cycle(symbols, intervals, retraining_interval_hours=24):
    """
    Run continuous learning cycle that periodically retrains models with new data
    
    Args:
        symbols: List of symbols to train models for
        intervals: List of intervals to train models for
        retraining_interval_hours: How often to retrain models (in hours)
        
    Note: This function runs indefinitely until interrupted
    """
    import time
    
    logging.info("Starting continuous learning cycle")
    
    while True:
        try:
            # Train all models
            train_all_models(symbols, intervals, retrain=True)
            
            # Wait for the specified interval
            logging.info(f"Waiting {retraining_interval_hours} hours until next retraining cycle")
            time.sleep(retraining_interval_hours * 3600)
            
        except KeyboardInterrupt:
            logging.info("Continuous learning cycle interrupted by user")
            break
            
        except Exception as e:
            logging.error(f"Error in continuous learning cycle: {e}")
            import traceback
            logging.error(traceback.format_exc())
            time.sleep(3600)  # Wait an hour and try again

if __name__ == "__main__":
    # Example usage
    from backfill_database import get_popular_symbols
    
    # Use a subset of popular symbols
    symbols = get_popular_symbols(limit=5)
    intervals = ['1h', '4h', '1d']
    
    # Train models
    train_all_models(symbols, intervals)
    
    # Make predictions
    predictions = predict_for_all(symbols, intervals)
    
    # Print predictions
    for symbol in predictions:
        for interval in predictions[symbol]:
            if predictions[symbol][interval] is not None:
                print(f"\nPredictions for {symbol}/{interval}:")
                print(predictions[symbol][interval])