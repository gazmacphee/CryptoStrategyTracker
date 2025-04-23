"""
Simplified ML module for basic predictions and backtesting.
This module implements core ML functionality that was previously archived.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os
import logging
from typing import Dict, List, Tuple, Optional, Union

# Import ML database operations
try:
    from db_ml_operations import save_ml_prediction, save_ml_model_performance
    ML_DB_AVAILABLE = True
except ImportError:
    print("Warning: db_ml_operations module not available. ML data will not be saved to database.")
    ML_DB_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

class SimplePredictionModel:
    """A simplified prediction model for cryptocurrency price forecasting"""
    
    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol
        self.interval = interval
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'ema_9', 'ema_21', 'ema_50', 'ema_200',
        ]
        self.target_column = 'close'
        self.model_path = os.path.join(MODEL_DIR, f'model_{symbol}_{interval}.joblib')
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target data for training
        
        Args:
            df: DataFrame with OHLCV and indicator data
            
        Returns:
            Tuple of (X, y) arrays ready for model training
        """
        # Ensure all necessary columns are present
        available_features = [col for col in self.feature_columns if col in df.columns]
        
        if len(available_features) < 5:
            logger.warning(f"Not enough features available. Using default OHLCV columns.")
            available_features = [col for col in ['open', 'high', 'low', 'close', 'volume'] 
                                if col in df.columns]
        
        # Drop rows with NaN values
        df_clean = df.dropna(subset=available_features + [self.target_column])
        
        if len(df_clean) == 0:
            logger.error("No valid data after removing NaN values")
            return np.array([]), np.array([])
        
        # Create sequences: use 10 past days to predict the next day
        X_seq, y_seq = [], []
        sequence_length = 10
        
        for i in range(len(df_clean) - sequence_length):
            X_seq.append(df_clean[available_features].iloc[i:i+sequence_length].values)
            y_seq.append(df_clean[self.target_column].iloc[i+sequence_length])
        
        X = np.array(X_seq)
        y = np.array(y_seq).reshape(-1, 1)
        
        # Scale features
        n_samples, n_steps, n_features = X.shape
        X_reshaped = X.reshape(n_samples * n_steps, n_features)
        X_scaled = self.scaler_X.fit_transform(X_reshaped)
        X_final = X_scaled.reshape(n_samples, n_steps, n_features)
        
        # Flatten X_final for Random Forest (which doesn't accept 3D inputs)
        X_flat = X_final.reshape(n_samples, n_steps * n_features)
        
        # Scale target
        y_scaled = self.scaler_y.fit_transform(y)
        
        return X_flat, y_scaled.flatten()
    
    def train(self, df: pd.DataFrame) -> bool:
        """
        Train the prediction model
        
        Args:
            df: DataFrame with OHLCV and indicator data
            
        Returns:
            True if training was successful, False otherwise
        """
        try:
            X, y = self.prepare_features(df)
            
            if len(X) == 0 or len(y) == 0:
                logger.error("No data available for training")
                return False
            
            # Create and train the model
            self.model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42
            )
            
            self.model.fit(X, y)
            
            # Save the model
            joblib.dump({
                'model': self.model,
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y
            }, self.model_path)
            
            # Evaluate and save model performance metrics
            if ML_DB_AVAILABLE:
                try:
                    # Split data for validation
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    y_pred = self.model.predict(X_test)
                    
                    # Calculate performance metrics
                    mse = ((y_pred - y_test) ** 2).mean()
                    mae = abs(y_pred - y_test).mean()
                    
                    # For regression tasks, we use MSE/MAE as primary metrics
                    # Convert predictions to binary (up/down) for classification metrics
                    y_pred_class = (y_pred > y_test).astype(int)
                    y_test_class = (y_test > y_test.mean()).astype(int)
                    
                    # Calculate classification metrics for directional accuracy
                    accuracy = (y_pred_class == y_test_class).mean()
                    
                    # Avoid division by zero warnings
                    if sum(y_test_class) > 0 and sum(y_pred_class) > 0:
                        precision = (y_pred_class & y_test_class).sum() / max(1, sum(y_pred_class))
                        recall = (y_pred_class & y_test_class).sum() / max(1, sum(y_test_class))
                        f1 = 2 * precision * recall / max(0.001, precision + recall)
                    else:
                        precision = recall = f1 = 0.0
                    
                    # Save metrics to database
                    save_ml_model_performance(
                        model_name='simple_price_predictor',
                        symbol=self.symbol,
                        interval=self.interval,
                        training_timestamp=datetime.now(),
                        accuracy=float(accuracy),
                        precision_score=float(precision),
                        recall=float(recall),
                        f1_score=float(f1),
                        mse=float(mse),
                        mae=float(mae),
                        training_params={
                            'n_estimators': 50,
                            'max_depth': 10,
                            'features_used': self.feature_columns
                        }
                    )
                    logger.info(f"Saved model performance metrics to database for {self.symbol}/{self.interval}")
                except Exception as e:
                    logger.error(f"Error saving model performance metrics: {e}")
            
            logger.info(f"Model trained and saved to {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def load(self) -> bool:
        """
        Load a previously trained model
        
        Returns:
            True if loading was successful, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"No saved model found at {self.model_path}")
                return False
            
            saved_data = joblib.load(self.model_path)
            self.model = saved_data['model']
            self.scaler_X = saved_data['scaler_X']
            self.scaler_y = saved_data['scaler_y']
            
            logger.info(f"Model loaded from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, df: pd.DataFrame, days_ahead: int = 7) -> pd.DataFrame:
        """
        Generate predictions for future days
        
        Args:
            df: DataFrame with recent data
            days_ahead: Number of days to predict ahead
            
        Returns:
            DataFrame with predictions
        """
        try:
            if self.model is None:
                success = self.load()
                if not success:
                    # If no model is saved, try to train one
                    success = self.train(df)
                    if not success:
                        logger.error("Could not load or train a model")
                        return self._generate_empty_predictions(days_ahead)
            
            # Prepare the most recent data
            X, _ = self.prepare_features(df)
            
            if len(X) == 0:
                logger.error("Could not prepare features for prediction")
                return self._generate_empty_predictions(days_ahead)
            
            # Use the most recent sequence for prediction
            last_sequence = X[-1].reshape(1, -1)
            
            # Generate predictions
            results = []
            current_input = last_sequence.copy()
            
            last_known_price = df['close'].iloc[-1]
            confidence_factor = 0.9  # Decreasing confidence for future predictions
            
            for i in range(days_ahead):
                # Make prediction
                pred_scaled = self.model.predict(current_input)
                pred_value = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
                
                # Calculate uncertainty (simplified approach)
                uncertainty = abs(pred_value - last_known_price) * 0.1 * (i + 1)
                lower_bound = max(0, pred_value - uncertainty)
                upper_bound = pred_value + uncertainty
                
                # Adjust confidence for future days
                confidence = confidence_factor ** (i + 1)
                
                # Store result
                pred_date = datetime.now() + timedelta(days=i+1)
                results.append({
                    'date': pred_date,
                    'predicted_price': pred_value,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'confidence': confidence
                })
                
                # Save prediction to database if available
                if ML_DB_AVAILABLE:
                    try:
                        # Calculate percent change based on latest known price
                        predicted_change_pct = (pred_value - last_known_price) / last_known_price
                        
                        # Create features used dictionary (simplified)
                        features_used = {
                            'lookback_days': 10,
                            'indicators': [col for col in self.feature_columns if col in df.columns],
                            'price_features': ['close', 'open', 'high', 'low'],
                            'model_type': 'RandomForestRegressor'
                        }
                        
                        # Save prediction to database
                        save_ml_prediction(
                            symbol=self.symbol,
                            interval=self.interval,
                            prediction_timestamp=datetime.now(),
                            target_timestamp=pred_date,
                            model_name='simple_price_predictor',
                            predicted_price=float(pred_value),
                            predicted_change_pct=float(predicted_change_pct),
                            confidence_score=float(confidence),
                            features_used=features_used,
                            prediction_type='price'
                        )
                    except Exception as e:
                        logger.error(f"Error saving prediction to database: {e}")
                
                # Update current_input for the next prediction
                # This is a simplified approach - we would need to update all features in a real model
                
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return self._generate_empty_predictions(days_ahead)
    
    def _generate_empty_predictions(self, days_ahead: int) -> pd.DataFrame:
        """Generate empty prediction results for error cases"""
        results = []
        for i in range(days_ahead):
            pred_date = datetime.now() + timedelta(days=i+1)
            results.append({
                'date': pred_date,
                'predicted_price': None,
                'lower_bound': None,
                'upper_bound': None,
                'confidence': None
            })
        return pd.DataFrame(results)


class MarketRegimeModel:
    """A simple model for detecting market regimes (bullish, bearish, sideways)"""
    
    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol
        self.interval = interval
    
    def detect_regime(self, df: pd.DataFrame) -> Dict:
        """
        Detect the current market regime
        
        Args:
            df: DataFrame with price history
            
        Returns:
            Dictionary with regime information
        """
        try:
            if len(df) < 20:
                return {
                    'regime': 'unknown',
                    'probabilities': {
                        'bullish': 0.33,
                        'bearish': 0.33,
                        'sideways': 0.34
                    },
                    'confidence': 0.0,
                    'regime_history': []
                }
            
            # Calculate simple trend indicators
            df = df.copy()
            df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
            
            # Calculate daily returns
            df['return'] = df['close'].pct_change()
            
            # Last 20 days for regime detection
            recent = df.iloc[-20:].copy()
            
            # Calculate volatility
            volatility = recent['return'].std()
            
            # Calculate trend direction
            ema_trend = recent['ema20'].iloc[-1] > recent['ema20'].iloc[0]
            price_change = (recent['close'].iloc[-1] / recent['close'].iloc[0]) - 1
            
            # Determine regime
            if price_change > 0.05 and ema_trend:
                regime = 'bullish'
                bull_prob, bear_prob, sideways_prob = 0.7, 0.1, 0.2
            elif price_change < -0.05 and not ema_trend:
                regime = 'bearish'
                bull_prob, bear_prob, sideways_prob = 0.1, 0.7, 0.2
            else:
                regime = 'sideways'
                bull_prob, bear_prob, sideways_prob = 0.2, 0.2, 0.6
            
            # Adjust for volatility
            confidence = max(0.5, min(0.9, 1.0 - volatility * 10))
            
            # Generate regime history (simplified)
            history = []
            for i in range(min(90, len(df))):
                day = datetime.now() - timedelta(days=90-i)
                if i < 30:
                    r = 'bearish' if df['return'].iloc[i] < 0 else 'bullish'
                elif i < 60:
                    r = 'sideways'
                else:
                    r = regime
                history.append({'date': day, 'regime': r})
            
            return {
                'regime': regime,
                'probabilities': {
                    'bullish': bull_prob,
                    'bearish': bear_prob,
                    'sideways': sideways_prob
                },
                'confidence': confidence,
                'regime_history': history
            }
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return {
                'regime': 'unknown',
                'probabilities': {
                    'bullish': 0.33,
                    'bearish': 0.33,
                    'sideways': 0.34
                },
                'confidence': 0.0,
                'regime_history': []
            }


class BacktestEngine:
    """Simple engine for backtesting ML-based trading strategies"""
    
    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol
        self.interval = interval
        
    def run_backtest(self, df: pd.DataFrame, strategy_params: Dict = None) -> Dict:
        """
        Run a backtest on historical data
        
        Args:
            df: DataFrame with historical price and indicator data
            strategy_params: Parameters for the strategy
            
        Returns:
            Dictionary with backtest results
        """
        try:
            if len(df) < 100:
                logger.warning("Not enough data for meaningful backtest")
                return self._empty_results()
            
            # Default strategy parameters
            if strategy_params is None:
                strategy_params = {
                    'prediction_threshold': 0.02,  # Buy when predicted increase > 2%
                    'stop_loss': 0.05,             # 5% stop loss
                    'take_profit': 0.1,            # 10% take profit
                    'max_holding_days': 14,        # Max holding period
                    'confidence_threshold': 0.6    # Minimum prediction confidence
                }
            
            # Prepare dataframe
            backtest_df = df.copy()
            
            # Train prediction model
            model = SimplePredictionModel(self.symbol, self.interval)
            model.train(backtest_df)
            
            # Simulate trading
            initial_balance = 10000.0
            balance = initial_balance
            position = None
            position_size = 0
            trades = []
            
            # Use 80% of data for training, 20% for testing
            train_size = int(len(backtest_df) * 0.8)
            train_df = backtest_df.iloc[:train_size]
            test_df = backtest_df.iloc[train_size:]
            
            # Testing period
            for i in range(len(test_df) - 5):
                current_date = test_df.index[i]
                current_price = test_df['close'].iloc[i]
                
                # Generate prediction using data up to this point
                historical = pd.concat([train_df, test_df.iloc[:i]])
                
                # Only make new predictions every 5 days to reduce computation
                if i % 5 == 0:
                    prediction_df = model.predict(historical, days_ahead=7)
                    future_prices = prediction_df['predicted_price'].values
                    confidence = prediction_df['confidence'].mean() if 'confidence' in prediction_df.columns else 0.6
                    
                    if len(future_prices) >= 5 and future_prices[4] is not None:
                        predicted_change = (future_prices[4] / current_price) - 1
                    else:
                        predicted_change = 0
                
                # Trading logic
                if position is None:  # No position
                    if (predicted_change > strategy_params['prediction_threshold'] and 
                            confidence > strategy_params['confidence_threshold']):
                        # Buy
                        position = current_price
                        position_size = balance * 0.95 / current_price  # Invest 95% of balance
                        balance -= position_size * current_price
                        trades.append({
                            'date': current_date,
                            'type': 'buy',
                            'price': current_price,
                            'size': position_size,
                            'balance': balance + position_size * current_price
                        })
                else:  # In position
                    days_in_position = i - trades[-1]['date']
                    price_change = (current_price / position) - 1
                    
                    # Check exit conditions
                    if (price_change <= -strategy_params['stop_loss'] or 
                            price_change >= strategy_params['take_profit'] or 
                            days_in_position >= strategy_params['max_holding_days']):
                        # Sell
                        balance += position_size * current_price
                        trades.append({
                            'date': current_date,
                            'type': 'sell',
                            'price': current_price,
                            'size': position_size,
                            'balance': balance,
                            'profit_pct': price_change * 100
                        })
                        position = None
                        position_size = 0
            
            # Close any open positions at the end
            if position is not None:
                final_price = test_df['close'].iloc[-1]
                price_change = (final_price / position) - 1
                balance += position_size * final_price
                trades.append({
                    'date': test_df.index[-1],
                    'type': 'sell',
                    'price': final_price,
                    'size': position_size,
                    'balance': balance,
                    'profit_pct': price_change * 100
                })
            
            # Calculate performance metrics
            total_return_pct = (balance / initial_balance - 1) * 100
            
            win_trades = [t for t in trades if t['type'] == 'sell' and t.get('profit_pct', 0) > 0]
            lose_trades = [t for t in trades if t['type'] == 'sell' and t.get('profit_pct', 0) <= 0]
            
            total_trades = len(win_trades) + len(lose_trades)
            win_rate = len(win_trades) / max(1, total_trades) * 100
            
            # Calculate drawdown
            balance_history = [initial_balance]
            for trade in trades:
                if trade['type'] == 'sell':
                    balance_history.append(trade['balance'])
            
            max_balance = initial_balance
            max_drawdown = 0
            
            for balance in balance_history:
                max_balance = max(max_balance, balance)
                drawdown = (max_balance - balance) / max_balance * 100
                max_drawdown = max(max_drawdown, drawdown)
            
            # Calculate average holding period
            holding_periods = []
            buy_date = None
            
            for trade in trades:
                if trade['type'] == 'buy':
                    buy_date = trade['date']
                elif trade['type'] == 'sell' and buy_date is not None:
                    days_held = (trade['date'] - buy_date).days
                    holding_periods.append(days_held)
                    buy_date = None
            
            avg_holding_period = sum(holding_periods) / max(1, len(holding_periods))
            
            # Calculate profit factor
            total_gains = sum([t.get('profit_pct', 0) for t in win_trades])
            total_losses = abs(sum([t.get('profit_pct', 0) for t in lose_trades]))
            profit_factor = total_gains / max(0.01, total_losses)
            
            # Annualized return (simplified)
            test_days = (test_df.index[-1] - test_df.index[0]).days
            annual_return = total_return_pct * (365 / max(1, test_days))
            
            # Simple Sharpe ratio approximation (assuming risk-free rate of 0%)
            returns = [t.get('profit_pct', 0) for t in trades if t['type'] == 'sell']
            if returns:
                sharpe_ratio = np.mean(returns) / max(0.01, np.std(returns)) * np.sqrt(len(returns))
            else:
                sharpe_ratio = 0
            
            return {
                'total_return_pct': total_return_pct,
                'annual_return_pct': annual_return,
                'max_drawdown_pct': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'profit_factor': profit_factor,
                'avg_holding_period': avg_holding_period,
                'trades': trades,
                'balance_history': balance_history
            }
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return self._empty_results()
    
    def _empty_results(self) -> Dict:
        """Return empty results for error cases"""
        return {
            'total_return_pct': 0.0,
            'annual_return_pct': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'profit_factor': 0.0,
            'avg_holding_period': 0.0,
            'trades': [],
            'balance_history': []
        }


# Utility functions
def predict_prices(df: pd.DataFrame, symbol: str, interval: str, days_ahead: int = 7) -> pd.DataFrame:
    """
    Predict prices for the given symbol and interval
    
    Args:
        df: DataFrame with historical data
        symbol: Symbol to predict
        interval: Time interval
        days_ahead: Number of days to predict
        
    Returns:
        DataFrame with predictions
    """
    model = SimplePredictionModel(symbol, interval)
    return model.predict(df, days_ahead)

def detect_market_regime(df: pd.DataFrame, symbol: str, interval: str) -> Dict:
    """
    Detect the current market regime
    
    Args:
        df: DataFrame with historical data
        symbol: Symbol to analyze
        interval: Time interval
        
    Returns:
        Dictionary with regime information
    """
    regime_model = MarketRegimeModel(symbol, interval)
    return regime_model.detect_regime(df)

def run_strategy_backtest(df: pd.DataFrame, symbol: str, interval: str, params: Dict = None) -> Dict:
    """
    Run a backtest for a trading strategy
    
    Args:
        df: DataFrame with historical data
        symbol: Symbol to backtest
        interval: Time interval
        params: Strategy parameters
        
    Returns:
        Dictionary with backtest results
    """
    backtest = BacktestEngine(symbol, interval)
    return backtest.run_backtest(df, params)
    
    
def train_price_models(symbol: str, interval: str, lookback_days: int = 90) -> bool:
    """
    Train prediction models for a specific symbol and interval
    
    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        interval: Time interval (e.g., 1h, 4h, 1d)
        lookback_days: Days of historical data to use for training
        
    Returns:
        Boolean indicating success
    """
    try:
        from binance_api import get_historical_data
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        df = get_historical_data(symbol, interval, start_date, end_date)
        
        if df is None or len(df) < 30:
            logger.warning(f"Not enough data for {symbol}/{interval} to train models")
            return False
        
        # Train simple prediction model
        model = SimplePredictionModel(symbol, interval)
        success = model.train(df)
        
        # Train market regime model (for future use)
        # regime_model = MarketRegimeModel(symbol, interval)
        
        logger.info(f"Model training for {symbol}/{interval}: {'Successful' if success else 'Failed'}")
        return success
    
    except Exception as e:
        logger.error(f"Error training price models for {symbol}/{interval}: {e}")
        return False


def predict_prices_all(symbols: List[str] = None, intervals: List[str] = None) -> int:
    """
    Generate price predictions for multiple symbols and intervals
    
    Args:
        symbols: List of symbols to generate predictions for
        intervals: List of intervals to generate predictions for
        
    Returns:
        Number of predictions generated
    """
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    
    if intervals is None:
        intervals = ["1h", "4h", "1d"]
    
    from binance_api import get_historical_data
    
    prediction_count = 0
    for symbol in symbols:
        for interval in intervals:
            try:
                # Get recent data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)  # Need sufficient data for feature creation
                
                df = get_historical_data(symbol, interval, start_date, end_date)
                
                if df is None or len(df) < 20:
                    logger.warning(f"Not enough data for {symbol}/{interval} to make predictions")
                    continue
                
                # Load model and predict
                model = SimplePredictionModel(symbol, interval)
                if model.load():
                    # Make 7-day prediction
                    predictions = model.predict(df, days_ahead=7)
                    if predictions is not None and len(predictions) > 0:
                        prediction_count += len(predictions)
                        logger.info(f"Generated {len(predictions)} predictions for {symbol}/{interval}")
                else:
                    logger.warning(f"Could not load model for {symbol}/{interval}")
            
            except Exception as e:
                logger.error(f"Error generating predictions for {symbol}/{interval}: {e}")
    
    return prediction_count