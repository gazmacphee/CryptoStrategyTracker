"""
Advanced Machine Learning module for pattern recognition and trading opportunities.
This module provides automated pattern analysis across different symbols and intervals
to identify buying and selling opportunities.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import joblib
from typing import Dict, List, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import concurrent.futures
import database
import trading_signals

# Import ML database operations and database extensions
try:
    import db_ml_operations
    ML_DB_AVAILABLE = True
except ImportError:
    print("Warning: db_ml_operations module not available. ML data will not be saved to database.")
    ML_DB_AVAILABLE = False

# Import database extensions to ensure all required database functions are available
try:
    import database_extensions
    from database_extensions import get_historical_data
except ImportError:
    print("Warning: database_extensions module not found. Some database functions may be missing.")
    # Create a fallback function (although we shouldn't need it)
    def get_historical_data(symbol, interval, lookback_days=30, start_date=None, end_date=None):
        print(f"Using fallback get_historical_data for {symbol}/{interval}")
        return None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = 'models/pattern_recognition'
os.makedirs(MODEL_DIR, exist_ok=True)

# Pattern types to detect
PATTERN_TYPES = {
    'double_bottom': 'Bullish reversal pattern that forms after a downtrend',
    'double_top': 'Bearish reversal pattern that forms after an uptrend',
    'head_shoulders': 'Bearish reversal pattern with three peaks',
    'inv_head_shoulders': 'Bullish reversal pattern with three troughs',
    'bull_flag': 'Bullish continuation pattern after a strong uptrend',
    'bear_flag': 'Bearish continuation pattern after a strong downtrend',
    'cup_handle': 'Bullish continuation pattern resembling a cup with a handle',
    'triangle': 'Continuation pattern showing converging trendlines',
    'channel': 'Price movement between parallel trendlines',
    'divergence': 'Price and indicator moving in opposite directions',
    'support_bounce': 'Price bounce from a support level',
    'resistance_breakdown': 'Price breaking through resistance',
    'volume_spike': 'Unusual increase in trading volume',
    'momentum_shift': 'Significant change in price momentum'
}

class PatternRecognitionModel:
    """Advanced model for recognizing patterns across symbols and timeframes"""
    
    def __init__(self, rebuild_models=False):
        self.models = {}
        self.scalers = {}
        self.feature_importances = {}
        self.rebuild_models = rebuild_models
        
    def prepare_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features useful for pattern recognition
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            DataFrame with extracted pattern features
        """
        # Ensure the DataFrame is sorted by timestamp
        df = df.sort_values('timestamp').copy()
        
        # Calculate rolling statistics for pattern detection
        df['price_range'] = df['high'] - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['body_to_range'] = df['body_size'] / df['price_range'].replace(0, np.nan)
        
        # Volume-based features
        df['rel_volume'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Calculate price changes over different windows
        for window in [3, 5, 10, 20]:
            df[f'price_change_{window}d'] = df['close'].pct_change(window)
            df[f'volume_change_{window}d'] = df['volume'].pct_change(window)
        
        # Higher-level patterns require lookback windows
        # Detect potential double bottoms
        df['low_min_20d'] = df['low'].rolling(20).min()
        df['potential_double_bottom'] = ((df['low'] - df['low_min_20d']).abs() < df['low'].rolling(20).std() * 0.1) & \
                                       (df['low'] > df['low'].shift(1)) & \
                                       (df['low'] > df['low'].shift(-1))
        
        # Detect potential double tops
        df['high_max_20d'] = df['high'].rolling(20).max()
        df['potential_double_top'] = ((df['high'] - df['high_max_20d']).abs() < df['high'].rolling(20).std() * 0.1) & \
                                     (df['high'] < df['high'].shift(1)) & \
                                     (df['high'] < df['high'].shift(-1))
        
        # Detect potential head and shoulders (simplified)
        df['left_shoulder'] = (df['high'] < df['high'].shift(1)) & (df['high'] > df['high'].shift(2))
        df['head'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        df['right_shoulder'] = (df['high'] < df['high'].shift(-1)) & (df['high'] > df['high'].shift(-2))
        
        # Detect support/resistance levels
        df['support_level'] = df['low'].rolling(20).min()
        df['resistance_level'] = df['high'].rolling(20).max()
        df['near_support'] = (df['close'] - df['support_level']) / df['close'] < 0.01
        df['near_resistance'] = (df['resistance_level'] - df['close']) / df['close'] < 0.01
        
        # Candlestick pattern features
        df['doji'] = df['body_size'] < 0.1 * df['price_range']
        df['hammer'] = (df['lower_shadow'] > 2 * df['body_size']) & (df['upper_shadow'] < 0.5 * df['body_size'])
        df['shooting_star'] = (df['upper_shadow'] > 2 * df['body_size']) & (df['lower_shadow'] < 0.5 * df['body_size'])
        
        # Indicator-based patterns
        if 'rsi' in df.columns:
            df['rsi_oversold'] = df['rsi'] < 30
            df['rsi_overbought'] = df['rsi'] > 70
            df['rsi_bullish_divergence'] = (df['close'] < df['close'].shift(5)) & (df['rsi'] > df['rsi'].shift(5))
            df['rsi_bearish_divergence'] = (df['close'] > df['close'].shift(5)) & (df['rsi'] < df['rsi'].shift(5))
        
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df['macd_crossover_bullish'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
            df['macd_crossover_bearish'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        # Calculate the slope of EMAs if available
        for ema in ['ema_9', 'ema_21', 'ema_50', 'ema_200']:
            if ema in df.columns:
                df[f'{ema}_slope'] = df[ema].pct_change(5)
        
        # Fill NaN values with 0 for boolean columns and median for numeric columns
        bool_columns = df.select_dtypes(include=[bool]).columns
        df[bool_columns] = df[bool_columns].fillna(False)
        
        # For numeric columns, fill with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def extract_labeled_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and label historical patterns based on future price movements
        
        Args:
            df: DataFrame with OHLCV, indicators, and pattern features
            
        Returns:
            DataFrame with labeled patterns
        """
        # Create future return columns to determine if a pattern was profitable
        for lookahead in [1, 3, 5, 10, 20]:
            df[f'future_return_{lookahead}d'] = df['close'].pct_change(periods=-lookahead)
        
        # Lower thresholds for pattern labeling to work with limited historical data
        
        # Label the patterns
        # 1. Double bottom (bullish)
        df['double_bottom'] = df['potential_double_bottom'] & (df['future_return_10d'] > 0.02)
        
        # 2. Double top (bearish)
        df['double_top'] = df['potential_double_top'] & (df['future_return_10d'] < -0.02)
        
        # 3. Bull flag (bullish continuation)
        df['bull_flag'] = (df['close'] > df['close'].rolling(20).mean()) & \
                           (df['close'].pct_change(10) > 0.03) & \
                           (df['close'].pct_change(5) < 0.02) & \
                           (df['future_return_10d'] > 0.02)
        
        # 4. Bear flag (bearish continuation)
        df['bear_flag'] = (df['close'] < df['close'].rolling(20).mean()) & \
                           (df['close'].pct_change(10) < -0.03) & \
                           (df['close'].pct_change(5) > -0.02) & \
                           (df['future_return_10d'] < -0.02)
        
        # 5. Support bounce (bullish)
        df['support_bounce'] = df['near_support'] & (df['close'] > df['open']) & \
                               (df['future_return_5d'] > 0.015)
        
        # 6. Resistance breakdown (bearish)
        df['resistance_breakdown'] = df['near_resistance'] & (df['close'] < df['open']) & \
                                     (df['future_return_5d'] < -0.015)
        
        # 7. Volume-supported reversal (can be bullish or bearish)
        df['volume_reversal'] = (df['rel_volume'] > 1.3) & \
                                (df['close'].pct_change() * df['close'].pct_change(1) < 0) & \
                                (abs(df['future_return_5d']) > 0.02)
        
        # 8. Momentum shift (direction depends on the shift)
        if 'rsi' in df.columns:
            df['momentum_shift_bullish'] = (df['rsi'].shift(5) < 35) & (df['rsi'] > 40) & \
                                           (df['future_return_10d'] > 0.02)
            
            df['momentum_shift_bearish'] = (df['rsi'].shift(5) > 65) & (df['rsi'] < 60) & \
                                           (df['future_return_10d'] < -0.02)
        
        # Combine all patterns and create a single target column
        pattern_columns = [
            'double_bottom', 'double_top', 'bull_flag', 'bear_flag',
            'support_bounce', 'resistance_breakdown', 'volume_reversal'
        ]
        
        if 'rsi' in df.columns:
            pattern_columns.extend(['momentum_shift_bullish', 'momentum_shift_bearish'])
        
        # Create target columns: 'pattern_type' and 'pattern_direction'
        df['has_pattern'] = df[pattern_columns].any(axis=1)
        df['pattern_direction'] = 'neutral'
        
        # Bullish patterns
        bullish_patterns = ['double_bottom', 'bull_flag', 'support_bounce']
        if 'rsi' in df.columns:
            bullish_patterns.append('momentum_shift_bullish')
        
        # Bearish patterns
        bearish_patterns = ['double_top', 'bear_flag', 'resistance_breakdown']
        if 'rsi' in df.columns:
            bearish_patterns.append('momentum_shift_bearish')
        
        # Set pattern direction
        df.loc[df[bullish_patterns].any(axis=1), 'pattern_direction'] = 'bullish'
        df.loc[df[bearish_patterns].any(axis=1), 'pattern_direction'] = 'bearish'
        
        # Identify the pattern type (taking the first one found for simplicity)
        df['pattern_type'] = 'none'
        for pattern in pattern_columns:
            df.loc[df[pattern] & (df['pattern_type'] == 'none'), 'pattern_type'] = pattern
        
        # Calculate pattern profitability based on future returns
        df['pattern_profit_5d'] = df['future_return_5d'] * (df['pattern_direction'] == 'bullish') - \
                                  df['future_return_5d'] * (df['pattern_direction'] == 'bearish')
        
        df['pattern_profit_10d'] = df['future_return_10d'] * (df['pattern_direction'] == 'bullish') - \
                                   df['future_return_10d'] * (df['pattern_direction'] == 'bearish')
        
        # Pattern success or failure
        df['pattern_success'] = ((df['pattern_direction'] == 'bullish') & (df['future_return_10d'] > 0)) | \
                               ((df['pattern_direction'] == 'bearish') & (df['future_return_10d'] < 0))
        
        return df
    
    def train_pattern_model(self, df: pd.DataFrame, symbol: str, interval: str) -> bool:
        """
        Train a model to recognize profitable patterns
        
        Args:
            df: DataFrame with OHLCV data and indicators
            symbol: Trading pair symbol
            interval: Time interval
            
        Returns:
            Boolean indicating if training was successful
        """
        try:
            if len(df) < 500:
                logger.warning(f"Not enough data for {symbol}/{interval} to train pattern model")
                return False
            
            # Process features for pattern recognition
            pattern_df = self.prepare_pattern_features(df)
            
            # Extract and label patterns
            labeled_df = self.extract_labeled_patterns(pattern_df)
            
            # Keep only rows with actual patterns for training
            pattern_data = labeled_df[labeled_df['has_pattern']]
            
            if len(pattern_data) < 10:
                logger.warning(f"Not enough pattern instances for {symbol}/{interval} to train pattern model")
                return False
                
            # Log pattern distribution
            logger.info(f"Found {len(pattern_data)} pattern instances for {symbol}/{interval}")
            pattern_types = pattern_data['pattern_type'].value_counts()
            logger.info(f"Pattern distribution: {pattern_types.to_dict()}")
            
            logger.info(f"Training pattern model for {symbol}/{interval} with {len(pattern_data)} patterns")
            
            # Select features for pattern recognition
            feature_columns = [
                'price_range', 'body_size', 'upper_shadow', 'lower_shadow',
                'body_to_range', 'rel_volume',
                'price_change_3d', 'price_change_5d', 'price_change_10d', 'price_change_20d',
                'volume_change_3d', 'volume_change_5d', 'volume_change_10d', 'volume_change_20d',
                'near_support', 'near_resistance',
                'doji', 'hammer', 'shooting_star'
            ]
            
            # Add indicator features if available
            if 'rsi' in pattern_data.columns:
                feature_columns.extend(['rsi', 'rsi_oversold', 'rsi_overbought', 
                                       'rsi_bullish_divergence', 'rsi_bearish_divergence'])
                
            if 'macd' in pattern_data.columns and 'macd_signal' in pattern_data.columns:
                feature_columns.extend(['macd', 'macd_signal', 'macd_histogram',
                                       'macd_crossover_bullish', 'macd_crossover_bearish'])
                
            # Add EMA features if available
            for ema in ['ema_9', 'ema_21', 'ema_50', 'ema_200']:
                if ema in pattern_data.columns:
                    feature_columns.append(ema)
                    feature_columns.append(f'{ema}_slope')
            
            # Filter out any columns that don't exist in the dataframe
            feature_columns = [col for col in feature_columns if col in pattern_data.columns]
            
            # Prepare features and target for pattern direction classification
            X = pattern_data[feature_columns].values
            y_direction = (pattern_data['pattern_direction'] == 'bullish').astype(int)
            
            # Split data for training and validation
            X_train, X_val, y_train, y_val = train_test_split(X, y_direction, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model for pattern direction
            model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=15,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_val_scaled)
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            logger.info(f"Pattern model for {symbol}/{interval} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
            
            # Save model performance to database if available
            if ML_DB_AVAILABLE:
                try:
                    # Calculate regression metrics (simplified - we're classifying but saving standard metrics)
                    y_prob = model.predict_proba(X_val_scaled)[:, 1]  # Probability of positive class
                    mse = ((y_val - y_prob)**2).mean()
                    mae = abs(y_val - y_prob).mean()
                    
                    # Save to database
                    db_ml_operations.save_ml_model_performance(
                        model_name=f"pattern_recognition_{symbol}_{interval}",
                        symbol=symbol,
                        interval=interval,
                        training_timestamp=datetime.now(),
                        accuracy=float(accuracy),
                        precision_score=float(precision),
                        recall=float(recall),
                        f1_score=float(f1),
                        mse=float(mse),
                        mae=float(mae),
                        training_params={
                            'features': feature_columns,
                            'model_type': 'RandomForestClassifier',
                            'training_rows': len(X_train)
                        }
                    )
                    logger.info(f"Saved model performance metrics for {symbol}/{interval} pattern recognition model")
                except Exception as e:
                    logger.error(f"Error saving model performance metrics: {e}")
            
            # Save feature importances
            feature_importances = dict(zip(feature_columns, model.feature_importances_))
            
            # Save model and scaler
            model_key = f"{symbol}_{interval}"
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            self.feature_importances[model_key] = feature_importances
            
            model_path = os.path.join(MODEL_DIR, f'pattern_model_{model_key}.joblib')
            joblib.dump({
                'model': model,
                'scaler': scaler,
                'feature_columns': feature_columns,
                'feature_importances': feature_importances,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall
                }
            }, model_path)
            
            logger.info(f"Pattern model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error training pattern model for {symbol}/{interval}: {e}")
            return False
    
    def load_pattern_model(self, symbol: str, interval: str) -> bool:
        """
        Load a previously trained pattern model
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            
        Returns:
            Boolean indicating if loading was successful
        """
        try:
            model_key = f"{symbol}_{interval}"
            model_path = os.path.join(MODEL_DIR, f'pattern_model_{model_key}.joblib')
            
            if not os.path.exists(model_path):
                logger.warning(f"No saved pattern model found for {symbol}/{interval}")
                return False
            
            saved_data = joblib.load(model_path)
            self.models[model_key] = saved_data['model']
            self.scalers[model_key] = saved_data['scaler']
            self.feature_importances[model_key] = saved_data['feature_importances']
            
            logger.info(f"Pattern model loaded for {symbol}/{interval}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading pattern model for {symbol}/{interval}: {e}")
            return False
    
    def detect_patterns(self, df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        """
        Detect patterns in the given data
        
        Args:
            df: DataFrame with OHLCV data and indicators
            symbol: Trading pair symbol
            interval: Time interval
            
        Returns:
            DataFrame with detected patterns
        """
        try:
            model_key = f"{symbol}_{interval}"
            
            # Try to load the model if it's not already loaded
            if model_key not in self.models:
                loaded = self.load_pattern_model(symbol, interval)
                
                # If loading fails and rebuild_models is True, train a new model
                if not loaded and self.rebuild_models:
                    logger.info(f"Training new pattern model for {symbol}/{interval}")
                    self.train_pattern_model(df, symbol, interval)
                # If still not available, return empty results
                if model_key not in self.models:
                    logger.warning(f"No pattern model available for {symbol}/{interval}")
                    return pd.DataFrame()
            
            # Process features for pattern recognition
            pattern_df = self.prepare_pattern_features(df)
            
            # Get the model and scaler
            model = self.models[model_key]
            scaler = self.scalers[model_key]
            
            # Get feature columns from feature importances
            feature_columns = list(self.feature_importances[model_key].keys())
            
            # Filter out any columns that don't exist in the dataframe
            available_features = [col for col in feature_columns if col in pattern_df.columns]
            
            # Use only the most recent 100 candles for pattern detection
            recent_df = pattern_df.iloc[-100:].copy()
            
            # Process predictions
            X = recent_df[available_features].values
            X_scaled = scaler.transform(X)
            
            # Predict pattern direction (1 for bullish, 0 for bearish)
            direction_pred = model.predict(X_scaled)
            direction_prob = model.predict_proba(X_scaled)
            
            # Add predictions to dataframe
            recent_df['predicted_direction'] = ['bullish' if p == 1 else 'bearish' for p in direction_pred]
            recent_df['prediction_confidence'] = np.max(direction_prob, axis=1)
            
            # Determine detected patterns based on feature values
            # We'll use simple rules to categorize patterns
            pattern_types = []
            pattern_strengths = []
            expected_returns = []
            
            for idx, row in recent_df.iterrows():
                # Determine pattern type based on feature values
                pattern_type = 'unknown'
                pattern_strength = row['prediction_confidence']
                expected_return = 0.0
                
                # Use a confidence threshold to filter out weak signals
                if pattern_strength < 0.6:
                    pattern_type = 'none'
                    expected_return = 0.0
                else:
                    # Bullish patterns
                    if row['predicted_direction'] == 'bullish':
                        if row.get('potential_double_bottom', False):
                            pattern_type = 'double_bottom'
                            expected_return = 0.03
                        elif row.get('near_support', False) and row['rel_volume'] > 1.2:
                            pattern_type = 'support_bounce'
                            expected_return = 0.02
                        elif row.get('rsi_oversold', False) and row.get('macd_crossover_bullish', False):
                            pattern_type = 'oversold_reversal'
                            expected_return = 0.025
                        elif row.get('hammer', False) and row['price_change_5d'] < -0.03:
                            pattern_type = 'hammer_reversal'
                            expected_return = 0.02
                        else:
                            pattern_type = 'bullish_signal'
                            expected_return = 0.015
                    
                    # Bearish patterns
                    else:
                        if row.get('potential_double_top', False):
                            pattern_type = 'double_top'
                            expected_return = 0.03
                        elif row.get('near_resistance', False) and row['rel_volume'] > 1.2:
                            pattern_type = 'resistance_breakdown'
                            expected_return = 0.02
                        elif row.get('rsi_overbought', False) and row.get('macd_crossover_bearish', False):
                            pattern_type = 'overbought_reversal'
                            expected_return = 0.025
                        elif row.get('shooting_star', False) and row['price_change_5d'] > 0.03:
                            pattern_type = 'shooting_star_reversal'
                            expected_return = 0.02
                        else:
                            pattern_type = 'bearish_signal'
                            expected_return = 0.015
                
                pattern_types.append(pattern_type)
                pattern_strengths.append(pattern_strength)
                expected_returns.append(expected_return if pattern_type != 'none' else 0.0)
            
            recent_df['pattern_type'] = pattern_types
            recent_df['pattern_strength'] = pattern_strengths
            recent_df['expected_return'] = expected_returns
            
            # Filter out rows without detected patterns
            patterns = recent_df[recent_df['pattern_type'] != 'none'].copy()
            
            # Add symbol and interval information
            patterns['symbol'] = symbol
            patterns['interval'] = interval
            
            # Calculate additional information for alerts
            now = datetime.now()
            patterns['detection_time'] = now
            patterns['days_since_signal'] = (now - patterns['timestamp']).dt.total_seconds() / (24 * 3600)
            
            # Filter to recent patterns (< 3 days old) with sufficient confidence
            recent_patterns = patterns[(patterns['days_since_signal'] < 3) & 
                                       (patterns['pattern_strength'] > 0.65)].copy()
            
            # Save detected patterns to database if available
            if ML_DB_AVAILABLE and len(recent_patterns) > 0:
                try:
                    for _, row in recent_patterns.iterrows():
                        # Determine expected outcome from pattern type
                        pattern_type = row['pattern_type']
                        expected_outcome = f"{'Increase' if row['predicted_direction'] == 'bullish' else 'Decrease'} of approximately {row['expected_return'] * 100:.1f}%"
                        
                        # Create description
                        if pattern_type in PATTERN_TYPES:
                            description = PATTERN_TYPES[pattern_type]
                        else:
                            description = f"{pattern_type.replace('_', ' ').title()} pattern detected with {row['pattern_strength']:.2f} strength. "
                            description += f"Expected price change: {row['expected_return'] * 100:.1f}%"
                        
                        # Save to database
                        db_ml_operations.save_detected_pattern(
                            symbol=symbol,
                            interval=interval,
                            timestamp=row['timestamp'],
                            pattern_type=pattern_type,
                            pattern_strength=float(row['pattern_strength']),
                            expected_outcome=expected_outcome,
                            confidence_score=float(row['prediction_confidence']),
                            description=description,
                            detection_timestamp=datetime.now()
                        )
                    logger.info(f"Saved {len(recent_patterns)} detected patterns to database for {symbol}/{interval}")
                except Exception as e:
                    logger.error(f"Error saving patterns to database: {e}")
            
            return recent_patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns for {symbol}/{interval}: {e}")
            return pd.DataFrame()


class MultiSymbolPatternAnalyzer:
    """Analyzes patterns across multiple symbols and timeframes"""
    
    def __init__(self, max_symbols=10, max_intervals=3):
        self.max_symbols = max_symbols
        self.max_intervals = max_intervals
        self.pattern_model = PatternRecognitionModel()
        
    def analyze_patterns_in_data(self, data_dict):
        """
        Analyze patterns in prepared data provided as a dictionary
        
        Args:
            data_dict: Dictionary mapping (symbol, interval) tuples to dataframes with OHLCV and indicators
            
        Returns:
            DataFrame with detected patterns across all symbols/intervals
        """
        import pandas as pd
        
        if not data_dict:
            logging.warning("No data provided for pattern analysis")
            return pd.DataFrame()
            
        all_patterns = []
        
        for (symbol, interval), df in data_dict.items():
            if df.empty:
                continue
                
            try:
                # Get patterns for this symbol/interval
                patterns = self.pattern_model.detect_patterns(df, symbol, interval)
                
                if not patterns.empty:
                    patterns['symbol'] = symbol
                    patterns['interval'] = interval
                    patterns['detected_at'] = pd.Timestamp.now()
                    all_patterns.append(patterns)
            except Exception as e:
                logging.error(f"Error analyzing patterns for {symbol}/{interval}: {e}")
                
        if all_patterns:
            # Combine all pattern dataframes
            combined_patterns = pd.concat(all_patterns, ignore_index=True)
            return combined_patterns
        else:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['symbol', 'interval', 'timestamp', 'pattern_type', 
                                          'strength', 'expected_direction', 'detected_at'])
        
    def get_symbols_data(self, symbols=None, intervals=None, days=30):
        """
        Get data for multiple symbols and intervals
        
        Args:
            symbols: List of symbols to analyze (or None to use popular symbols)
            intervals: List of intervals to analyze
            days: Number of days to look back
            
        Returns:
            Dictionary mapping (symbol, interval) to dataframes
        """
        try:
            # Use database_extensions to ensure needed functions
            import database_extensions
            
            # Get data access functions from database module
            from database import get_historical_data, get_available_symbols
            
            # Get popular symbols if not specified
            if symbols is None or len(symbols) == 0:
                # Use the get_available_symbols function
                symbols = get_available_symbols(limit=self.max_symbols)
                print(f"Using {len(symbols)} symbols from get_available_symbols")
            
            # Default intervals if not specified
            if intervals is None or len(intervals) == 0:
                intervals = ['1h', '4h', '1d']
            
            # Limit the number of symbols and intervals
            symbols = symbols[:self.max_symbols]
            intervals = intervals[:self.max_intervals]
            
            # Use multithreading to fetch data in parallel
            data = {}
            
            def fetch_symbol_interval(symbol, interval):
                # Directly import the database modules here to ensure they're available
                import database
                from database_extensions import get_historical_data, ensure_float_df, safe_float_convert
                
                print(f"Fetching data for {symbol}/{interval} with lookback of {days} days")
                
                # Use get_historical_data function from database_extensions module
                df = get_historical_data(symbol, interval, lookback_days=days)
                
                if df is not None and not df.empty:
                    # Log the data retrieval success
                    print(f"Successfully retrieved {len(df)} rows for {symbol}/{interval}")
                    
                    # Add any required post-processing here
                    import pandas as pd
                    
                    # Convert any Decimal types to float for numerical operations
                    df = ensure_float_df(df, columns=['open', 'high', 'low', 'close', 'volume'])
                    
                    # Ensure timestamp is in datetime format
                    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Make sure we have all required columns
                    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    for col in required_columns:
                        if col not in df.columns:
                            print(f"Warning: Required column '{col}' missing from data")
                            return (symbol, interval), None
                    
                    # Check for NaN values and replace them
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if df[col].isna().any():
                            print(f"Warning: Found NaN values in {col} column, filling with appropriate values")
                            if col == 'volume':
                                df[col] = df[col].fillna(0)
                            else:
                                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                    
                    # Sort by timestamp to ensure chronological order
                    df = df.sort_values('timestamp')
                    
                    print(f"Data for {symbol}/{interval} is ready for processing with {len(df)} rows")
                    return (symbol, interval), df
                else:
                    print(f"No data retrieved for {symbol}/{interval}")
                    # Let's try with special handling for 1h interval
                    if interval == '1h':
                        print(f"Attempting special handling for 1h data via 30m data conversion")
                        try:
                            # Try to convert 30m data to 1h if available
                            df_30m = get_historical_data(symbol, '30m', lookback_days=days)
                            if df_30m is not None and not df_30m.empty:
                                # We found 30m data, let's convert it to 1h
                                print(f"Found {len(df_30m)} records of 30m data, converting to 1h")
                                
                                # Convert decimal values
                                df_30m = ensure_float_df(df_30m)
                                
                                # Convert timestamp to datetime
                                if 'timestamp' in df_30m.columns and not pd.api.types.is_datetime64_any_dtype(df_30m['timestamp']):
                                    df_30m['timestamp'] = pd.to_datetime(df_30m['timestamp'])
                                
                                # Set timestamp as index for resampling
                                df_30m = df_30m.set_index('timestamp')
                                
                                # Resample to 1h
                                df_1h = df_30m.resample('1H').agg({
                                    'open': 'first',
                                    'high': 'max',
                                    'low': 'min',
                                    'close': 'last',
                                    'volume': 'sum'
                                })
                                
                                # Reset index to get timestamp as column again
                                df_1h = df_1h.reset_index()
                                
                                # Set the interval to 1h
                                if 'interval' in df_30m.columns:
                                    df_1h['interval'] = '1h'
                                    
                                print(f"Successfully converted 30m data to 1h: {len(df_1h)} rows")
                                return (symbol, interval), df_1h
                        except Exception as e:
                            print(f"Error in special handling for 1h data: {e}")
                
                return (symbol, interval), None
            
            # Use ThreadPoolExecutor to parallelize data fetching
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for symbol in symbols:
                    for interval in intervals:
                        futures.append(executor.submit(fetch_symbol_interval, symbol, interval))
                
                for future in concurrent.futures.as_completed(futures):
                    key, df = future.result()
                    if df is not None:
                        data[key] = df
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching multi-symbol data: {e}")
            return {}
    
    def train_pattern_models(self, symbols=None, intervals=None, days=90):
        """
        Train pattern models for multiple symbols and intervals
        
        Args:
            symbols: List of symbols to train models for
            intervals: List of intervals to train models for
            days: Number of days of data to use for training
            
        Returns:
            Dictionary with training results
        """
        # Get data for symbols and intervals
        data = self.get_symbols_data(symbols, intervals, days)
        
        if not data:
            logger.warning("No data available for training pattern models")
            # Return a properly structured empty result
            return {
                'total': 0,
                'successful': 0,
                'details': {}
            }
        
        # Train models for each symbol and interval
        results = {}
        for (symbol, interval), df in data.items():
            success = self.pattern_model.train_pattern_model(df, symbol, interval)
            results[(symbol, interval)] = success
        
        # Return success rates
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Successfully trained {success_count}/{len(results)} pattern models")
        
        return {
            'total': len(results),
            'successful': success_count,
            'details': results
        }
    
    def analyze_all_patterns(self, symbols=None, intervals=None, days=30):
        """
        Analyze patterns across all specified symbols and intervals
        
        Args:
            symbols: List of symbols to analyze
            intervals: List of intervals to analyze
            days: Number of days of data to analyze
            
        Returns:
            DataFrame with detected patterns and recommendations
        """
        # Get data for symbols and intervals
        data = self.get_symbols_data(symbols, intervals, days)
        
        if not data:
            logger.warning("No data available for pattern analysis")
            return pd.DataFrame()
        
        # Detect patterns for each symbol and interval
        all_patterns = []
        
        for (symbol, interval), df in data.items():
            patterns = self.pattern_model.detect_patterns(df, symbol, interval)
            if not patterns.empty:
                all_patterns.append(patterns)
        
        if not all_patterns:
            logger.info("No patterns detected across any symbols or intervals")
            return pd.DataFrame()
        
        # Combine all patterns into a single DataFrame
        combined = pd.concat(all_patterns, ignore_index=True)
        
        # Calculate a composite score for better sorting/filtering
        combined['composite_score'] = combined['pattern_strength'] * combined['expected_return'] * \
                                     (1.0 / (combined['days_since_signal'] + 0.1))
        
        # Sort by composite score
        sorted_patterns = combined.sort_values('composite_score', ascending=False)
        
        # Add action recommendations
        sorted_patterns['recommended_action'] = sorted_patterns.apply(
            lambda row: f"{'BUY' if row['predicted_direction'] == 'bullish' else 'SELL'} {row['symbol']} on {row['interval']} timeframe",
            axis=1
        )
        
        # Format the results for display
        display_columns = [
            'symbol', 'interval', 'timestamp', 'close', 
            'pattern_type', 'predicted_direction', 'pattern_strength',
            'expected_return', 'days_since_signal', 'recommended_action'
        ]
        
        return sorted_patterns[display_columns]
    
    def get_pattern_recommendations(self, min_strength=0.7, max_days_old=2, limit=10):
        """
        Get high-confidence pattern recommendations for trading
        
        Args:
            min_strength: Minimum pattern strength to include (0.0-1.0)
            max_days_old: Maximum age of patterns in days
            limit: Maximum number of recommendations to return
            
        Returns:
            DataFrame with top pattern recommendations
        """
        # Analyze patterns across popular symbols and timeframes
        all_patterns = self.analyze_all_patterns()
        
        if all_patterns.empty:
            return pd.DataFrame()
        
        # Filter by strength and recency
        strong_recent = all_patterns[
            (all_patterns['pattern_strength'] >= min_strength) & 
            (all_patterns['days_since_signal'] <= max_days_old)
        ].copy()
        
        if strong_recent.empty:
            return pd.DataFrame()
        
        # Sort by composite score and take top N
        recommendations = strong_recent.sort_values('composite_score', ascending=False).head(limit)
        
        # Format for display
        recommendations['timestamp'] = recommendations['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        recommendations['pattern_strength'] = recommendations['pattern_strength'].apply(lambda x: f"{x:.2f}")
        recommendations['expected_return'] = recommendations['expected_return'].apply(lambda x: f"{x*100:.1f}%")
        recommendations['days_since_signal'] = recommendations['days_since_signal'].apply(lambda x: f"{x:.1f}")
        
        display_columns = [
            'symbol', 'interval', 'timestamp', 'close', 
            'pattern_type', 'predicted_direction', 'pattern_strength',
            'expected_return', 'recommended_action'
        ]
        
        return recommendations[display_columns]
    
    def save_trading_opportunity(self, pattern_row):
        """
        Save a detected pattern as a trading signal
        
        Args:
            pattern_row: Series/row from pattern DataFrame
            
        Returns:
            Boolean indicating success
        """
        try:
            # Convert pattern to trading signal format
            signal_type = 'buy' if pattern_row['predicted_direction'] == 'bullish' else 'sell'
            
            # Create a minimal DataFrame with the required columns for trading signals
            signal_df = pd.DataFrame({
                'timestamp': [pattern_row['timestamp']],
                'close': [pattern_row['close']],
                'buy_signal': [signal_type == 'buy'],
                'sell_signal': [signal_type == 'sell']
            })
            
            # Add indicator signals if available
            for indicator in ['bb_signal', 'rsi_signal', 'macd_signal', 'ema_signal']:
                if indicator in pattern_row:
                    signal_df[indicator] = pattern_row[indicator]
            
            # Strategy parameters
            strategy_params = {
                'pattern_type': pattern_row['pattern_type'],
                'pattern_strength': float(pattern_row['pattern_strength']),
                'expected_return': float(pattern_row['expected_return'].replace('%', '')) / 100 
                if isinstance(pattern_row['expected_return'], str) 
                else pattern_row['expected_return']
            }
            
            # Save to database as trading signal
            signal_saved = trading_signals.save_trading_signals(
                signal_df, 
                pattern_row['symbol'], 
                pattern_row['interval'],
                strategy_name="pattern_recognition",
                strategy_params=strategy_params
            )
            
            # Also save as detected pattern in the ML database if available
            try:
                # Check if ML database operations are available
                if ML_DB_AVAILABLE:
                    # Format expected outcome description
                    direction = pattern_row['predicted_direction']
                    expected_return = pattern_row['expected_return']
                    if isinstance(expected_return, str):
                        try:
                            expected_return = float(expected_return.replace('%', '')) / 100
                        except:
                            expected_return = 0.0
                    
                    expected_outcome = f"{'Increase' if direction == 'bullish' else 'Decrease'} of approximately {expected_return*100:.2f}%"
                    
                    # Create a description based on pattern type
                    pattern_type = pattern_row['pattern_type']
                    if pattern_type in PATTERN_TYPES:
                        description = PATTERN_TYPES[pattern_type]
                    else:
                        description = f"{direction.capitalize()} pattern detected"
                    
                    # Save to the detected_patterns table
                    db_ml_operations.save_detected_pattern(
                        symbol=pattern_row['symbol'],
                        interval=pattern_row['interval'],
                        timestamp=pattern_row['timestamp'],
                        pattern_type=pattern_type,
                        pattern_strength=float(pattern_row['pattern_strength']),
                        expected_outcome=expected_outcome,
                        confidence_score=float(pattern_row['pattern_strength']),
                        description=description,
                        detection_timestamp=datetime.now()
                    )
                    logger.info(f"Saved pattern {pattern_type} for {pattern_row['symbol']}/{pattern_row['interval']} to ML database")
            except Exception as e:
                logger.error(f"Error saving detected pattern to ML database: {e}")
            
            return signal_saved
            
        except Exception as e:
            logger.error(f"Error saving trading opportunity: {e}")
            return False
    
    def save_all_recommendations(self, min_strength=0.75):
        """
        Save all high-confidence recommendations as trading signals
        
        Args:
            min_strength: Minimum pattern strength to save
            
        Returns:
            Number of signals saved
        """
        # Get current recommendations
        recommendations = self.get_pattern_recommendations(min_strength=min_strength)
        
        if recommendations.empty:
            return 0
        
        # Save each recommendation
        saved_count = 0
        for _, row in recommendations.iterrows():
            success = self.save_trading_opportunity(row)
            if success:
                saved_count += 1
        
        logger.info(f"Saved {saved_count} pattern-based trading signals")
        return saved_count


# Utility functions
def train_all_pattern_models():
    """Train pattern models for popular symbols and intervals"""
    logger.info("Starting pattern model training")
    analyzer = MultiSymbolPatternAnalyzer()
    
    try:
        logger.info("Calling train_pattern_models()")
        results = analyzer.train_pattern_models()
        logger.info(f"Got result type: {type(results)}")
        
        if isinstance(results, dict):
            logger.info(f"Result keys: {list(results.keys())}")
        else:
            logger.info(f"Result is not a dictionary: {results}")
        
        # Ensure results is properly structured
        if not isinstance(results, dict):
            logger.warning("Invalid train_pattern_models result format")
            logger.info("Returning fallback empty dictionary with proper structure")
            return {
                'total': 0,
                'successful': 0,
                'details': {}
            }
        
        # Ensure result has the expected keys
        if 'total' not in results or 'successful' not in results:
            logger.warning("Missing keys in train_pattern_models result")
            # Try to reconstruct the dictionary if details is available
            if 'details' in results and isinstance(results['details'], dict):
                successful_models = sum(1 for v in results['details'].values() if v)
                results['total'] = len(results['details'])
                results['successful'] = successful_models
                logger.info(f"Reconstructed results from details: total={results['total']}, successful={results['successful']}")
            else:
                logger.info("Could not reconstruct from details, using fallback structure")
                results = {
                    'total': 0,
                    'successful': 0,
                    'details': results.get('details', {}) if isinstance(results, dict) else {}
                }
        
        logger.info(f"Returning training results: {results['successful']}/{results['total']} models trained")
        return results
        
    except Exception as e:
        logger.error(f"Error during pattern model training: {str(e)}")
        logger.exception("Exception details:")
        return {
            'total': 0,
            'successful': 0,
            'details': {},
            'error': str(e)
        }

def get_pattern_recommendations(min_strength=0.7, max_days_old=2, limit=10):
    """Get current pattern recommendations for trading"""
    analyzer = MultiSymbolPatternAnalyzer()
    return analyzer.get_pattern_recommendations(min_strength, max_days_old, limit)

def analyze_all_market_patterns():
    """Analyze patterns across all popular symbols and timeframes"""
    analyzer = MultiSymbolPatternAnalyzer()
    return analyzer.analyze_all_patterns()

def save_current_recommendations():
    """Save current recommendations as trading signals"""
    analyzer = MultiSymbolPatternAnalyzer()
    return analyzer.save_all_recommendations()

def main():
    """Main entry point for command line execution"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Advanced ML Pattern Recognition for Cryptocurrency Trading')
    parser.add_argument('--train', action='store_true', help='Train pattern models for popular symbols')
    parser.add_argument('--analyze', action='store_true', help='Analyze current market patterns')
    parser.add_argument('--save', action='store_true', help='Save detected patterns as trading signals')
    parser.add_argument('--verbose', action='store_true', help='Display detailed information')
    
    args = parser.parse_args()
    
    # Default behavior - run everything if no specific args provided
    run_all = not (args.train or args.analyze or args.save)
    
    # Train pattern models if requested
    if args.train or run_all:
        print("Training pattern models...")
        train_results = train_all_pattern_models()
        
        # Get keys safely with defaults as a precaution
        total = train_results.get('total', 0)
        successful = train_results.get('successful', 0)
        print(f"Training completed: {successful}/{total} models trained")
        
        if args.verbose:
            print(f"Full results: {train_results}")
    
    # Analyze market patterns if requested
    if args.analyze or run_all:
        print("\nAnalyzing current market patterns...")
        recommendations = get_pattern_recommendations()
        
        if not isinstance(recommendations, pd.DataFrame):
            print("Error: recommendations is not a DataFrame")
            if args.verbose:
                print(f"Type: {type(recommendations)}")
                print(f"Value: {recommendations}")
        elif recommendations.empty:
            print("No trading opportunities detected at this time")
        else:
            print(f"\nFound {len(recommendations)} trading opportunities:")
            display_cols = ['symbol', 'interval', 'pattern_type', 'predicted_direction', 'pattern_strength', 'expected_return']
            # Only select columns that exist in the DataFrame
            available_cols = [col for col in display_cols if col in recommendations.columns]
            print(recommendations[available_cols])
    
    # Save recommendations as trading signals if requested
    if args.save or run_all:
        print("\nSaving detected patterns as trading signals...")
        saved_count = save_current_recommendations()
        print(f"Saved {saved_count} patterns as trading signals")


if __name__ == "__main__":
    main()