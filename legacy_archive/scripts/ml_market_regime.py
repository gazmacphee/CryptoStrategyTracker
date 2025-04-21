"""
Market Regime Detection Module

This module provides functionality to:
1. Detect different market regimes (trending, ranging, volatile)
2. Adjust prediction weights based on current market regime
3. Provide context-aware forecasts

Market regimes are detected based on technical indicators such as:
- Trend: ADX, Moving averages
- Range: Bollinger Band Width, ATR
- Volatility: Recent price volatility, RSI extremes

This helps tailor predictions to current market conditions.
"""

import os
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Local imports
from database import get_historical_data, get_indicators
from utils import timeframe_to_seconds

class MarketRegimeDetector:
    """
    Market Regime Detector for cryptocurrency markets
    Identifies current market conditions to help adjust prediction models
    """
    
    def __init__(self, symbol, interval, lookback_periods=100):
        """
        Initialize the market regime detector
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Time interval (e.g., '1h', '4h', '1d')
            lookback_periods: How many periods to analyze for regime detection
        """
        self.symbol = symbol
        self.interval = interval
        self.lookback_periods = lookback_periods
        
        # Define regime types
        self.REGIME_TYPES = {
            0: "RANGING",
            1: "TRENDING_UP",
            2: "TRENDING_DOWN",
            3: "VOLATILE"
        }
        
        # Initialize model
        self.kmeans_model = None
        self.scaler = None
        
        # Model paths
        self.model_path = os.path.join('models', f"{symbol}_{interval}_regime_model.joblib")
        self.scaler_path = os.path.join('models', f"{symbol}_{interval}_regime_scaler.joblib")
        self.metadata_path = os.path.join('models', f"{symbol}_{interval}_regime_metadata.json")
        
        # Ensure model directory exists
        os.makedirs('models', exist_ok=True)
        
        # Load model if exists
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.load_model()
    
    def prepare_regime_features(self, df):
        """
        Calculate features used for regime detection
        
        Args:
            df: DataFrame with price and indicator data
            
        Returns:
            DataFrame with regime detection features
        """
        if df.empty:
            logging.warning(f"Empty dataframe provided for {self.symbol}/{self.interval}")
            return pd.DataFrame()
            
        # Ensure dataframe is sorted by timestamp
        df = df.sort_values('timestamp')
        
        # Create a copy to avoid modifying the original
        regime_df = df.copy()
        
        # 1. Trend Features
        # Price momentum: % change over multiple periods
        for period in [5, 10, 20]:
            regime_df[f'price_change_{period}'] = regime_df['close'].pct_change(periods=period)
        
        # Simple moving averages and their relationships
        for window in [20, 50, 100]:
            regime_df[f'sma_{window}'] = regime_df['close'].rolling(window=window).mean()
        
        # SMA relationships (indicates trend direction and strength)
        regime_df['sma_ratio_20_50'] = regime_df['sma_20'] / regime_df['sma_50']
        regime_df['sma_ratio_20_100'] = regime_df['sma_20'] / regime_df['sma_100']
        
        # 2. Range Features
        # Bollinger Band width (normalized by price)
        if 'bb_upper' in regime_df.columns and 'bb_lower' in regime_df.columns:
            regime_df['bb_width'] = (regime_df['bb_upper'] - regime_df['bb_lower']) / regime_df['close']
        else:
            # Calculate Bollinger Bands if not already present
            window = 20
            std_multiplier = 2
            regime_df['sma_bb'] = regime_df['close'].rolling(window=window).mean()
            regime_df['bb_std'] = regime_df['close'].rolling(window=window).std()
            regime_df['bb_upper'] = regime_df['sma_bb'] + (std_multiplier * regime_df['bb_std'])
            regime_df['bb_lower'] = regime_df['sma_bb'] - (std_multiplier * regime_df['bb_std'])
            regime_df['bb_width'] = (regime_df['bb_upper'] - regime_df['bb_lower']) / regime_df['close']
        
        # 3. Volatility Features
        # Average True Range (ATR) - normalized by price
        if 'atr' in regime_df.columns:
            regime_df['normalized_atr'] = regime_df['atr'] / regime_df['close']
        else:
            # Calculate a simple proxy for ATR
            high_low = regime_df['high'] - regime_df['low']
            high_close = np.abs(regime_df['high'] - regime_df['close'].shift())
            low_close = np.abs(regime_df['low'] - regime_df['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            regime_df['atr_proxy'] = true_range.rolling(window=14).mean()
            regime_df['normalized_atr'] = regime_df['atr_proxy'] / regime_df['close']
        
        # Price volatility (standard deviation of returns)
        for window in [5, 10, 20]:
            returns = regime_df['close'].pct_change()
            regime_df[f'volatility_{window}'] = returns.rolling(window=window).std()
        
        # 4. Technical Indicator Features
        # RSI and extremes
        if 'rsi' in regime_df.columns:
            # RSI distance from midpoint (50) - higher means more extreme
            regime_df['rsi_extreme'] = np.abs(regime_df['rsi'] - 50) / 50
        
        # MACD histogram (trend strength and direction)
        if 'macd' in regime_df.columns and 'macd_signal' in regime_df.columns:
            regime_df['macd_histogram'] = regime_df['macd'] - regime_df['macd_signal']
            regime_df['macd_histogram_abs'] = np.abs(regime_df['macd_histogram'])
        
        # ADX (trend strength)
        if 'adx' in regime_df.columns:
            # ADX > 25 indicates a strong trend
            regime_df['strong_trend'] = (regime_df['adx'] > 25).astype(int)
        
        # Drop rows with NaN values (due to lookback windows)
        regime_df = regime_df.dropna()
        
        # Select only the feature columns for regime detection
        feature_cols = [
            'price_change_5', 'price_change_10', 'price_change_20',
            'sma_ratio_20_50', 'sma_ratio_20_100',
            'bb_width', 'normalized_atr',
            'volatility_5', 'volatility_10', 'volatility_20'
        ]
        
        # Add technical indicators if available
        if 'rsi_extreme' in regime_df.columns:
            feature_cols.append('rsi_extreme')
        
        if 'macd_histogram_abs' in regime_df.columns:
            feature_cols.append('macd_histogram_abs')
        
        if 'strong_trend' in regime_df.columns:
            feature_cols.append('strong_trend')
        
        # Create feature matrix
        features_df = regime_df[feature_cols]
        
        # Add timestamp column for reference
        features_df['timestamp'] = regime_df['timestamp']
        
        return features_df
    
    def train_regime_model(self, lookback_days=90, n_clusters=4):
        """
        Train a K-means clustering model to identify market regimes
        
        Args:
            lookback_days: How many days of historical data to use
            n_clusters: Number of market regimes to identify
            
        Returns:
            Boolean indicating success
        """
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
        
        if df.empty:
            logging.error(f"No data available for {self.symbol}/{self.interval}")
            return False
        
        # Prepare features for regime detection
        features_df = self.prepare_regime_features(df)
        
        if features_df.empty:
            logging.error(f"Failed to prepare features for {self.symbol}/{self.interval}")
            return False
        
        # Store timestamp column separately
        timestamps = features_df['timestamp']
        features_df = features_df.drop(columns=['timestamp'])
        
        # Scale features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features_df)
        
        # Train K-means clustering model
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.kmeans_model.fit(features_scaled)
        
        # Get cluster labels
        labels = self.kmeans_model.labels_
        
        # Analyze clusters to map them to regime types
        # This is a heuristic approach to label the clusters
        cluster_features = {}
        for i in range(n_clusters):
            cluster_data = features_df.iloc[labels == i]
            cluster_features[i] = {
                'count': len(cluster_data),
                'mean': cluster_data.mean().to_dict(),
                'std': cluster_data.std().to_dict()
            }
        
        # Map clusters to regime types
        regime_mapping = self._map_clusters_to_regimes(cluster_features)
        
        # Calculate transition probabilities between regimes
        transitions = self._calculate_transition_probabilities(labels)
        
        # Save model metadata
        metadata = {
            'symbol': self.symbol,
            'interval': self.interval,
            'n_clusters': n_clusters,
            'cluster_features': cluster_features,
            'regime_mapping': regime_mapping,
            'transition_probabilities': transitions,
            'feature_columns': list(features_df.columns),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save model
        joblib.dump(self.kmeans_model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        logging.info(f"Trained market regime detection model for {self.symbol}/{self.interval}")
        return True
    
    def _map_clusters_to_regimes(self, cluster_features):
        """
        Map K-means clusters to meaningful market regimes
        
        Args:
            cluster_features: Dictionary with cluster statistics
            
        Returns:
            Dictionary mapping cluster IDs to regime types
        """
        # Initialize mapping
        mapping = {}
        cluster_ids = list(cluster_features.keys())
        
        for cluster_id in cluster_ids:
            features = cluster_features[cluster_id]['mean']
            
            # Check for ranging market
            if (abs(features['price_change_20']) < 0.05 and 
                features['bb_width'] < 0.06 and
                features['volatility_20'] < 0.015):
                mapping[cluster_id] = self.REGIME_TYPES[0]  # RANGING
            
            # Check for uptrend
            elif (features['price_change_20'] > 0.05 and
                  features['sma_ratio_20_50'] > 1.01):
                mapping[cluster_id] = self.REGIME_TYPES[1]  # TRENDING_UP
            
            # Check for downtrend
            elif (features['price_change_20'] < -0.05 and
                  features['sma_ratio_20_50'] < 0.99):
                mapping[cluster_id] = self.REGIME_TYPES[2]  # TRENDING_DOWN
            
            # Check for volatility
            elif features['volatility_20'] > 0.02 or features['normalized_atr'] > 0.03:
                mapping[cluster_id] = self.REGIME_TYPES[3]  # VOLATILE
            
            # Default to RANGING if no clear pattern
            else:
                mapping[cluster_id] = self.REGIME_TYPES[0]  # RANGING
        
        return mapping
    
    def _calculate_transition_probabilities(self, labels):
        """
        Calculate transition probabilities between market regimes
        
        Args:
            labels: Array of cluster labels in time sequence
            
        Returns:
            Dictionary with transition probabilities
        """
        n_clusters = len(set(labels))
        transitions = np.zeros((n_clusters, n_clusters))
        
        # Count transitions
        for i in range(len(labels) - 1):
            curr_label = labels[i]
            next_label = labels[i + 1]
            transitions[curr_label, next_label] += 1
        
        # Convert to probabilities
        for i in range(n_clusters):
            row_sum = np.sum(transitions[i, :])
            if row_sum > 0:
                transitions[i, :] = transitions[i, :] / row_sum
        
        # Convert to nested dictionary
        transition_dict = {}
        for i in range(n_clusters):
            transition_dict[str(i)] = {}
            for j in range(n_clusters):
                transition_dict[str(i)][str(j)] = float(transitions[i, j])
        
        return transition_dict
    
    def load_model(self):
        """
        Load a trained market regime detection model
        
        Returns:
            Boolean indicating success
        """
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.kmeans_model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                
                # Load metadata if available
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                
                logging.info(f"Loaded market regime detection model for {self.symbol}/{self.interval}")
                return True
            else:
                logging.warning(f"Model files not found for {self.symbol}/{self.interval}")
                return False
        except Exception as e:
            logging.error(f"Error loading market regime model: {e}")
            return False
    
    def detect_current_regime(self):
        """
        Detect the current market regime
        
        Returns:
            Dictionary with regime information
        """
        # Check if model is loaded
        if self.kmeans_model is None or self.scaler is None:
            if not self.load_model():
                logging.error(f"No trained model available for {self.symbol}/{self.interval}")
                return None
        
        # Get recent data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)  # Need enough data for feature calculation
        
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
        
        # Prepare features for regime detection
        features_df = self.prepare_regime_features(df)
        
        if features_df.empty:
            logging.error(f"Failed to prepare features for {self.symbol}/{self.interval}")
            return None
        
        # Store timestamp column separately
        timestamps = features_df['timestamp']
        features_df = features_df.drop(columns=['timestamp'])
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Predict clusters
        labels = self.kmeans_model.predict(features_scaled)
        
        # Get the most recent regime
        current_regime_cluster = labels[-1]
        
        # Get the regime name from mapping if available
        if hasattr(self, 'metadata') and 'regime_mapping' in self.metadata:
            current_regime_name = self.metadata['regime_mapping'].get(str(current_regime_cluster),
                                                                    self.REGIME_TYPES.get(0, "UNKNOWN"))
        else:
            # Fallback to default mapping
            current_regime_name = self.REGIME_TYPES.get(current_regime_cluster, "UNKNOWN")
        
        # Calculate regime stability (how long this regime has lasted)
        stability = 1
        for i in range(len(labels) - 2, -1, -1):
            if labels[i] == current_regime_cluster:
                stability += 1
            else:
                break
        
        # Get regime transition probabilities if available
        transition_probs = {}
        if (hasattr(self, 'metadata') and 
            'transition_probabilities' in self.metadata and 
            str(current_regime_cluster) in self.metadata['transition_probabilities']):
            transition_probs = self.metadata['transition_probabilities'][str(current_regime_cluster)]
        
        # Get next most likely regime
        if transition_probs:
            next_regime_cluster = max(transition_probs.items(), key=lambda x: x[1])[0]
            next_regime_prob = transition_probs[next_regime_cluster]
            
            if hasattr(self, 'metadata') and 'regime_mapping' in self.metadata:
                next_regime_name = self.metadata['regime_mapping'].get(next_regime_cluster,
                                                                    self.REGIME_TYPES.get(0, "UNKNOWN"))
            else:
                next_regime_name = self.REGIME_TYPES.get(int(next_regime_cluster), "UNKNOWN")
        else:
            next_regime_cluster = None
            next_regime_prob = None
            next_regime_name = "UNKNOWN"
        
        # Create regime information dictionary
        regime_info = {
            'current_regime': {
                'cluster': int(current_regime_cluster),
                'name': current_regime_name,
                'stability': stability,
                'recent_history': [int(l) for l in labels[-min(10, len(labels)):]],
            },
            'next_likely_regime': {
                'cluster': int(next_regime_cluster) if next_regime_cluster is not None else None,
                'name': next_regime_name,
                'probability': float(next_regime_prob) if next_regime_prob is not None else None
            },
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'interval': self.interval
        }
        
        # Add recent price information
        regime_info['price_info'] = {
            'current_price': float(df['close'].iloc[-1]),
            'price_change_24h': float(df['close'].iloc[-1] / df['close'].iloc[-24] - 1) if len(df) >= 24 else None,
            'volatility': float(df['close'].pct_change().std() * 100)  # As percentage
        }
        
        return regime_info
    
    def get_model_weights_for_regime(self, regime_name):
        """
        Get optimal model weights for current market regime
        
        Args:
            regime_name: String name of current regime
            
        Returns:
            Dictionary with model weights for the regime
        """
        # Define weights for different models in different regimes
        # These are heuristic weights based on typical model performance in different regimes
        regime_weights = {
            "RANGING": {
                "gbr": 0.3,   # Gradient Boosting is good at capturing non-linear patterns
                "rf": 0.3,    # Random Forest performs well in ranging markets
                "lr": 0.4     # Linear Regression can be effective for mean-reversion
            },
            "TRENDING_UP": {
                "gbr": 0.5,   # Gradient Boosting works well for trends
                "rf": 0.4,    # Random Forest can capture trend patterns
                "lr": 0.1     # Linear Regression is less effective for trends
            },
            "TRENDING_DOWN": {
                "gbr": 0.5,   # Gradient Boosting works well for trends
                "rf": 0.4,    # Random Forest can capture trend patterns
                "lr": 0.1     # Linear Regression is less effective for trends
            },
            "VOLATILE": {
                "gbr": 0.6,   # Gradient Boosting handles volatility well
                "rf": 0.3,    # Random Forest is moderately effective in volatile markets
                "lr": 0.1     # Linear Regression performs poorly in volatile markets
            }
        }
        
        # Return weights for the specified regime, defaulting to RANGING if not found
        return regime_weights.get(regime_name, regime_weights["RANGING"])
    
    def get_trading_strategy_for_regime(self, regime_name):
        """
        Get recommended trading strategy for current market regime
        
        Args:
            regime_name: String name of current regime
            
        Returns:
            Dictionary with trading strategy recommendations
        """
        # Define strategies for different regimes
        regime_strategies = {
            "RANGING": {
                "name": "Mean Reversion",
                "description": "Look for overbought/oversold conditions and trade reversals back to the mean.",
                "indicators": ["RSI", "Bollinger Bands", "Stochastic"],
                "timeframes": ["15m", "1h", "4h"],  # Shorter timeframes work well for ranging markets
                "risk_level": "Low to Medium",
                "stop_loss": "Tight (1-2%)",
                "take_profit": "Small but frequent (1-3%)"
            },
            "TRENDING_UP": {
                "name": "Trend Following",
                "description": "Look for pullbacks and enter in the direction of the trend.",
                "indicators": ["Moving Averages", "MACD", "ADX"],
                "timeframes": ["4h", "1d"],  # Longer timeframes work better for trends
                "risk_level": "Medium",
                "stop_loss": "Below key support levels (3-5%)",
                "take_profit": "Trailing stops or multiple targets (5-15%)"
            },
            "TRENDING_DOWN": {
                "name": "Trend Following (Short)",
                "description": "Look for bounces and enter short positions in the direction of the downtrend.",
                "indicators": ["Moving Averages", "MACD", "ADX"],
                "timeframes": ["4h", "1d"],  # Longer timeframes work better for trends
                "risk_level": "Medium to High",
                "stop_loss": "Above key resistance levels (3-5%)",
                "take_profit": "Trailing stops or multiple targets (5-15%)"
            },
            "VOLATILE": {
                "name": "Breakout Trading",
                "description": "Wait for consolidation patterns and trade breakouts with momentum.",
                "indicators": ["Volume", "ATR", "Bollinger Bands"],
                "timeframes": ["1h", "4h"],  # Moderate timeframes for volatility
                "risk_level": "High",
                "stop_loss": "Wider stops (5-8%)",
                "take_profit": "Larger targets (10-20%) or trailing stops"
            }
        }
        
        # Return strategy for the specified regime, defaulting to RANGING if not found
        return regime_strategies.get(regime_name, regime_strategies["RANGING"])


# Function to train market regime models for multiple symbol/interval combinations
def train_all_regime_models(symbols, intervals, lookback_days=90):
    """
    Train market regime models for all symbol/interval combinations
    
    Args:
        symbols: List of symbols to train models for
        intervals: List of intervals to train models for
        lookback_days: How many days of historical data to use
        
    Returns:
        Dictionary with training results
    """
    results = {}
    
    for symbol in symbols:
        results[symbol] = {}
        
        for interval in intervals:
            logging.info(f"Training market regime model for {symbol}/{interval}")
            
            try:
                # Initialize and train regime model
                detector = MarketRegimeDetector(symbol, interval)
                success = detector.train_regime_model(lookback_days=lookback_days)
                
                if success:
                    # Get current regime
                    regime_info = detector.detect_current_regime()
                    
                    results[symbol][interval] = {
                        'status': 'success',
                        'current_regime': regime_info
                    }
                else:
                    results[symbol][interval] = {
                        'status': 'failed',
                        'error': 'Training failed'
                    }
            except Exception as e:
                logging.error(f"Error training regime model for {symbol}/{interval}: {e}")
                results[symbol][interval] = {
                    'status': 'error',
                    'error': str(e)
                }
                
    return results