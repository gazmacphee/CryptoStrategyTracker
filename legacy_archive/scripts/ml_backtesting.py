"""
ML Backtesting Module

This module provides functionality to:
1. Evaluate ML prediction performance on historical data
2. Visualize prediction accuracy across different market conditions
3. Compare different ML models and ensemble configurations
4. Generate performance reports and metrics

By backtesting ML predictions, users can gain confidence in the models
and better understand their strengths and weaknesses.
"""

import os
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Local imports
from database import get_historical_data, get_indicators
from ml_prediction import MLPredictor
from ml_ensemble import EnsemblePredictor
from ml_market_regime import MarketRegimeDetector
from ml_sentiment_integration import SentimentIntegrator

class MLBacktester:
    """
    Backtester for ML prediction models
    """
    
    def __init__(self, symbol, interval):
        """
        Initialize the backtester
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Time interval (e.g., '1h', '4h', '1d')
        """
        self.symbol = symbol
        self.interval = interval
        
        # Initialize ML predictors
        self.predictor = MLPredictor(symbol, interval)
        self.ensemble_predictor = EnsemblePredictor(symbol, interval)
        
        # Initialize regime detector
        self.regime_detector = MarketRegimeDetector(symbol, interval)
        
        # Initialize sentiment integrator
        self.sentiment_integrator = SentimentIntegrator(symbol, interval)
        
        # Results storage
        self.backtest_results = None
        self.performance_metrics = None
        
        # Result paths
        self.results_path = os.path.join('models', f"{symbol}_{interval}_backtest_results.json")
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
    
    def run_backtest(self, start_date=None, end_date=None, days_back=90, 
                    step_size=1, prediction_horizon=1, include_regimes=True, 
                    include_sentiment=True):
        """
        Run a backtest of ML predictions on historical data
        
        Args:
            start_date: Start date for backtest (default: days_back from end_date)
            end_date: End date for backtest (default: current date)
            days_back: How many days to look back if start_date not provided
            step_size: How many periods to step forward for each prediction
            prediction_horizon: How many periods ahead to predict
            include_regimes: Whether to include market regime detection
            include_sentiment: Whether to include sentiment analysis
            
        Returns:
            DataFrame with backtest results
        """
        # Set date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=days_back)
        
        # Get historical data
        df = get_historical_data(self.symbol, self.interval, start_date, end_date)
        
        # Get indicators data
        indicators_df = get_indicators(self.symbol, self.interval, start_date, end_date)
        
        # Merge price data with indicators
        if not indicators_df.empty:
            df = pd.merge(df, indicators_df, on='timestamp', how='left')
        
        if df.empty:
            logging.error(f"No historical data available for {self.symbol}/{self.interval}")
            return None
        
        # Ensure the models are loaded
        try:
            self.predictor.load_model()
            has_base_model = True
        except Exception:
            logging.warning(f"Base model for {self.symbol}/{self.interval} not available")
            has_base_model = False
        
        try:
            self.ensemble_predictor.load_model()
            has_ensemble_model = True
        except Exception:
            logging.warning(f"Ensemble model for {self.symbol}/{self.interval} not available")
            has_ensemble_model = False
        
        # Train models if not available
        if not has_base_model:
            try:
                logging.info(f"Training base model for {self.symbol}/{self.interval}")
                self.predictor.train_model(lookback_days=days_back)
                has_base_model = True
            except Exception as e:
                logging.error(f"Error training base model: {e}")
        
        if not has_ensemble_model:
            try:
                logging.info(f"Training ensemble model for {self.symbol}/{self.interval}")
                self.ensemble_predictor.train_model(lookback_days=days_back)
                has_ensemble_model = True
            except Exception as e:
                logging.error(f"Error training ensemble model: {e}")
        
        # If neither model is available, we can't run the backtest
        if not has_base_model and not has_ensemble_model:
            logging.error(f"No models available for {self.symbol}/{self.interval}")
            return None
        
        # Prepare data for prediction
        df = df.sort_values('timestamp')
        
        # Calculate actual future price changes for comparison
        for i in range(1, prediction_horizon + 1):
            df[f'actual_change_{i}'] = df['close'].pct_change(periods=i).shift(-i)
        
        # Initialize market regimes if enabled
        regime_labels = None
        if include_regimes:
            try:
                # Train regime model if not already trained
                if not os.path.exists(self.regime_detector.model_path):
                    self.regime_detector.train_regime_model(lookback_days=days_back)
                
                # Prepare features for regime detection
                regime_features = self.regime_detector.prepare_regime_features(df)
                
                if not regime_features.empty:
                    # Extract timestamps
                    regime_timestamps = regime_features['timestamp']
                    regime_features = regime_features.drop('timestamp', axis=1)
                    
                    # Scale features
                    regime_features_scaled = self.regime_detector.scaler.transform(regime_features)
                    
                    # Predict regimes
                    regime_labels = self.regime_detector.kmeans_model.predict(regime_features_scaled)
                    
                    # Map labels to regime names
                    regime_names = []
                    for label in regime_labels:
                        if (hasattr(self.regime_detector, 'metadata') and 
                            'regime_mapping' in self.regime_detector.metadata):
                            regime_name = self.regime_detector.metadata['regime_mapping'].get(
                                str(label), self.regime_detector.REGIME_TYPES.get(0, "UNKNOWN")
                            )
                        else:
                            regime_name = self.regime_detector.REGIME_TYPES.get(label, "UNKNOWN")
                        regime_names.append(regime_name)
                    
                    # Create DataFrame with regime info
                    regime_df = pd.DataFrame({
                        'timestamp': regime_timestamps,
                        'regime_label': regime_labels,
                        'regime_name': regime_names
                    })
                    
                    # Merge with main DataFrame
                    df = pd.merge(df, regime_df, on='timestamp', how='left')
            except Exception as e:
                logging.error(f"Error setting up market regimes: {e}")
                include_regimes = False
        
        # Initialize sentiment data if enabled
        if include_sentiment:
            try:
                # Fetch sentiment data for the entire period
                sentiment_df = self.sentiment_integrator.fetch_sentiment_data(days_back=days_back + 10)
                
                if sentiment_df is not None and not sentiment_df.empty:
                    # Aggregate sentiment data by day
                    sentiment_df['date'] = pd.to_datetime(sentiment_df['timestamp']).dt.date
                    sentiment_daily = sentiment_df.groupby('date').agg({
                        'sentiment_score': 'mean',
                        'volume': 'sum'
                    }).reset_index()
                    
                    # Convert timestamp in main df to date for merging
                    df['date'] = pd.to_datetime(df['timestamp']).dt.date
                    
                    # Merge with main DataFrame
                    df = pd.merge(df, sentiment_daily, on='date', how='left')
                    
                    # Forward fill missing sentiment values
                    df['sentiment_score'] = df['sentiment_score'].ffill()
                    df['volume'] = df['volume'].ffill()
                    
                    # Drop the date column
                    df = df.drop('date', axis=1)
                else:
                    logging.warning(f"No sentiment data available for {self.symbol}")
                    include_sentiment = False
            except Exception as e:
                logging.error(f"Error setting up sentiment data: {e}")
                include_sentiment = False
        
        # Initialize results storage
        results = []
        
        # Create time windows for backtest
        test_indices = list(range(0, len(df) - prediction_horizon, step_size))
        
        # Run backtest for each time window
        for idx in test_indices:
            # Get the data up to this point
            test_data = df.iloc[:idx + 1]
            
            # Skip if not enough data
            if len(test_data) < self.predictor.feature_window + 5:
                continue
            
            # Get the timestamp for this prediction
            timestamp = test_data['timestamp'].iloc[-1]
            
            # Get the current price
            current_price = test_data['close'].iloc[-1]
            
            # Get the actual future price changes
            actual_changes = [df[f'actual_change_{i}'].iloc[idx] for i in range(1, prediction_horizon + 1)
                             if idx < len(df) and f'actual_change_{i}' in df.columns]
            
            if not actual_changes or any(pd.isna(actual_changes)):
                continue  # Skip if we don't have actual future prices
            
            # Use the first prediction horizon for comparisons
            actual_change = actual_changes[0]
            
            # Get the actual future price
            actual_future_price = current_price * (1 + actual_change)
            
            # Predictions from different models
            predictions = {}
            
            # Base model prediction
            if has_base_model:
                try:
                    # Prepare features for base model
                    X, _ = self.predictor.prepare_data(test_data)
                    
                    if X is not None and not X.empty:
                        # Scale features
                        X_scaled = self.predictor.scaler.transform(X.iloc[-1:])
                        
                        # Make prediction
                        base_pred_change = float(self.predictor.model.predict(X_scaled)[0])
                        base_pred_price = current_price * (1 + base_pred_change)
                        
                        # Calculate confidence
                        base_confidence = self.predictor._get_prediction_confidence(base_pred_change)
                        
                        # Store prediction
                        predictions['base'] = {
                            'predicted_change': base_pred_change,
                            'predicted_price': base_pred_price,
                            'confidence': base_confidence,
                            'error': base_pred_change - actual_change,
                            'abs_error': abs(base_pred_change - actual_change),
                            'direction_correct': (base_pred_change > 0 and actual_change > 0) or
                                              (base_pred_change < 0 and actual_change < 0)
                        }
                except Exception as e:
                    logging.warning(f"Error making base prediction at {timestamp}: {e}")
            
            # Ensemble model prediction
            if has_ensemble_model:
                try:
                    # Prepare features for ensemble model
                    X, _ = self.ensemble_predictor.prepare_data(test_data)
                    
                    if X is not None and not X.empty:
                        # Scale features
                        X_scaled = self.ensemble_predictor.scaler.transform(X.iloc[-1:])
                        
                        # Make prediction
                        ensemble_pred_change = float(self.ensemble_predictor.model.predict(X_scaled)[0])
                        ensemble_pred_price = current_price * (1 + ensemble_pred_change)
                        
                        # Calculate confidence
                        ensemble_confidence = self.ensemble_predictor._get_prediction_confidence(ensemble_pred_change)
                        
                        # Store prediction
                        predictions['ensemble'] = {
                            'predicted_change': ensemble_pred_change,
                            'predicted_price': ensemble_pred_price,
                            'confidence': ensemble_confidence,
                            'error': ensemble_pred_change - actual_change,
                            'abs_error': abs(ensemble_pred_change - actual_change),
                            'direction_correct': (ensemble_pred_change > 0 and actual_change > 0) or
                                             (ensemble_pred_change < 0 and actual_change < 0)
                        }
                except Exception as e:
                    logging.warning(f"Error making ensemble prediction at {timestamp}: {e}")
            
            # Add regime-adjusted prediction if enabled
            if include_regimes and 'regime_name' in df.columns:
                try:
                    # Get the current regime
                    current_regime = df['regime_name'].iloc[idx]
                    
                    # Use the best performing model based on the regime
                    if has_ensemble_model:
                        # Start with ensemble prediction
                        regime_pred_change = predictions['ensemble']['predicted_change']
                        regime_confidence = predictions['ensemble']['confidence']
                        
                        # Get optimal weights for this regime
                        regime_weights = self.regime_detector.get_model_weights_for_regime(current_regime)
                        
                        # Adjust prediction if we have sub-model predictions
                        if hasattr(self.ensemble_predictor, 'gbr_model') and self.ensemble_predictor.gbr_model:
                            # Make predictions with each sub-model
                            gbr_pred = self.ensemble_predictor.gbr_model.predict(X_scaled)[0]
                            rf_pred = self.ensemble_predictor.rf_model.predict(X_scaled)[0]
                            lr_pred = self.ensemble_predictor.lr_model.predict(X_scaled)[0]
                            
                            # Weighted average based on regime
                            regime_pred_change = (
                                gbr_pred * regime_weights['gbr'] +
                                rf_pred * regime_weights['rf'] +
                                lr_pred * regime_weights['lr']
                            )
                            
                            # Adjust confidence based on regime certainty
                            # Higher confidence in regimes where the model historically performs well
                            if current_regime in ["TRENDING_UP", "TRENDING_DOWN"]:
                                # Trend-following models are more reliable in trending markets
                                regime_confidence = min(regime_confidence * 1.1, 1.0)
                            elif current_regime == "RANGING":
                                # Models can struggle in ranging markets
                                regime_confidence = max(regime_confidence * 0.9, 0.1)
                    else:
                        # Fall back to base model
                        regime_pred_change = predictions['base']['predicted_change']
                        regime_confidence = predictions['base']['confidence']
                    
                    # Calculate predicted price
                    regime_pred_price = current_price * (1 + regime_pred_change)
                    
                    # Store prediction
                    predictions['regime_adjusted'] = {
                        'predicted_change': regime_pred_change,
                        'predicted_price': regime_pred_price,
                        'confidence': regime_confidence,
                        'error': regime_pred_change - actual_change,
                        'abs_error': abs(regime_pred_change - actual_change),
                        'direction_correct': (regime_pred_change > 0 and actual_change > 0) or
                                        (regime_pred_change < 0 and actual_change < 0),
                        'regime': current_regime
                    }
                except Exception as e:
                    logging.warning(f"Error making regime-adjusted prediction at {timestamp}: {e}")
            
            # Add sentiment-adjusted prediction if enabled
            if include_sentiment and 'sentiment_score' in df.columns:
                try:
                    # Get the current sentiment
                    current_sentiment = df['sentiment_score'].iloc[idx]
                    
                    # Use the best performing model as base
                    if has_ensemble_model:
                        base_sent_change = predictions['ensemble']['predicted_change']
                        base_sent_confidence = predictions['ensemble']['confidence']
                    else:
                        base_sent_change = predictions['base']['predicted_change']
                        base_sent_confidence = predictions['base']['confidence']
                    
                    # Define sentiment adjustment factor
                    sentiment_factor = 0.2
                    
                    # Adjust prediction based on sentiment
                    sentiment_adjustment = current_sentiment * sentiment_factor
                    sentiment_pred_change = base_sent_change + sentiment_adjustment
                    
                    # Adjust confidence based on sentiment alignment
                    if (base_sent_change > 0 and current_sentiment > 0) or (base_sent_change < 0 and current_sentiment < 0):
                        # Sentiment agrees with prediction, increase confidence
                        agreement_factor = 0.1
                        sentiment_confidence = min(base_sent_confidence + (abs(current_sentiment) * agreement_factor), 1.0)
                    else:
                        # Sentiment disagrees with prediction, decrease confidence
                        disagreement_factor = 0.15
                        sentiment_confidence = max(base_sent_confidence - (abs(current_sentiment) * disagreement_factor), 0.1)
                    
                    # Calculate predicted price
                    sentiment_pred_price = current_price * (1 + sentiment_pred_change)
                    
                    # Store prediction
                    predictions['sentiment_adjusted'] = {
                        'predicted_change': sentiment_pred_change,
                        'predicted_price': sentiment_pred_price,
                        'confidence': sentiment_confidence,
                        'error': sentiment_pred_change - actual_change,
                        'abs_error': abs(sentiment_pred_change - actual_change),
                        'direction_correct': (sentiment_pred_change > 0 and actual_change > 0) or
                                         (sentiment_pred_change < 0 and actual_change < 0),
                        'sentiment': current_sentiment
                    }
                except Exception as e:
                    logging.warning(f"Error making sentiment-adjusted prediction at {timestamp}: {e}")
            
            # Store the result
            result = {
                'timestamp': timestamp,
                'current_price': current_price,
                'actual_change': actual_change,
                'actual_future_price': actual_future_price,
                'models': predictions
            }
            
            # Add market regime if available
            if include_regimes and 'regime_name' in df.columns:
                result['regime'] = df['regime_name'].iloc[idx]
            
            # Add sentiment if available
            if include_sentiment and 'sentiment_score' in df.columns:
                result['sentiment'] = df['sentiment_score'].iloc[idx]
            
            results.append(result)
        
        # Convert to DataFrame for easy analysis
        self.backtest_results = results
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        # Save results
        self._save_results()
        
        return results
    
    def _calculate_performance_metrics(self):
        """
        Calculate performance metrics for backtest results
        """
        if not self.backtest_results:
            return None
        
        # Initialize metrics
        metrics = {
            'overall': {},
            'by_model': {},
            'by_regime': {},
            'by_sentiment': {}
        }
        
        # Get list of models
        models = set()
        for result in self.backtest_results:
            models.update(result['models'].keys())
        
        # Calculate overall metrics for each model
        for model in models:
            model_metrics = {
                'count': 0,
                'mae': 0,
                'rmse': 0,
                'direction_accuracy': 0,
                'avg_confidence': 0
            }
            
            for result in self.backtest_results:
                if model in result['models']:
                    model_metrics['count'] += 1
                    model_metrics['mae'] += result['models'][model]['abs_error']
                    model_metrics['rmse'] += result['models'][model]['error'] ** 2
                    model_metrics['direction_accuracy'] += int(result['models'][model]['direction_correct'])
                    model_metrics['avg_confidence'] += result['models'][model]['confidence']
            
            # Calculate averages
            if model_metrics['count'] > 0:
                model_metrics['mae'] /= model_metrics['count']
                model_metrics['rmse'] = (model_metrics['rmse'] / model_metrics['count']) ** 0.5
                model_metrics['direction_accuracy'] /= model_metrics['count']
                model_metrics['avg_confidence'] /= model_metrics['count']
            
            metrics['by_model'][model] = model_metrics
        
        # Calculate metrics by regime if available
        regimes = set()
        for result in self.backtest_results:
            if 'regime' in result:
                regimes.add(result['regime'])
        
        if regimes:
            for regime in regimes:
                regime_metrics = {model: {
                    'count': 0,
                    'mae': 0,
                    'rmse': 0,
                    'direction_accuracy': 0,
                    'avg_confidence': 0
                } for model in models}
                
                for result in self.backtest_results:
                    if 'regime' in result and result['regime'] == regime:
                        for model in result['models']:
                            regime_metrics[model]['count'] += 1
                            regime_metrics[model]['mae'] += result['models'][model]['abs_error']
                            regime_metrics[model]['rmse'] += result['models'][model]['error'] ** 2
                            regime_metrics[model]['direction_accuracy'] += int(result['models'][model]['direction_correct'])
                            regime_metrics[model]['avg_confidence'] += result['models'][model]['confidence']
                
                # Calculate averages
                for model in models:
                    if regime_metrics[model]['count'] > 0:
                        regime_metrics[model]['mae'] /= regime_metrics[model]['count']
                        regime_metrics[model]['rmse'] = (regime_metrics[model]['rmse'] / regime_metrics[model]['count']) ** 0.5
                        regime_metrics[model]['direction_accuracy'] /= regime_metrics[model]['count']
                        regime_metrics[model]['avg_confidence'] /= regime_metrics[model]['count']
                
                metrics['by_regime'][regime] = regime_metrics
        
        # Calculate metrics by sentiment range if available
        sentiment_ranges = [
            ('very_negative', -1.0, -0.5),
            ('negative', -0.5, -0.1),
            ('neutral', -0.1, 0.1),
            ('positive', 0.1, 0.5),
            ('very_positive', 0.5, 1.0)
        ]
        
        if any('sentiment' in result for result in self.backtest_results):
            for range_name, min_val, max_val in sentiment_ranges:
                sentiment_metrics = {model: {
                    'count': 0,
                    'mae': 0,
                    'rmse': 0,
                    'direction_accuracy': 0,
                    'avg_confidence': 0
                } for model in models}
                
                for result in self.backtest_results:
                    if 'sentiment' in result and min_val <= result['sentiment'] < max_val:
                        for model in result['models']:
                            sentiment_metrics[model]['count'] += 1
                            sentiment_metrics[model]['mae'] += result['models'][model]['abs_error']
                            sentiment_metrics[model]['rmse'] += result['models'][model]['error'] ** 2
                            sentiment_metrics[model]['direction_accuracy'] += int(result['models'][model]['direction_correct'])
                            sentiment_metrics[model]['avg_confidence'] += result['models'][model]['confidence']
                
                # Calculate averages
                for model in models:
                    if sentiment_metrics[model]['count'] > 0:
                        sentiment_metrics[model]['mae'] /= sentiment_metrics[model]['count']
                        sentiment_metrics[model]['rmse'] = (sentiment_metrics[model]['rmse'] / sentiment_metrics[model]['count']) ** 0.5
                        sentiment_metrics[model]['direction_accuracy'] /= sentiment_metrics[model]['count']
                        sentiment_metrics[model]['avg_confidence'] /= sentiment_metrics[model]['count']
                
                metrics['by_sentiment'][range_name] = sentiment_metrics
        
        # Calculate overall metrics
        metrics['overall'] = {
            'total_predictions': len(self.backtest_results),
            'date_range': {
                'start': min(r['timestamp'] for r in self.backtest_results),
                'end': max(r['timestamp'] for r in self.backtest_results)
            },
            'best_model': max(metrics['by_model'].items(), key=lambda x: x[1]['direction_accuracy'])[0]
        }
        
        self.performance_metrics = metrics
        return metrics
    
    def _save_results(self):
        """
        Save backtest results and metrics to disk
        """
        if not self.backtest_results or not self.performance_metrics:
            return
        
        # Create a dictionary with all results
        save_data = {
            'symbol': self.symbol,
            'interval': self.interval,
            'timestamp': datetime.now().isoformat(),
            'results': self.backtest_results,
            'metrics': self.performance_metrics
        }
        
        # Save to file
        with open(self.results_path, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            json.dump(save_data, f, default=str)
        
        logging.info(f"Saved backtest results to {self.results_path}")
    
    def load_results(self):
        """
        Load saved backtest results from disk
        
        Returns:
            Dictionary with backtest results and metrics
        """
        if not os.path.exists(self.results_path):
            logging.warning(f"No saved backtest results found for {self.symbol}/{self.interval}")
            return None
        
        try:
            with open(self.results_path, 'r') as f:
                data = json.load(f)
            
            self.backtest_results = data['results']
            self.performance_metrics = data['metrics']
            
            return data
        except Exception as e:
            logging.error(f"Error loading backtest results: {e}")
            return None
    
    def create_performance_visualizations(self, render_to=None):
        """
        Create visualizations of backtest performance
        
        Args:
            render_to: Streamlit container to render plots in (optional)
            
        Returns:
            Dictionary with plotly figures for each visualization
        """
        if not self.backtest_results or not self.performance_metrics:
            if not self.load_results():
                logging.warning("No backtest results available for visualization")
                return None
        
        figures = {}
        
        # Create DataFrame from results for easier plotting
        results_df = pd.DataFrame([
            {
                'timestamp': pd.to_datetime(r['timestamp']),
                'current_price': r['current_price'],
                'actual_change': r['actual_change'],
                'actual_future_price': r['actual_future_price'],
                'regime': r.get('regime', 'Unknown'),
                'sentiment': r.get('sentiment', 0),
                **{f"{model}_predicted_change": r['models'][model]['predicted_change'] 
                   for model in r['models']},
                **{f"{model}_confidence": r['models'][model]['confidence'] 
                   for model in r['models']},
                **{f"{model}_error": r['models'][model]['error'] 
                   for model in r['models']},
                **{f"{model}_direction_correct": r['models'][model]['direction_correct'] 
                   for model in r['models']}
            }
            for r in self.backtest_results if r['models']
        ])
        
        # 1. Create price chart with predictions
        price_fig = go.Figure()
        
        # Add actual price
        price_fig.add_trace(go.Scatter(
            x=results_df['timestamp'],
            y=results_df['current_price'],
            mode='lines',
            name=f"{self.symbol} Price",
            line=dict(color='black', width=2)
        ))
        
        # Add models
        colors = {
            'base': 'blue',
            'ensemble': 'green',
            'regime_adjusted': 'purple',
            'sentiment_adjusted': 'orange'
        }
        
        for model in self.performance_metrics['by_model']:
            # Create future price based on predictions
            model_col = f"{model}_predicted_change"
            if model_col in results_df.columns:
                results_df[f"{model}_future_price"] = results_df['current_price'] * (1 + results_df[model_col])
                
                # Add to chart (as scatter points)
                price_fig.add_trace(go.Scatter(
                    x=results_df['timestamp'],
                    y=results_df[f"{model}_future_price"],
                    mode='markers',
                    name=f"{model.replace('_', ' ').title()} Predictions",
                    marker=dict(
                        color=colors.get(model, 'gray'),
                        size=6,
                        opacity=0.7,
                        symbol='circle'
                    )
                ))
        
        price_fig.update_layout(
            title=f"{self.symbol}/{self.interval} - Price with Predictions",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        figures['price_chart'] = price_fig
        
        # 2. Create accuracy comparison chart
        models = list(self.performance_metrics['by_model'].keys())
        
        # Overall direction accuracy
        direction_accuracies = [self.performance_metrics['by_model'][m]['direction_accuracy'] * 100 for m in models]
        mae_values = [self.performance_metrics['by_model'][m]['mae'] * 100 for m in models]  # Convert to percentage
        
        # Create bar chart
        accuracy_fig = go.Figure()
        
        # Add direction accuracy bars
        accuracy_fig.add_trace(go.Bar(
            x=models,
            y=direction_accuracies,
            name='Direction Accuracy (%)',
            marker_color='green'
        ))
        
        # Add MAE bars
        accuracy_fig.add_trace(go.Bar(
            x=models,
            y=mae_values,
            name='Mean Absolute Error (%)',
            marker_color='red',
            visible='legendonly'  # Hide by default
        ))
        
        accuracy_fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Percentage",
            height=400,
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        figures['accuracy_chart'] = accuracy_fig
        
        # 3. Create regime performance chart if available
        if 'by_regime' in self.performance_metrics and self.performance_metrics['by_regime']:
            regimes = list(self.performance_metrics['by_regime'].keys())
            
            # Create new figure
            regime_fig = go.Figure()
            
            # Add bars for each model's accuracy in each regime
            for model in models:
                accuracies = [self.performance_metrics['by_regime'][r][model]['direction_accuracy'] * 100 
                             for r in regimes 
                             if self.performance_metrics['by_regime'][r][model]['count'] > 0]
                
                # Only include regimes with data
                valid_regimes = [r for r in regimes 
                               if model in self.performance_metrics['by_regime'][r] 
                               and self.performance_metrics['by_regime'][r][model]['count'] > 0]
                
                if valid_regimes and accuracies:
                    regime_fig.add_trace(go.Bar(
                        x=valid_regimes,
                        y=accuracies,
                        name=f"{model.replace('_', ' ').title()}",
                        marker_color=colors.get(model, 'gray')
                    ))
            
            regime_fig.update_layout(
                title="Model Performance by Market Regime",
                xaxis_title="Market Regime",
                yaxis_title="Direction Accuracy (%)",
                height=400,
                barmode='group',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            
            figures['regime_chart'] = regime_fig
        
        # 4. Create confidence vs. accuracy chart
        confidence_fig = go.Figure()
        
        # Bin predictions by confidence
        confidence_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        
        for model in models:
            # Get relevant columns
            conf_col = f"{model}_confidence"
            dir_col = f"{model}_direction_correct"
            
            if conf_col in results_df.columns and dir_col in results_df.columns:
                # Create confidence bins
                results_df[f"{model}_conf_bin"] = pd.cut(
                    results_df[conf_col], 
                    bins=confidence_bins, 
                    labels=bin_labels,
                    include_lowest=True
                )
                
                # Group by confidence bin and calculate accuracy
                bin_accuracy = results_df.groupby(f"{model}_conf_bin")[dir_col].mean() * 100
                bin_counts = results_df.groupby(f"{model}_conf_bin")[dir_col].count()
                
                # Only include bins with sufficient data
                valid_bins = bin_accuracy[bin_counts >= 5].index
                valid_accuracies = bin_accuracy[bin_counts >= 5].values
                
                if len(valid_bins) > 0:
                    confidence_fig.add_trace(go.Bar(
                        x=valid_bins,
                        y=valid_accuracies,
                        name=f"{model.replace('_', ' ').title()}",
                        marker_color=colors.get(model, 'gray')
                    ))
        
        confidence_fig.update_layout(
            title="Confidence vs. Accuracy",
            xaxis_title="Confidence Range",
            yaxis_title="Direction Accuracy (%)",
            height=400,
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        figures['confidence_chart'] = confidence_fig
        
        # 5. Create error distribution chart
        error_fig = go.Figure()
        
        for model in models:
            error_col = f"{model}_error"
            
            if error_col in results_df.columns:
                # Convert errors to percentage for better visualization
                errors_pct = results_df[error_col] * 100
                
                error_fig.add_trace(go.Histogram(
                    x=errors_pct,
                    name=f"{model.replace('_', ' ').title()}",
                    marker_color=colors.get(model, 'gray'),
                    opacity=0.7,
                    nbinsx=50
                ))
        
        error_fig.update_layout(
            title="Error Distribution",
            xaxis_title="Prediction Error (%)",
            yaxis_title="Count",
            height=400,
            barmode='overlay',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        figures['error_chart'] = error_fig
        
        # If a streamlit container is provided, render the plots directly
        if render_to:
            try:
                import streamlit as st
                
                # Price chart
                render_to.subheader("Price Predictions vs. Actual")
                render_to.plotly_chart(figures['price_chart'], use_container_width=True)
                
                # Model comparison
                render_to.subheader("Model Performance Comparison")
                render_to.plotly_chart(figures['accuracy_chart'], use_container_width=True)
                
                # Regime performance if available
                if 'regime_chart' in figures:
                    render_to.subheader("Performance by Market Regime")
                    render_to.plotly_chart(figures['regime_chart'], use_container_width=True)
                
                # Confidence vs. accuracy
                render_to.subheader("Confidence vs. Accuracy")
                render_to.plotly_chart(figures['confidence_chart'], use_container_width=True)
                
                # Error distribution
                render_to.subheader("Error Distribution")
                render_to.plotly_chart(figures['error_chart'], use_container_width=True)
            except Exception as e:
                logging.error(f"Error rendering plots to streamlit: {e}")
        
        return figures
    
    def get_performance_metrics_table(self):
        """
        Get a table of performance metrics for easy comparison
        
        Returns:
            DataFrame with performance metrics
        """
        if not self.performance_metrics:
            if not self.load_results():
                logging.warning("No performance metrics available")
                return None
        
        # Create a DataFrame for overall model metrics
        models = list(self.performance_metrics['by_model'].keys())
        
        overall_data = []
        for model in models:
            metrics = self.performance_metrics['by_model'][model]
            overall_data.append({
                'Model': model.replace('_', ' ').title(),
                'Direction Accuracy (%)': f"{metrics['direction_accuracy']*100:.1f}%",
                'MAE (%)': f"{metrics['mae']*100:.2f}%",
                'RMSE (%)': f"{metrics['rmse']*100:.2f}%",
                'Avg. Confidence': f"{metrics['confidence']*100:.1f}%",
                'Prediction Count': metrics['count']
            })
        
        overall_df = pd.DataFrame(overall_data)
        
        # Create a DataFrame for regime-specific metrics if available
        regime_df = None
        if 'by_regime' in self.performance_metrics and self.performance_metrics['by_regime']:
            regimes = list(self.performance_metrics['by_regime'].keys())
            
            regime_data = []
            for regime in regimes:
                for model in models:
                    if (model in self.performance_metrics['by_regime'][regime] and 
                        self.performance_metrics['by_regime'][regime][model]['count'] > 0):
                        
                        metrics = self.performance_metrics['by_regime'][regime][model]
                        regime_data.append({
                            'Regime': regime,
                            'Model': model.replace('_', ' ').title(),
                            'Direction Accuracy (%)': f"{metrics['direction_accuracy']*100:.1f}%",
                            'MAE (%)': f"{metrics['mae']*100:.2f}%",
                            'Prediction Count': metrics['count']
                        })
            
            regime_df = pd.DataFrame(regime_data)
        
        # Return both tables
        return {
            'overall': overall_df,
            'by_regime': regime_df
        }
    
    def get_model_recommendations(self):
        """
        Get recommendations for best models based on backtest results
        
        Returns:
            Dictionary with model recommendations
        """
        if not self.performance_metrics:
            if not self.load_results():
                logging.warning("No performance metrics available for recommendations")
                return None
        
        recommendations = {
            'overall_best': None,
            'by_regime': {},
            'explanation': ''
        }
        
        # Find overall best model
        models = list(self.performance_metrics['by_model'].keys())
        if models:
            best_model = max(models, key=lambda m: self.performance_metrics['by_model'][m]['direction_accuracy'])
            best_accuracy = self.performance_metrics['by_model'][best_model]['direction_accuracy'] * 100
            
            recommendations['overall_best'] = {
                'model': best_model,
                'accuracy': best_accuracy,
                'metrics': self.performance_metrics['by_model'][best_model]
            }
        
        # Find best model by regime
        if 'by_regime' in self.performance_metrics and self.performance_metrics['by_regime']:
            for regime in self.performance_metrics['by_regime']:
                valid_models = [m for m in models 
                             if m in self.performance_metrics['by_regime'][regime] 
                             and self.performance_metrics['by_regime'][regime][m]['count'] >= 5]
                
                if valid_models:
                    best_regime_model = max(valid_models, 
                                          key=lambda m: self.performance_metrics['by_regime'][regime][m]['direction_accuracy'])
                    best_regime_accuracy = self.performance_metrics['by_regime'][regime][best_regime_model]['direction_accuracy'] * 100
                    
                    recommendations['by_regime'][regime] = {
                        'model': best_regime_model,
                        'accuracy': best_regime_accuracy,
                        'metrics': self.performance_metrics['by_regime'][regime][best_regime_model]
                    }
        
        # Create explanation
        explanation_parts = []
        if recommendations['overall_best']:
            explanation_parts.append(
                f"The {recommendations['overall_best']['model'].replace('_', ' ').title()} model performed best overall "
                f"with {recommendations['overall_best']['accuracy']:.1f}% direction accuracy."
            )
        
        if recommendations['by_regime']:
            explanation_parts.append("Different models perform best in different market regimes:")
            for regime, rec in recommendations['by_regime'].items():
                explanation_parts.append(
                    f"• {regime}: {rec['model'].replace('_', ' ').title()} model "
                    f"({rec['accuracy']:.1f}% accuracy)"
                )
        
        explanation_parts.append(
            "For optimal results, consider using the regime-adjusted ensemble model, which "
            "automatically selects the best model weights based on current market conditions."
        )
        
        recommendations['explanation'] = "\n".join(explanation_parts)
        
        return recommendations


# Helper function to run backtests for multiple symbol/interval combinations
def run_all_backtests(symbols, intervals, days_back=90, step_size=1, include_regimes=True, include_sentiment=True):
    """
    Run backtests for all symbol/interval combinations
    
    Args:
        symbols: List of symbols to backtest
        intervals: List of intervals to backtest
        days_back: How many days to look back
        step_size: How many periods to step forward for each prediction
        include_regimes: Whether to include market regime detection
        include_sentiment: Whether to include sentiment analysis
        
    Returns:
        Dictionary with backtest results
    """
    results = {}
    
    for symbol in symbols:
        results[symbol] = {}
        
        for interval in intervals:
            logging.info(f"Running backtest for {symbol}/{interval}")
            
            try:
                # Initialize backtester
                backtester = MLBacktester(symbol, interval)
                
                # Run backtest
                backtest_results = backtester.run_backtest(
                    days_back=days_back,
                    step_size=step_size,
                    include_regimes=include_regimes,
                    include_sentiment=include_sentiment
                )
                
                if backtest_results:
                    # Get metrics
                    metrics = backtester.performance_metrics
                    results[symbol][interval] = {
                        'status': 'success',
                        'metrics': metrics
                    }
                else:
                    results[symbol][interval] = {
                        'status': 'failed',
                        'error': 'Backtest failed to generate results'
                    }
            except Exception as e:
                logging.error(f"Error running backtest for {symbol}/{interval}: {e}")
                results[symbol][interval] = {
                    'status': 'error',
                    'error': str(e)
                }
                
    return results