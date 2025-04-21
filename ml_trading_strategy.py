"""
ML-Based Trading Strategy Generator

This module provides functionality to:
1. Generate optimized trading strategies based on ML predictions
2. Customize strategies based on risk preferences and market regimes
3. Simulate and backtest trading strategies
4. Provide complete trading rules and parameters
"""

import os
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Local imports
from database import get_historical_data, get_indicators
from ml_prediction import MLPredictor
from ml_ensemble import EnsemblePredictor
from ml_market_regime import MarketRegimeDetector
from ml_sentiment_integration import SentimentIntegrator
from ml_backtesting import MLBacktester

class TradingStrategyGenerator:
    """
    Generates optimized trading strategies based on ML predictions
    """
    
    def __init__(self, symbol, interval):
        """
        Initialize the trading strategy generator
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Time interval (e.g., '1h', '4h', '1d')
        """
        self.symbol = symbol
        self.interval = interval
        
        # Initialize ML components
        self.predictor = MLPredictor(symbol, interval)
        self.ensemble_predictor = EnsemblePredictor(symbol, interval)
        self.regime_detector = MarketRegimeDetector(symbol, interval)
        self.sentiment_integrator = SentimentIntegrator(symbol, interval)
        self.backtester = MLBacktester(symbol, interval)
        
        # Define default strategy parameters
        self.default_parameters = {
            'entry_threshold': 0.02,       # Minimum predicted change to enter a trade (%)
            'confidence_threshold': 0.6,    # Minimum prediction confidence to enter a trade
            'stop_loss': 0.05,             # Default stop loss (%)
            'take_profit': 0.1,            # Default take profit (%)
            'max_position_size': 0.2,      # Maximum position size as % of portfolio
            'trailing_stop': False,         # Whether to use trailing stop
            'trailing_distance': 0.02,      # Trailing stop distance (%)
            'use_market_regime': True,      # Whether to adjust strategy based on market regime
            'use_sentiment': True,          # Whether to incorporate sentiment
            'exit_on_opposite_signal': True # Exit position on opposite signal
        }
        
        # Results storage
        self.optimized_parameters = None
        self.strategy_performance = None
        
        # Result paths
        self.parameters_path = os.path.join('models', f"{symbol}_{interval}_strategy_parameters.json")
        self.performance_path = os.path.join('models', f"{symbol}_{interval}_strategy_performance.json")
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Load optimized parameters if available
        self.load_parameters()
    
    def generate_strategy(self, risk_level='medium', optimize=True, days_back=90):
        """
        Generate a trading strategy with optimized parameters
        
        Args:
            risk_level: Risk level of the strategy ('low', 'medium', 'high')
            optimize: Whether to optimize parameters with backtesting
            days_back: How many days of historical data to use for optimization
            
        Returns:
            Dictionary with strategy parameters and rules
        """
        # Set base parameters based on risk level
        base_parameters = self._get_base_parameters(risk_level)
        
        # Get current market regime if available
        current_regime = None
        if base_parameters['use_market_regime']:
            try:
                regime_info = self.regime_detector.detect_current_regime()
                if regime_info and 'current_regime' in regime_info:
                    current_regime = regime_info['current_regime']['name']
                    # Adjust parameters based on regime
                    base_parameters = self._adjust_for_regime(base_parameters, current_regime)
            except Exception as e:
                logging.warning(f"Could not detect market regime: {e}")
        
        # Optimize parameters if requested
        if optimize:
            try:
                if not self.backtester.backtest_results:
                    # Run backtest if not already done
                    self.backtester.run_backtest(
                        days_back=days_back,
                        include_regimes=base_parameters['use_market_regime'],
                        include_sentiment=base_parameters['use_sentiment']
                    )
                
                # Optimize parameters
                optimized_params = self._optimize_parameters(base_parameters, current_regime)
                if optimized_params:
                    self.optimized_parameters = optimized_params
                    self._save_parameters()
            except Exception as e:
                logging.error(f"Error optimizing parameters: {e}")
                # Fall back to base parameters
                self.optimized_parameters = base_parameters
        else:
            # Use base parameters
            self.optimized_parameters = base_parameters
        
        # Generate strategy rules
        strategy_rules = self._generate_strategy_rules(self.optimized_parameters, current_regime)
        
        # Combine parameters and rules into complete strategy
        strategy = {
            'symbol': self.symbol,
            'interval': self.interval,
            'risk_level': risk_level,
            'parameters': self.optimized_parameters,
            'rules': strategy_rules,
            'market_regime': current_regime,
            'timestamp': datetime.now().isoformat()
        }
        
        return strategy
    
    def _get_base_parameters(self, risk_level):
        """
        Get base strategy parameters based on risk level
        
        Args:
            risk_level: Risk level of the strategy ('low', 'medium', 'high')
            
        Returns:
            Dictionary with strategy parameters
        """
        # Start with default parameters
        params = self.default_parameters.copy()
        
        # Adjust based on risk level
        if risk_level == 'low':
            params.update({
                'entry_threshold': 0.015,    # Lower threshold to find more opportunities
                'confidence_threshold': 0.7,  # Higher confidence requirement
                'stop_loss': 0.03,           # Tighter stop loss
                'take_profit': 0.05,         # Lower profit target
                'max_position_size': 0.1,    # Smaller position size
                'trailing_stop': True,        # Use trailing stop for risk management
                'trailing_distance': 0.01     # Tight trailing stop
            })
        elif risk_level == 'medium':
            # Already set to medium in defaults
            pass
        elif risk_level == 'high':
            params.update({
                'entry_threshold': 0.03,      # Higher threshold for stronger moves
                'confidence_threshold': 0.5,   # Lower confidence requirement
                'stop_loss': 0.08,            # Wider stop loss
                'take_profit': 0.15,          # Higher profit target
                'max_position_size': 0.3,     # Larger position size
                'trailing_stop': False,        # Don't use trailing stop
                'trailing_distance': 0.03      # Wider trailing distance if used
            })
        
        return params
    
    def _adjust_for_regime(self, parameters, regime):
        """
        Adjust strategy parameters based on market regime
        
        Args:
            parameters: Base strategy parameters
            regime: Current market regime
            
        Returns:
            Dictionary with adjusted parameters
        """
        # Make a copy of parameters to avoid modifying the original
        params = parameters.copy()
        
        # Adjust based on regime
        if regime == "RANGING":
            # In ranging markets, tighten up parameters
            params.update({
                'entry_threshold': max(0.01, params['entry_threshold'] * 0.8),  # Lower threshold
                'take_profit': params['take_profit'] * 0.7,  # Reduce profit targets
                'stop_loss': params['stop_loss'] * 0.8,  # Tighter stops
                'trailing_stop': True,  # Use trailing stops in ranging markets
                'exit_on_opposite_signal': True  # Exit quickly on opposite signals
            })
        elif regime == "TRENDING_UP":
            # In uptrends, focus on capturing the trend
            params.update({
                'entry_threshold': params['entry_threshold'] * 0.9,  # Slightly lower threshold
                'take_profit': params['take_profit'] * 1.5,  # Higher profit targets
                'stop_loss': params['stop_loss'] * 1.2,  # Wider stops to avoid getting shaken out
                'trailing_stop': True,  # Use trailing stops to ride the trend
                'trailing_distance': params['trailing_distance'] * 1.5  # Wider trailing distance
            })
        elif regime == "TRENDING_DOWN":
            # In downtrends, be more conservative with longs
            params.update({
                'entry_threshold': params['entry_threshold'] * 1.2,  # Higher threshold for long entries
                'confidence_threshold': params['confidence_threshold'] * 1.1,  # Higher confidence requirement
                'take_profit': params['take_profit'] * 0.8,  # Lower profit targets
                'stop_loss': params['stop_loss'] * 0.9,  # Tighter stops
                'max_position_size': params['max_position_size'] * 0.8  # Smaller position size
            })
        elif regime == "VOLATILE":
            # In volatile markets, be more cautious
            params.update({
                'entry_threshold': params['entry_threshold'] * 1.5,  # Much higher threshold
                'confidence_threshold': params['confidence_threshold'] * 1.2,  # Higher confidence requirement
                'take_profit': params['take_profit'] * 1.2,  # Higher profit targets
                'stop_loss': params['stop_loss'] * 1.5,  # Wider stops to handle volatility
                'max_position_size': params['max_position_size'] * 0.6,  # Much smaller position size
                'trailing_stop': True,  # Use trailing stops
                'trailing_distance': params['trailing_distance'] * 2  # Much wider trailing distance
            })
        
        # Ensure parameters are within reasonable bounds
        params['entry_threshold'] = max(0.005, min(0.1, params['entry_threshold']))
        params['confidence_threshold'] = max(0.3, min(0.9, params['confidence_threshold']))
        params['stop_loss'] = max(0.01, min(0.15, params['stop_loss']))
        params['take_profit'] = max(0.02, min(0.3, params['take_profit']))
        params['max_position_size'] = max(0.05, min(0.5, params['max_position_size']))
        params['trailing_distance'] = max(0.005, min(0.1, params['trailing_distance']))
        
        return params
    
    def _optimize_parameters(self, base_parameters, current_regime=None):
        """
        Optimize strategy parameters based on backtest results
        
        Args:
            base_parameters: Base strategy parameters
            current_regime: Current market regime (if known)
            
        Returns:
            Dictionary with optimized parameters
        """
        if not self.backtester.backtest_results:
            logging.warning("No backtest results available for parameter optimization")
            return base_parameters
        
        # Make a copy of parameters to optimize
        params = base_parameters.copy()
        
        # If we have a current regime and backtest metrics by regime, optimize for that regime
        regime_specific = False
        if (current_regime and 
            hasattr(self.backtester, 'performance_metrics') and 
            'by_regime' in self.backtester.performance_metrics and 
            current_regime in self.backtester.performance_metrics['by_regime']):
            
            # We have regime-specific metrics, optimize for this regime
            regime_specific = True
            
            # Find best model for this regime
            models = list(self.backtester.performance_metrics['by_regime'][current_regime].keys())
            
            if models:
                best_model = max(models, 
                               key=lambda m: self.backtester.performance_metrics['by_regime'][current_regime][m]['direction_accuracy'])
                best_accuracy = self.backtester.performance_metrics['by_regime'][current_regime][best_model]['direction_accuracy']
                
                logging.info(f"Best model for {current_regime} regime: {best_model} ({best_accuracy*100:.1f}% accuracy)")
                
                # Adjust confidence threshold based on model accuracy
                if best_accuracy > 0.7:
                    # High accuracy model, lower the confidence threshold
                    params['confidence_threshold'] = max(0.5, params['confidence_threshold'] * 0.9)
                elif best_accuracy < 0.6:
                    # Low accuracy model, increase the confidence threshold
                    params['confidence_threshold'] = min(0.8, params['confidence_threshold'] * 1.2)
        
        # If not optimizing for a specific regime, use overall metrics
        if not regime_specific:
            # Find best model overall
            models = list(self.backtester.performance_metrics['by_model'].keys())
            
            if models:
                best_model = max(models, 
                               key=lambda m: self.backtester.performance_metrics['by_model'][m]['direction_accuracy'])
                best_accuracy = self.backtester.performance_metrics['by_model'][best_model]['direction_accuracy']
                
                logging.info(f"Best overall model: {best_model} ({best_accuracy*100:.1f}% accuracy)")
                
                # Adjust confidence threshold based on model accuracy
                if best_accuracy > 0.7:
                    # High accuracy model, lower the confidence threshold
                    params['confidence_threshold'] = max(0.5, params['confidence_threshold'] * 0.9)
                elif best_accuracy < 0.6:
                    # Low accuracy model, increase the confidence threshold
                    params['confidence_threshold'] = min(0.8, params['confidence_threshold'] * 1.2)
        
        # Convert backtest results to DataFrame for analysis
        results_df = pd.DataFrame([
            {
                'timestamp': pd.to_datetime(r['timestamp']),
                'current_price': r['current_price'],
                'actual_change': r['actual_change'],
                'regime': r.get('regime', 'Unknown'),
                'sentiment': r.get('sentiment', 0),
                **{f"{model}_predicted_change": r['models'][model]['predicted_change'] 
                   for model in r['models']},
                **{f"{model}_confidence": r['models'][model]['confidence'] 
                   for model in r['models']},
                **{f"{model}_direction_correct": r['models'][model]['direction_correct'] 
                   for model in r['models']}
            }
            for r in self.backtester.backtest_results if r['models']
        ])
        
        # Analyze prediction error distribution to adjust entry threshold
        if not results_df.empty:
            # Use the best model for adjustments
            if 'ensemble_predicted_change' in results_df.columns:
                best_model_col = 'ensemble_predicted_change'
            elif 'regime_adjusted_predicted_change' in results_df.columns:
                best_model_col = 'regime_adjusted_predicted_change'
            elif 'sentiment_adjusted_predicted_change' in results_df.columns:
                best_model_col = 'sentiment_adjusted_predicted_change'
            elif 'base_predicted_change' in results_df.columns:
                best_model_col = 'base_predicted_change'
            else:
                best_model_col = None
            
            if best_model_col:
                # Calculate the median of absolute prediction changes
                median_abs_change = results_df[best_model_col].abs().median()
                
                # Adjust entry threshold based on median prediction magnitude
                if median_abs_change > 0:
                    # Set entry threshold to slightly lower than median change
                    adjusted_threshold = median_abs_change * 0.8
                    # Ensure it's within reasonable bounds
                    params['entry_threshold'] = max(0.005, min(0.1, adjusted_threshold))
                    
                    logging.info(f"Adjusted entry threshold to {params['entry_threshold']:.4f} based on median prediction magnitude")
        
        # Optimize stop loss and take profit based on typical price movements
        if not results_df.empty:
            # Calculate typical daily volatility (standard deviation of daily % changes)
            daily_changes = results_df['actual_change'].abs()
            
            if not daily_changes.empty:
                # 25th, 50th (median), and 75th percentiles of daily changes
                q25, median, q75 = np.percentile(daily_changes, [25, 50, 75])
                
                # Set stop loss based on 75th percentile of daily changes
                # Multiply by a factor to give some room (avoid being stopped out too easily)
                adjusted_stop = q75 * 1.5
                
                # Set take profit as a multiple of the stop loss
                adjusted_take_profit = adjusted_stop * 2
                
                # Ensure they're within reasonable bounds
                params['stop_loss'] = max(0.01, min(0.15, adjusted_stop))
                params['take_profit'] = max(0.02, min(0.3, adjusted_take_profit))
                
                logging.info(f"Adjusted stop loss to {params['stop_loss']:.4f} and take profit to {params['take_profit']:.4f}")
        
        return params
    
    def _generate_strategy_rules(self, parameters, current_regime=None):
        """
        Generate trading rules based on strategy parameters
        
        Args:
            parameters: Strategy parameters
            current_regime: Current market regime (if known)
            
        Returns:
            Dictionary with trading rules
        """
        # Determine which model to use for predictions
        if hasattr(self.backtester, 'performance_metrics'):
            if (current_regime and 
                'by_regime' in self.backtester.performance_metrics and 
                current_regime in self.backtester.performance_metrics['by_regime']):
                
                # Find best model for this regime
                models = list(self.backtester.performance_metrics['by_regime'][current_regime].keys())
                best_model = max(models, 
                               key=lambda m: self.backtester.performance_metrics['by_regime'][current_regime][m]['direction_accuracy'])
            else:
                # Find best model overall
                models = list(self.backtester.performance_metrics['by_model'].keys())
                best_model = max(models, 
                               key=lambda m: self.backtester.performance_metrics['by_model'][m]['direction_accuracy'])
        else:
            # Default to ensemble model if available
            if os.path.exists(self.ensemble_predictor.model_path):
                best_model = 'ensemble'
            else:
                best_model = 'base'
        
        # Generate entry rules
        entry_rules = [
            f"Use the {best_model.replace('_', ' ').title()} model for predictions",
            f"Enter LONG when predicted price change > {parameters['entry_threshold']*100:.1f}% with confidence > {parameters['confidence_threshold']*100:.1f}%",
            f"Enter SHORT when predicted price change < -{parameters['entry_threshold']*100:.1f}% with confidence > {parameters['confidence_threshold']*100:.1f}%"
        ]
        
        # Add regime-specific rules if applicable
        if parameters['use_market_regime'] and current_regime:
            if current_regime == "RANGING":
                entry_rules.append("In ranging markets, look for reversals at support and resistance levels")
                entry_rules.append("Consider using smaller position sizes in ranging markets")
            elif current_regime == "TRENDING_UP":
                entry_rules.append("In uptrend markets, look for pullbacks as entry opportunities for long positions")
                entry_rules.append("Avoid shorts against the primary uptrend unless the prediction is very strong")
            elif current_regime == "TRENDING_DOWN":
                entry_rules.append("In downtrend markets, look for bounces as entry opportunities for short positions")
                entry_rules.append("Be more conservative with long positions against the primary downtrend")
            elif current_regime == "VOLATILE":
                entry_rules.append("In volatile markets, require higher prediction confidence and use smaller position sizes")
                entry_rules.append("Wait for volatility to settle before entering new positions")
        
        # Generate position sizing rules
        position_rules = [
            f"Maximum position size: {parameters['max_position_size']*100:.1f}% of available capital per trade",
            "Scale position size based on prediction confidence and market volatility",
            "Reduce position size during high market uncertainty or decreased prediction confidence"
        ]
        
        # Generate exit rules
        exit_rules = [
            f"Set stop loss at {parameters['stop_loss']*100:.1f}% from entry price",
            f"Set take profit at {parameters['take_profit']*100:.1f}% from entry price"
        ]
        
        if parameters['trailing_stop']:
            exit_rules.append(f"Use trailing stop loss at {parameters['trailing_distance']*100:.1f}% below highest price since entry (for longs)")
            exit_rules.append(f"Use trailing stop loss at {parameters['trailing_distance']*100:.1f}% above lowest price since entry (for shorts)")
        
        if parameters['exit_on_opposite_signal']:
            exit_rules.append("Exit position when model predicts movement in opposite direction to current position")
        
        if parameters['use_sentiment']:
            exit_rules.append("Consider exiting when sentiment shifts significantly against position direction")
        
        # Generate risk management rules
        risk_rules = [
            "Never risk more than 2% of total portfolio on a single trade",
            "Avoid multiple correlated positions that increase overall risk exposure",
            "Reduce position sizes during high market stress or volatility",
            "Have maximum drawdown limits for daily/weekly/monthly periods"
        ]
        
        # Generate combined trading plan
        trading_plan = "\n".join([
            f"## Trading Strategy for {self.symbol}/{self.interval}",
            f"Current Market Regime: {current_regime}" if current_regime else "",
            "",
            "### Entry Rules",
            "\n".join(f"* {rule}" for rule in entry_rules),
            "",
            "### Position Sizing",
            "\n".join(f"* {rule}" for rule in position_rules),
            "",
            "### Exit Rules",
            "\n".join(f"* {rule}" for rule in exit_rules),
            "",
            "### Risk Management",
            "\n".join(f"* {rule}" for rule in risk_rules)
        ])
        
        return {
            'best_model': best_model,
            'entry_rules': entry_rules,
            'position_rules': position_rules,
            'exit_rules': exit_rules,
            'risk_rules': risk_rules,
            'trading_plan': trading_plan
        }
    
    def simulate_strategy(self, start_date=None, end_date=None, days_back=30, initial_capital=10000):
        """
        Simulate the trading strategy on historical data
        
        Args:
            start_date: Start date for simulation
            end_date: End date for simulation
            days_back: How many days to simulate if start_date not provided
            initial_capital: Initial capital for simulation
            
        Returns:
            Dictionary with simulation results
        """
        # Load strategy parameters if not already set
        if self.optimized_parameters is None:
            self.load_parameters()
            if self.optimized_parameters is None:
                # Generate default parameters if none available
                self.optimized_parameters = self._get_base_parameters('medium')
        
        # Set date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=days_back)
        
        # Load backtest results from the same period if available
        use_backtest_predictions = False
        if hasattr(self.backtester, 'backtest_results') and self.backtester.backtest_results:
            # Convert timestamps to datetime
            for result in self.backtester.backtest_results:
                if isinstance(result['timestamp'], str):
                    result['timestamp'] = pd.to_datetime(result['timestamp'])
            
            # Filter backtest results to simulation period
            filtered_results = [
                r for r in self.backtester.backtest_results 
                if start_date <= r['timestamp'] <= end_date
            ]
            
            if len(filtered_results) > 0:
                use_backtest_predictions = True
                backtest_results = filtered_results
                logging.info(f"Using {len(backtest_results)} backtest predictions for simulation")
            else:
                # Run a new backtest
                logging.info("Running new backtest for simulation period")
                self.backtester.run_backtest(
                    start_date=start_date,
                    end_date=end_date,
                    include_regimes=self.optimized_parameters['use_market_regime'],
                    include_sentiment=self.optimized_parameters['use_sentiment']
                )
                backtest_results = self.backtester.backtest_results
                use_backtest_predictions = True
        
        # If we don't have backtest results, use historical data directly
        if not use_backtest_predictions:
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
        
        # Initialize simulation state
        simulation = {
            'initial_capital': initial_capital,
            'current_capital': initial_capital,
            'position': None,  # None, 'long', or 'short'
            'position_size': 0,
            'entry_price': 0,
            'entry_time': None,
            'highest_price': 0,  # For trailing stops on longs
            'lowest_price': float('inf'),  # For trailing stops on shorts
            'trades': [],
            'equity_curve': [],
            'parameters': self.optimized_parameters
        }
        
        # Run simulation
        if use_backtest_predictions:
            self._simulate_with_backtest(simulation, backtest_results)
        else:
            self._simulate_with_historical_data(simulation, df)
        
        # Calculate performance metrics
        metrics = self._calculate_simulation_metrics(simulation)
        simulation['metrics'] = metrics
        
        # Save performance
        self.strategy_performance = simulation
        self._save_performance()
        
        return simulation
    
    def _simulate_with_backtest(self, simulation, backtest_results):
        """
        Simulate strategy using backtest prediction results
        
        Args:
            simulation: Simulation state
            backtest_results: List of backtest result dictionaries
        """
        # Sort backtest results by timestamp
        sorted_results = sorted(backtest_results, key=lambda x: x['timestamp'])
        
        # Process each prediction
        for i, result in enumerate(sorted_results):
            timestamp = result['timestamp']
            current_price = result['current_price']
            
            # Find the best model to use
            best_model = self._get_best_model_from_backtest(result)
            
            if best_model in result['models']:
                prediction = result['models'][best_model]
                predicted_change = prediction['predicted_change']
                confidence = prediction['confidence']
                
                # Get market regime if available
                current_regime = result.get('regime', None)
                
                # Update simulation state with current price
                self._update_simulation_state(simulation, timestamp, current_price, current_regime)
                
                # Check for trade exit
                if simulation['position'] is not None:
                    exit_price = self._check_exit_conditions(
                        simulation, 
                        current_price, 
                        predicted_change, 
                        confidence
                    )
                    
                    if exit_price is not None:
                        self._exit_position(simulation, timestamp, exit_price)
                
                # Check for trade entry
                if simulation['position'] is None:
                    entry_signal = self._check_entry_conditions(
                        predicted_change, 
                        confidence, 
                        current_regime
                    )
                    
                    if entry_signal is not None:
                        self._enter_position(
                            simulation, 
                            timestamp, 
                            current_price, 
                            entry_signal
                        )
            
            # Record equity at this point
            self._record_equity(simulation, timestamp, current_price)
    
    def _simulate_with_historical_data(self, simulation, df):
        """
        Simulate strategy using historical data directly
        
        Args:
            simulation: Simulation state
            df: DataFrame with historical data
        """
        # Ensure dataframe is sorted by timestamp
        df = df.sort_values('timestamp')
        
        # Process each row
        for i, row in df.iterrows():
            timestamp = row['timestamp']
            current_price = row['close']
            
            # Skip if not enough data for prediction
            if i < 30:  # Need enough data for features
                self._record_equity(simulation, timestamp, current_price)
                continue
            
            # Get prediction for this point
            try:
                # Get regime if available
                current_regime = None
                if self.optimized_parameters['use_market_regime']:
                    try:
                        regime_data = df.iloc[:i+1]
                        regime_features = self.regime_detector.prepare_regime_features(regime_data)
                        
                        if not regime_features.empty:
                            # Extract most recent features
                            recent_features = regime_features.drop('timestamp', axis=1).iloc[-1:]
                            
                            # Scale features
                            recent_features_scaled = self.regime_detector.scaler.transform(recent_features)
                            
                            # Predict regime
                            regime_label = self.regime_detector.kmeans_model.predict(recent_features_scaled)[0]
                            
                            # Map to regime name
                            if (hasattr(self.regime_detector, 'metadata') and 
                                'regime_mapping' in self.regime_detector.metadata):
                                current_regime = self.regime_detector.metadata['regime_mapping'].get(
                                    str(regime_label), self.regime_detector.REGIME_TYPES.get(0, "UNKNOWN")
                                )
                            else:
                                current_regime = self.regime_detector.REGIME_TYPES.get(regime_label, "UNKNOWN")
                    except Exception as e:
                        logging.warning(f"Could not detect regime: {e}")
                
                # Prepare data for prediction
                predict_data = df.iloc[:i+1]
                
                # Use ensemble predictor if available
                if os.path.exists(self.ensemble_predictor.model_path):
                    X, _ = self.ensemble_predictor.prepare_data(predict_data)
                    
                    if X is not None and not X.empty:
                        # Scale features
                        X_scaled = self.ensemble_predictor.scaler.transform(X.iloc[-1:])
                        
                        # Make prediction
                        predicted_change = float(self.ensemble_predictor.model.predict(X_scaled)[0])
                        
                        # Calculate confidence
                        confidence = self.ensemble_predictor._get_prediction_confidence(predicted_change)
                    else:
                        continue  # Skip if features not available
                else:
                    # Fall back to base predictor
                    X, _ = self.predictor.prepare_data(predict_data)
                    
                    if X is not None and not X.empty:
                        # Scale features
                        X_scaled = self.predictor.scaler.transform(X.iloc[-1:])
                        
                        # Make prediction
                        predicted_change = float(self.predictor.model.predict(X_scaled)[0])
                        
                        # Calculate confidence
                        confidence = self.predictor._get_prediction_confidence(predicted_change)
                    else:
                        continue  # Skip if features not available
                
                # Update simulation state with current price
                self._update_simulation_state(simulation, timestamp, current_price, current_regime)
                
                # Check for trade exit
                if simulation['position'] is not None:
                    exit_price = self._check_exit_conditions(
                        simulation, 
                        current_price, 
                        predicted_change, 
                        confidence
                    )
                    
                    if exit_price is not None:
                        self._exit_position(simulation, timestamp, exit_price)
                
                # Check for trade entry
                if simulation['position'] is None:
                    entry_signal = self._check_entry_conditions(
                        predicted_change, 
                        confidence, 
                        current_regime
                    )
                    
                    if entry_signal is not None:
                        self._enter_position(
                            simulation, 
                            timestamp, 
                            current_price, 
                            entry_signal
                        )
            except Exception as e:
                logging.warning(f"Error in simulation at {timestamp}: {e}")
            
            # Record equity at this point
            self._record_equity(simulation, timestamp, current_price)
    
    def _get_best_model_from_backtest(self, result):
        """
        Determine best model to use from backtest result
        
        Args:
            result: Single backtest result dictionary
            
        Returns:
            Name of best model to use
        """
        # Check if we have regime-adjusted predictions
        if 'regime_adjusted' in result['models'] and self.optimized_parameters['use_market_regime']:
            return 'regime_adjusted'
        
        # Check if we have sentiment-adjusted predictions
        if 'sentiment_adjusted' in result['models'] and self.optimized_parameters['use_sentiment']:
            return 'sentiment_adjusted'
        
        # Check if we have ensemble predictions
        if 'ensemble' in result['models']:
            return 'ensemble'
        
        # Fall back to base model
        if 'base' in result['models']:
            return 'base'
        
        # Get first available model
        return list(result['models'].keys())[0]
    
    def _update_simulation_state(self, simulation, timestamp, current_price, current_regime=None):
        """
        Update simulation state with current price
        
        Args:
            simulation: Simulation state dictionary
            timestamp: Current timestamp
            current_price: Current price
            current_regime: Current market regime (if known)
        """
        # Update highest and lowest prices for trailing stops
        if simulation['position'] == 'long' and current_price > simulation['highest_price']:
            simulation['highest_price'] = current_price
        elif simulation['position'] == 'short' and current_price < simulation['lowest_price']:
            simulation['lowest_price'] = current_price
    
    def _check_exit_conditions(self, simulation, current_price, predicted_change, confidence):
        """
        Check if we should exit current position
        
        Args:
            simulation: Simulation state dictionary
            current_price: Current price
            predicted_change: Predicted price change
            confidence: Prediction confidence
            
        Returns:
            Exit price if should exit, None otherwise
        """
        if simulation['position'] is None:
            return None
        
        entry_price = simulation['entry_price']
        params = simulation['parameters']
        
        # Initialize exit price to current price (default)
        exit_price = current_price
        
        # Check stop loss
        if simulation['position'] == 'long':
            stop_price = entry_price * (1 - params['stop_loss'])
            
            if params['trailing_stop'] and simulation['highest_price'] > entry_price:
                # Trailing stop for longs
                trail_stop = simulation['highest_price'] * (1 - params['trailing_distance'])
                # Use the higher of the fixed stop and trailing stop
                stop_price = max(stop_price, trail_stop)
            
            if current_price <= stop_price:
                return stop_price  # Stop loss triggered
            
            # Check take profit
            take_profit_price = entry_price * (1 + params['take_profit'])
            if current_price >= take_profit_price:
                return take_profit_price  # Take profit triggered
            
            # Check for opposite signal
            if (params['exit_on_opposite_signal'] and 
                predicted_change < -params['entry_threshold'] and 
                confidence >= params['confidence_threshold']):
                return current_price  # Exit on opposite signal
        
        elif simulation['position'] == 'short':
            stop_price = entry_price * (1 + params['stop_loss'])
            
            if params['trailing_stop'] and simulation['lowest_price'] < entry_price:
                # Trailing stop for shorts
                trail_stop = simulation['lowest_price'] * (1 + params['trailing_distance'])
                # Use the lower of the fixed stop and trailing stop
                stop_price = min(stop_price, trail_stop)
            
            if current_price >= stop_price:
                return stop_price  # Stop loss triggered
            
            # Check take profit
            take_profit_price = entry_price * (1 - params['take_profit'])
            if current_price <= take_profit_price:
                return take_profit_price  # Take profit triggered
            
            # Check for opposite signal
            if (params['exit_on_opposite_signal'] and 
                predicted_change > params['entry_threshold'] and 
                confidence >= params['confidence_threshold']):
                return current_price  # Exit on opposite signal
        
        return None  # Don't exit
    
    def _check_entry_conditions(self, predicted_change, confidence, current_regime=None):
        """
        Check if we should enter a new position
        
        Args:
            predicted_change: Predicted price change
            confidence: Prediction confidence
            current_regime: Current market regime (if known)
            
        Returns:
            'long', 'short', or None
        """
        params = self.optimized_parameters
        
        # Adjust entry threshold based on regime if applicable
        entry_threshold = params['entry_threshold']
        confidence_threshold = params['confidence_threshold']
        
        if params['use_market_regime'] and current_regime:
            # Apply regime-specific adjustments
            if current_regime == "RANGING":
                # Require stronger signals in ranging markets
                entry_threshold *= 1.2
            elif current_regime == "TRENDING_UP":
                # More lenient for longs in uptrend
                if predicted_change > 0:
                    entry_threshold *= 0.8
                else:
                    # More strict for shorts in uptrend
                    entry_threshold *= 1.5
                    confidence_threshold *= 1.2
            elif current_regime == "TRENDING_DOWN":
                # More lenient for shorts in downtrend
                if predicted_change < 0:
                    entry_threshold *= 0.8
                else:
                    # More strict for longs in downtrend
                    entry_threshold *= 1.5
                    confidence_threshold *= 1.2
            elif current_regime == "VOLATILE":
                # Require stronger signals in volatile markets
                entry_threshold *= 1.5
                confidence_threshold *= 1.2
        
        # Check entry conditions
        if predicted_change > entry_threshold and confidence >= confidence_threshold:
            return 'long'
        elif predicted_change < -entry_threshold and confidence >= confidence_threshold:
            return 'short'
        
        return None  # No entry signal
    
    def _enter_position(self, simulation, timestamp, current_price, position_type):
        """
        Enter a new position
        
        Args:
            simulation: Simulation state dictionary
            timestamp: Current timestamp
            current_price: Current price
            position_type: 'long' or 'short'
        """
        params = simulation['parameters']
        
        # Calculate position size
        available_capital = simulation['current_capital']
        position_size = available_capital * params['max_position_size']
        
        # Record entry
        simulation['position'] = position_type
        simulation['entry_price'] = current_price
        simulation['entry_time'] = timestamp
        simulation['position_size'] = position_size
        
        # Reset highest/lowest price trackers
        simulation['highest_price'] = current_price
        simulation['lowest_price'] = current_price
        
        # Log the trade
        logging.info(f"Entered {position_type} position at {timestamp}: price={current_price}, size=${position_size:.2f}")
    
    def _exit_position(self, simulation, timestamp, exit_price):
        """
        Exit current position
        
        Args:
            simulation: Simulation state dictionary
            timestamp: Current timestamp
            exit_price: Exit price
        """
        # Calculate profit/loss
        position_type = simulation['position']
        entry_price = simulation['entry_price']
        position_size = simulation['position_size']
        
        if position_type == 'long':
            price_change = (exit_price / entry_price) - 1
        else:  # short
            price_change = 1 - (exit_price / entry_price)
        
        profit_loss = position_size * price_change
        
        # Update capital
        simulation['current_capital'] += profit_loss
        
        # Record the trade
        trade = {
            'position_type': position_type,
            'entry_time': simulation['entry_time'],
            'entry_price': entry_price,
            'exit_time': timestamp,
            'exit_price': exit_price,
            'position_size': position_size,
            'profit_loss': profit_loss,
            'profit_loss_pct': price_change * 100,
            'duration': (timestamp - simulation['entry_time']).total_seconds() / 3600  # hours
        }
        
        simulation['trades'].append(trade)
        
        # Reset position
        simulation['position'] = None
        simulation['position_size'] = 0
        simulation['entry_price'] = 0
        simulation['entry_time'] = None
        simulation['highest_price'] = 0
        simulation['lowest_price'] = float('inf')
        
        # Log the trade
        logging.info(
            f"Exited {position_type} position at {timestamp}: entry=${entry_price:.2f}, "
            f"exit=${exit_price:.2f}, P/L=${profit_loss:.2f} ({price_change*100:.2f}%)"
        )
    
    def _record_equity(self, simulation, timestamp, current_price):
        """
        Record equity value at this point in time
        
        Args:
            simulation: Simulation state dictionary
            timestamp: Current timestamp
            current_price: Current price
        """
        # Calculate current equity value
        equity = simulation['current_capital']
        
        # Add value of open position if any
        if simulation['position'] is not None:
            entry_price = simulation['entry_price']
            position_size = simulation['position_size']
            
            if simulation['position'] == 'long':
                price_change = (current_price / entry_price) - 1
            else:  # short
                price_change = 1 - (current_price / entry_price)
            
            position_value = position_size * (1 + price_change)
            equity += position_value - position_size  # Add only the profit/loss
        
        # Record equity
        simulation['equity_curve'].append({
            'timestamp': timestamp,
            'equity': equity,
            'price': current_price
        })
    
    def _calculate_simulation_metrics(self, simulation):
        """
        Calculate performance metrics for simulation
        
        Args:
            simulation: Simulation state dictionary
            
        Returns:
            Dictionary with performance metrics
        """
        trades = simulation['trades']
        equity_curve = simulation['equity_curve']
        
        if not trades or not equity_curve:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return': 0,
                'max_drawdown': 0
            }
        
        # Calculate trade metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['profit_loss'] > 0)
        losing_trades = sum(1 for t in trades if t['profit_loss'] <= 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t['profit_loss'] for t in trades if t['profit_loss'] > 0)
        total_loss = sum(abs(t['profit_loss']) for t in trades if t['profit_loss'] < 0)
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate returns
        initial_equity = simulation['initial_capital']
        final_equity = equity_curve[-1]['equity']
        
        total_return = (final_equity / initial_equity) - 1
        
        # Calculate drawdown
        max_equity = initial_equity
        max_drawdown = 0
        
        for point in equity_curve:
            equity = point['equity']
            if equity > max_equity:
                max_equity = equity
            
            drawdown = (max_equity - equity) / max_equity
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio (simple approximation)
        equity_values = [p['equity'] for p in equity_curve]
        returns = []
        
        for i in range(1, len(equity_values)):
            returns.append((equity_values[i] / equity_values[i-1]) - 1)
        
        avg_return = np.mean(returns) if returns else 0
        std_return = np.std(returns) if len(returns) > 1 else 0
        
        # Assuming 252 trading days per year and 0% risk-free rate
        sharpe_ratio = (avg_return * 252**0.5) / std_return if std_return > 0 else 0
        
        # Calculate average metrics
        avg_profit = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        
        expectancy = (win_rate * avg_profit) - ((1 - win_rate) * avg_loss)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'expectancy': expectancy,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'initial_capital': initial_equity,
            'final_capital': final_equity
        }
    
    def load_parameters(self):
        """
        Load strategy parameters from disk
        
        Returns:
            Boolean indicating success
        """
        if os.path.exists(self.parameters_path):
            try:
                with open(self.parameters_path, 'r') as f:
                    data = json.load(f)
                
                self.optimized_parameters = data['parameters']
                return True
            except Exception as e:
                logging.error(f"Error loading strategy parameters: {e}")
                return False
        else:
            logging.warning(f"No saved strategy parameters found for {self.symbol}/{self.interval}")
            return False
    
    def _save_parameters(self):
        """
        Save strategy parameters to disk
        
        Returns:
            Boolean indicating success
        """
        if self.optimized_parameters is None:
            return False
        
        try:
            data = {
                'symbol': self.symbol,
                'interval': self.interval,
                'parameters': self.optimized_parameters,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.parameters_path, 'w') as f:
                json.dump(data, f)
            
            return True
        except Exception as e:
            logging.error(f"Error saving strategy parameters: {e}")
            return False
    
    def _save_performance(self):
        """
        Save strategy performance to disk
        
        Returns:
            Boolean indicating success
        """
        if self.strategy_performance is None:
            return False
        
        try:
            # Create a copy of the performance to modify
            performance = dict(self.strategy_performance)
            
            # Convert timestamp objects to ISO format strings for JSON serialization
            for trade in performance['trades']:
                trade['entry_time'] = trade['entry_time'].isoformat()
                trade['exit_time'] = trade['exit_time'].isoformat()
            
            for point in performance['equity_curve']:
                point['timestamp'] = point['timestamp'].isoformat()
            
            data = {
                'symbol': self.symbol,
                'interval': self.interval,
                'performance': performance,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.performance_path, 'w') as f:
                json.dump(data, f)
            
            return True
        except Exception as e:
            logging.error(f"Error saving strategy performance: {e}")
            return False
    
    def create_strategy_visualizations(self, render_to=None):
        """
        Create visualizations of strategy performance
        
        Args:
            render_to: Streamlit container to render plots in (optional)
            
        Returns:
            Dictionary with plotly figures for each visualization
        """
        if self.strategy_performance is None:
            self.simulate_strategy()  # Run simulation if not done yet
        
        if self.strategy_performance is None:
            logging.error("No strategy performance data available for visualization")
            return None
        
        figures = {}
        
        # Create equity curve
        equity_df = pd.DataFrame([
            {
                'timestamp': point['timestamp'] if isinstance(point['timestamp'], datetime) else pd.to_datetime(point['timestamp']),
                'equity': point['equity'],
                'price': point['price']
            }
            for point in self.strategy_performance['equity_curve']
        ])
        
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add equity curve
        fig1.add_trace(
            go.Scatter(
                x=equity_df['timestamp'],
                y=equity_df['equity'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ),
            secondary_y=False
        )
        
        # Add price
        fig1.add_trace(
            go.Scatter(
                x=equity_df['timestamp'],
                y=equity_df['price'],
                mode='lines',
                name=f'{self.symbol} Price',
                line=dict(color='gray', width=1, dash='dot')
            ),
            secondary_y=True
        )
        
        # Add entry/exit points
        if self.strategy_performance['trades']:
            # Convert trade data to DataFrame
            trades_df = pd.DataFrame([
                {
                    'entry_time': trade['entry_time'] if isinstance(trade['entry_time'], datetime) else pd.to_datetime(trade['entry_time']),
                    'exit_time': trade['exit_time'] if isinstance(trade['exit_time'], datetime) else pd.to_datetime(trade['exit_time']),
                    'entry_price': trade['entry_price'],
                    'exit_price': trade['exit_price'],
                    'position_type': trade['position_type'],
                    'profit_loss': trade['profit_loss'],
                    'profit_loss_pct': trade['profit_loss_pct']
                }
                for trade in self.strategy_performance['trades']
            ])
            
            # Add entry points
            fig1.add_trace(
                go.Scatter(
                    x=trades_df[trades_df['position_type'] == 'long']['entry_time'],
                    y=trades_df[trades_df['position_type'] == 'long']['entry_price'],
                    mode='markers',
                    name='Long Entry',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ),
                secondary_y=True
            )
            
            fig1.add_trace(
                go.Scatter(
                    x=trades_df[trades_df['position_type'] == 'short']['entry_time'],
                    y=trades_df[trades_df['position_type'] == 'short']['entry_price'],
                    mode='markers',
                    name='Short Entry',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ),
                secondary_y=True
            )
            
            # Add exit points
            fig1.add_trace(
                go.Scatter(
                    x=trades_df[trades_df['position_type'] == 'long']['exit_time'],
                    y=trades_df[trades_df['position_type'] == 'long']['exit_price'],
                    mode='markers',
                    name='Long Exit',
                    marker=dict(color='green', size=8, symbol='circle', line=dict(color='white', width=1))
                ),
                secondary_y=True
            )
            
            fig1.add_trace(
                go.Scatter(
                    x=trades_df[trades_df['position_type'] == 'short']['exit_time'],
                    y=trades_df[trades_df['position_type'] == 'short']['exit_price'],
                    mode='markers',
                    name='Short Exit',
                    marker=dict(color='red', size=8, symbol='circle', line=dict(color='white', width=1))
                ),
                secondary_y=True
            )
        
        fig1.update_layout(
            title=f"Strategy Performance: {self.symbol}/{self.interval}",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            yaxis2_title="Price ($)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=500
        )
        
        figures['equity_curve'] = fig1
        
        # Create trade performance breakdown
        if self.strategy_performance['trades']:
            # Create pie chart of win/loss ratio
            metrics = self.strategy_performance['metrics']
            win_loss_data = [metrics['winning_trades'], metrics['losing_trades']]
            
            fig2 = go.Figure(data=[go.Pie(
                labels=['Winning Trades', 'Losing Trades'],
                values=win_loss_data,
                hole=.4,
                marker_colors=['green', 'red']
            )])
            
            fig2.update_layout(
                title="Trade Win/Loss Ratio",
                height=400
            )
            
            figures['win_loss_ratio'] = fig2
            
            # Create histogram of trade profits/losses
            trades_df = pd.DataFrame(self.strategy_performance['trades'])
            
            fig3 = go.Figure()
            
            fig3.add_trace(go.Histogram(
                x=trades_df['profit_loss_pct'],
                nbinsx=20,
                marker_color=['red' if x <= 0 else 'green' for x in trades_df['profit_loss_pct']],
                opacity=0.7
            ))
            
            fig3.update_layout(
                title="Distribution of Trade Returns",
                xaxis_title="Profit/Loss (%)",
                yaxis_title="Number of Trades",
                height=400
            )
            
            figures['trade_distribution'] = fig3
            
            # Create scatter plot of trade duration vs profit/loss
            fig4 = go.Figure(data=go.Scatter(
                x=trades_df['duration'],
                y=trades_df['profit_loss_pct'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=['red' if x <= 0 else 'green' for x in trades_df['profit_loss_pct']],
                    opacity=0.7
                ),
                text=[f"{t['position_type']} - P/L: {t['profit_loss_pct']:.2f}%" for t in self.strategy_performance['trades']]
            ))
            
            fig4.update_layout(
                title="Trade Duration vs. Return",
                xaxis_title="Duration (hours)",
                yaxis_title="Profit/Loss (%)",
                height=400
            )
            
            figures['duration_vs_return'] = fig4
        
        # If a streamlit container is provided, render the plots directly
        if render_to:
            try:
                import streamlit as st
                
                # Equity curve
                render_to.subheader("Strategy Performance")
                render_to.plotly_chart(figures['equity_curve'], use_container_width=True)
                
                # Display metrics
                metrics = self.strategy_performance['metrics']
                render_to.subheader("Performance Metrics")
                
                col1, col2, col3, col4 = render_to.columns(4)
                col1.metric("Total Return", f"{metrics['total_return_pct']:.2f}%")
                col2.metric("Win Rate", f"{metrics['win_rate']*100:.1f}%")
                col3.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
                col4.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
                
                if 'win_loss_ratio' in figures:
                    col1, col2 = render_to.columns(2)
                    col1.plotly_chart(figures['win_loss_ratio'], use_container_width=True)
                    col2.plotly_chart(figures['trade_distribution'], use_container_width=True)
                    
                    render_to.plotly_chart(figures['duration_vs_return'], use_container_width=True)
            except Exception as e:
                logging.error(f"Error rendering plots to streamlit: {e}")
        
        return figures


# Helper function to generate trading strategies for multiple symbol/interval combinations
def generate_all_trading_strategies(symbols, intervals, risk_level='medium', optimize=True, days_back=90):
    """
    Generate trading strategies for all symbol/interval combinations
    
    Args:
        symbols: List of symbols to generate strategies for
        intervals: List of intervals to generate strategies for
        risk_level: Risk level of the strategies
        optimize: Whether to optimize parameters with backtesting
        days_back: How many days of historical data to use for optimization
        
    Returns:
        Dictionary with strategies
    """
    strategies = {}
    
    for symbol in symbols:
        strategies[symbol] = {}
        
        for interval in intervals:
            logging.info(f"Generating trading strategy for {symbol}/{interval}")
            
            try:
                # Initialize strategy generator
                generator = TradingStrategyGenerator(symbol, interval)
                
                # Generate strategy
                strategy = generator.generate_strategy(
                    risk_level=risk_level,
                    optimize=optimize,
                    days_back=days_back
                )
                
                strategies[symbol][interval] = {
                    'status': 'success',
                    'strategy': strategy
                }
            except Exception as e:
                logging.error(f"Error generating strategy for {symbol}/{interval}: {e}")
                strategies[symbol][interval] = {
                    'status': 'error',
                    'error': str(e)
                }
                
    return strategies