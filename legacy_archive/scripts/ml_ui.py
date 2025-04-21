"""
UI Components for Machine Learning Predictions

This module provides Streamlit UI components for:
1. Training ML models and ensembles
2. Displaying predictions with sentiment and market regime analysis
3. Showing model performance metrics and backtesting results
4. Managing continuous learning
5. Generating optimized trading strategies
"""

import os
import logging
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import time
import json

# ML libraries
import joblib

# Local imports
from ml_prediction import MLPredictor, train_all_models, predict_for_all, continuous_learning_cycle
from ml_ensemble import EnsemblePredictor, train_ensemble_models
from ml_market_regime import MarketRegimeDetector, train_all_regime_models
from ml_sentiment_integration import SentimentIntegrator, integrate_sentiment_with_prediction
from ml_backtesting import MLBacktester, run_all_backtests
from ml_trading_strategy import TradingStrategyGenerator, generate_all_trading_strategies
from binance_api import get_available_symbols
from utils import timeframe_to_interval, get_timeframe_options

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def render_ml_predictions_tab():
    """
    Render the ML Predictions tab in the Streamlit UI
    """
    st.header("Machine Learning Price Predictions")
    
    # Create subtabs for different ML functionalities
    subtabs = st.tabs([
        "Predictions", 
        "Model Training", 
        "Market Regime", 
        "Sentiment Analysis", 
        "Backtesting", 
        "Trading Strategies", 
        "Performance Metrics", 
        "Continuous Learning"
    ])
    
    # Render each subtab
    with subtabs[0]:
        render_predictions_subtab()
    
    with subtabs[1]:
        render_training_subtab()
    
    with subtabs[2]:
        render_market_regime_subtab()
    
    with subtabs[3]:
        render_sentiment_analysis_subtab()
    
    with subtabs[4]:
        render_backtesting_subtab()
    
    with subtabs[5]:
        render_trading_strategy_subtab()
    
    with subtabs[6]:
        render_metrics_subtab()
    
    with subtabs[7]:
        render_continuous_learning_subtab()

def render_predictions_subtab():
    """
    Render the Predictions subtab
    """
    st.subheader("Price Movement Predictions")
    
    # Sidebar for prediction controls
    st.sidebar.header("Prediction Settings")
    
    # Get available symbols
    available_symbols = get_available_symbols()
    default_symbol = "BTCUSDT"
    if default_symbol not in available_symbols and available_symbols:
        default_symbol = available_symbols[0]
    
    # Symbol selection
    symbol = st.sidebar.selectbox(
        "Select Cryptocurrency",
        options=available_symbols,
        index=available_symbols.index(default_symbol) if default_symbol in available_symbols else 0,
        key="ml_pred_symbol"
    )
    
    # Timeframe selection
    timeframe_options = get_timeframe_options()
    interval = st.sidebar.selectbox(
        "Select Timeframe",
        options=list(timeframe_options.keys()),
        index=3,  # Default to 1h
        key="ml_pred_interval"
    )
    binance_interval = timeframe_to_interval(interval)
    
    # Prediction periods
    prediction_periods = st.sidebar.slider(
        "Prediction Periods",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of future periods to predict"
    )
    
    # Run prediction button
    run_prediction = st.sidebar.button("Generate Prediction")
    
    # Check if model exists
    model_path = os.path.join('models', f"{symbol}_{binance_interval}_model.joblib")
    model_exists = os.path.exists(model_path)
    
    if not model_exists:
        st.warning(f"No trained model exists for {symbol}/{binance_interval}. Please train a model first.")
    
    # Run prediction if requested
    if run_prediction and model_exists:
        with st.spinner(f"Generating predictions for {symbol}/{interval}..."):
            try:
                # Initialize predictor
                predictor = MLPredictor(symbol, binance_interval)
                
                # Make prediction
                predictions_df = predictor.predict_future(periods=prediction_periods)
                
                if predictions_df.empty:
                    st.error("Failed to generate predictions. There may not be enough recent data.")
                else:
                    # Display results
                    st.success("Prediction generated successfully!")
                    
                    # Get the most recent actual price
                    latest_price = predictions_df[predictions_df['is_prediction'] == False]['close'].iloc[-1]
                    
                    # Get the prediction row
                    pred_row = predictions_df[predictions_df['is_prediction'] == True].iloc[0]
                    predicted_price = pred_row['predicted_price']
                    predicted_change = pred_row['predicted_change']
                    confidence = pred_row['confidence']
                    
                    # Calculate prediction time
                    pred_time = pred_row['timestamp']
                    
                    # Display prediction metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Current Price",
                            f"${latest_price:.2f}",
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            f"Predicted Price ({interval})",
                            f"${predicted_price:.2f}",
                            delta=f"{predicted_change*100:.2f}%",
                            delta_color="normal" if predicted_change >= 0 else "inverse"
                        )
                    
                    with col3:
                        st.metric(
                            "Confidence",
                            f"{confidence*100:.1f}%",
                            delta=None
                        )
                    
                    # Display prediction timestamp
                    st.info(f"Prediction for {pred_time.strftime('%Y-%m-%d %H:%M')} (next {interval} candle)")
                    
                    # Display prediction interpretation
                    if predicted_change > 0:
                        if confidence > 0.7:
                            signal = "Strong Buy Signal"
                            color = "green"
                        else:
                            signal = "Weak Buy Signal"
                            color = "lightgreen"
                    else:
                        if confidence > 0.7:
                            signal = "Strong Sell Signal"
                            color = "red"
                        else:
                            signal = "Weak Sell Signal"
                            color = "lightcoral"
                    
                    st.markdown(f"**Prediction Signal:** <span style='color:{color}'>{signal}</span>", unsafe_allow_html=True)
                    
                    # Add explanation
                    if predicted_change > 0:
                        explanation = (f"The model predicts the price will **increase {predicted_change*100:.2f}%** "
                                    f"in the next {interval} with a confidence of {confidence*100:.1f}%.")
                    else:
                        explanation = (f"The model predicts the price will **decrease {abs(predicted_change)*100:.2f}%** "
                                    f"in the next {interval} with a confidence of {confidence*100:.1f}%.")
                    
                    st.markdown(explanation)
                    
                    # Create a price chart with prediction
                    st.subheader("Price Chart with Prediction")
                    
                    # Extract data for chart
                    historical_data = predictions_df[predictions_df['is_prediction'] == False].iloc[-50:]  # Last 50 points
                    prediction_data = predictions_df[predictions_df['is_prediction'] == True]
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Add historical prices
                    fig.add_trace(go.Candlestick(
                        x=historical_data['timestamp'],
                        open=historical_data['open'],
                        high=historical_data['high'],
                        low=historical_data['low'],
                        close=historical_data['close'],
                        name="Historical Prices"
                    ))
                    
                    # Add predicted price
                    fig.add_trace(go.Scatter(
                        x=prediction_data['timestamp'],
                        y=[latest_price] + [predicted_price] + [None] * (len(prediction_data) - 1),
                        mode='lines+markers',
                        name="Predicted Price",
                        line=dict(color='rgba(255, 165, 0, 0.8)', width=3, dash='dot'),
                        marker=dict(size=10, color='orange')
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{symbol} Price Prediction",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        height=500,
                        xaxis_rangeslider_visible=False
                    )
                    
                    # Show chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add disclaimer
                    st.caption("**Disclaimer:** Predictions are based on historical patterns and technical indicators. They should not be considered as financial advice. Cryptocurrency markets are highly volatile and past performance does not guarantee future results.")
                    
                    # Load model metadata for insights
                    metadata_path = os.path.join('models', f"{symbol}_{binance_interval}_metadata.json")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Display model insights
                        with st.expander("Model Insights"):
                            st.write("### Model Performance Metrics")
                            
                            metrics_df = pd.DataFrame({
                                'Metric': ['Direction Accuracy', 'Mean Squared Error', 'R-squared'],
                                'Value': [
                                    f"{metadata['metrics'].get('direction_accuracy', 0) * 100:.1f}%",
                                    f"{metadata['metrics'].get('mse', 0):.6f}",
                                    f"{metadata['metrics'].get('r2', 0):.3f}"
                                ]
                            })
                            
                            st.dataframe(metrics_df, use_container_width=True)
                            
                            st.write("### Feature Importance")
                            if 'feature_importance' in metadata['metrics']:
                                feature_imp = metadata['metrics']['feature_importance']
                                # Convert to DataFrame for display
                                fi_df = pd.DataFrame({
                                    'Feature': list(feature_imp.keys()),
                                    'Importance': list(feature_imp.values())
                                })
                                fi_df = fi_df.sort_values('Importance', ascending=False).head(10)
                                
                                # Create bar chart
                                fig = go.Figure(go.Bar(
                                    x=fi_df['Importance'],
                                    y=fi_df['Feature'],
                                    orientation='h',
                                    marker_color='darkblue'
                                ))
                                
                                fig.update_layout(
                                    title="Top 10 Most Important Features",
                                    xaxis_title="Importance",
                                    yaxis_title="Feature",
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Feature importance not available for this model")
                    
            except Exception as e:
                st.error(f"Error generating prediction: {str(e)}")
                logging.error(f"Prediction error: {e}")
                import traceback
                logging.error(traceback.format_exc())
    
    # No prediction requested or model doesn't exist
    elif not run_prediction and model_exists:
        # Show info about available model
        st.info(f"Model for {symbol}/{interval} is available for predictions. Click 'Generate Prediction' to create a forecast.")
        
        # Display last prediction if available
        prediction_cache_file = os.path.join('models', f"{symbol}_{binance_interval}_last_prediction.json")
        if os.path.exists(prediction_cache_file):
            try:
                with open(prediction_cache_file, 'r') as f:
                    last_prediction = json.load(f)
                
                st.subheader("Last Prediction")
                
                # Check if prediction is recent (within 24 hours)
                pred_time = datetime.fromisoformat(last_prediction['timestamp'])
                now = datetime.now()
                
                if (now - pred_time).total_seconds() < 86400:  # 24 hours
                    # Display last prediction
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Base Price",
                            f"${last_prediction['base_price']:.2f}",
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            "Predicted Price",
                            f"${last_prediction['predicted_price']:.2f}",
                            delta=f"{last_prediction['predicted_change']*100:.2f}%",
                            delta_color="normal" if last_prediction['predicted_change'] >= 0 else "inverse"
                        )
                    
                    with col3:
                        st.metric(
                            "Confidence",
                            f"{last_prediction['confidence']*100:.1f}%",
                            delta=None
                        )
                    
                    st.caption(f"Prediction made at {pred_time.strftime('%Y-%m-%d %H:%M')} (Click 'Generate Prediction' for a new forecast)")
                else:
                    st.warning("The last prediction is more than 24 hours old. Generate a new prediction for the latest forecast.")
            except Exception as e:
                logging.error(f"Error loading last prediction: {e}")
    
    # Share models across multiple symbols
    st.subheader("Quick Multi-Symbol Predictions")
    
    # Get list of models
    models_dir = "models"
    available_models = []
    
    if os.path.exists(models_dir):
        for filename in os.listdir(models_dir):
            if filename.endswith("_model.joblib"):
                model_id = filename.replace("_model.joblib", "")
                available_models.append(model_id)
    
    if available_models:
        # Generate predictions for all models
        if st.button("Generate Quick Predictions for All Symbols"):
            with st.spinner("Generating predictions for all models..."):
                # Parse models to get symbol/interval combinations
                symbols = []
                intervals = []
                
                for model_id in available_models:
                    parts = model_id.split("_")
                    if len(parts) >= 2:
                        symbol = parts[0]
                        interval = parts[1]
                        if symbol not in symbols:
                            symbols.append(symbol)
                        if interval not in intervals:
                            intervals.append(interval)
                
                # Generate predictions
                all_predictions = predict_for_all(symbols, intervals)
                
                if all_predictions:
                    # Create a table of predictions
                    prediction_data = []
                    
                    for symbol, intervals_data in all_predictions.items():
                        for interval, pred_data in intervals_data.items():
                            if pred_data.get('status') == 'success':
                                prediction_data.append({
                                    'Symbol': symbol,
                                    'Interval': interval,
                                    'Direction': '🔼' if pred_data['direction'] == 'up' else '🔽',
                                    'Change': f"{pred_data['predicted_change']*100:.2f}%",
                                    'Confidence': f"{pred_data['confidence']*100:.1f}%",
                                    'signal_value': 1 if pred_data['direction'] == 'up' else -1,
                                    'confidence_value': pred_data['confidence'],
                                    'change_value': pred_data['predicted_change']
                                })
                    
                    if prediction_data:
                        # Convert to DataFrame
                        pred_df = pd.DataFrame(prediction_data)
                        
                        # Sort by confidence and direction
                        pred_df['combined_score'] = pred_df['confidence_value'] * abs(pred_df['change_value']) * pred_df['signal_value']
                        pred_df = pred_df.sort_values('combined_score', ascending=False)
                        
                        # Remove sorting columns
                        display_df = pred_df.drop(['signal_value', 'confidence_value', 'change_value', 'combined_score'], axis=1)
                        
                        # Show the table
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Group and display strongest signals
                        st.subheader("Strongest Signals")
                        
                        # Top buy signals
                        buy_signals = pred_df[pred_df['signal_value'] > 0].head(3)
                        if not buy_signals.empty:
                            st.markdown("#### Top Buy Signals")
                            for _, row in buy_signals.iterrows():
                                st.markdown(f"**{row['Symbol']} ({row['Interval']})**: {row['Change']} with {row['Confidence']} confidence")
                        
                        # Top sell signals
                        sell_signals = pred_df[pred_df['signal_value'] < 0].head(3)
                        if not sell_signals.empty:
                            st.markdown("#### Top Sell Signals")
                            for _, row in sell_signals.iterrows():
                                st.markdown(f"**{row['Symbol']} ({row['Interval']})**: {row['Change']} with {row['Confidence']} confidence")
                    else:
                        st.warning("No valid predictions generated. Try training models first.")
                else:
                    st.error("Failed to generate predictions.")
        else:
            st.info("Click the button to generate quick predictions for all available models.")
    else:
        st.warning("No trained models available. Train models in the 'Model Training' tab first.")

def render_training_subtab():
    """
    Render the Model Training subtab
    """
    st.subheader("Train Prediction Models")
    
    # Sidebar for training controls
    st.sidebar.header("Training Settings")
    
    # Get available symbols
    available_symbols = get_available_symbols(limit=15)
    
    # Symbol selection (multi-select)
    selected_symbols = st.sidebar.multiselect(
        "Select Cryptocurrencies to Train",
        options=available_symbols,
        default=["BTCUSDT", "ETHUSDT"],
        key="ml_train_symbols"
    )
    
    # Interval selection (multi-select)
    timeframe_options = get_timeframe_options()
    intervals = list(timeframe_options.keys())
    selected_intervals = st.sidebar.multiselect(
        "Select Timeframes to Train",
        options=intervals,
        default=["1h", "4h"],
        key="ml_train_intervals"
    )
    
    # Convert to Binance intervals
    binance_intervals = [timeframe_to_interval(interval) for interval in selected_intervals]
    
    # Training lookback period
    lookback_days = st.sidebar.slider(
        "Training Data Period (days)",
        min_value=30,
        max_value=365,
        value=90,
        help="How many days of historical data to use for training"
    )
    
    # Retrain existing models
    retrain_existing = st.sidebar.checkbox(
        "Retrain Existing Models",
        value=False,
        help="Whether to retrain models that already exist"
    )
    
    # Training controls
    col1, col2 = st.columns(2)
    
    with col1:
        # Train single model
        if st.button("Train Selected Models"):
            if not selected_symbols:
                st.error("Please select at least one cryptocurrency.")
            elif not selected_intervals:
                st.error("Please select at least one timeframe.")
            else:
                # Show progress message
                progress_placeholder = st.empty()
                progress_placeholder.info(f"Training models for {len(selected_symbols)} symbols and {len(selected_intervals)} intervals. This may take a few minutes...")
                
                # Initialize session state for training
                if 'ml_training_thread' not in st.session_state:
                    st.session_state.ml_training_thread = None
                    st.session_state.ml_training_results = None
                
                # Define training thread function
                def train_models_thread():
                    try:
                        results = train_all_models(
                            selected_symbols, 
                            binance_intervals,
                            lookback_days=lookback_days,
                            retrain=retrain_existing
                        )
                        st.session_state.ml_training_results = results
                    except Exception as e:
                        logging.error(f"Training error: {e}")
                        st.session_state.ml_training_results = {"error": str(e)}
                
                # Start training in background thread
                if st.session_state.ml_training_thread is None or not st.session_state.ml_training_thread.is_alive():
                    st.session_state.ml_training_thread = threading.Thread(target=train_models_thread)
                    st.session_state.ml_training_thread.daemon = True
                    st.session_state.ml_training_thread.start()
                    st.session_state.ml_training_results = None
                
                # Check if training is complete
                if st.session_state.ml_training_results is not None:
                    progress_placeholder.success("Training complete!")
                    
                    # Display training results
                    training_results = st.session_state.ml_training_results
                    
                    if "error" in training_results:
                        st.error(f"Training error: {training_results['error']}")
                    else:
                        # Create a summary of results
                        success_count = 0
                        failed_count = 0
                        
                        for symbol, intervals_data in training_results.items():
                            for interval, result_data in intervals_data.items():
                                if result_data.get('status') == 'success':
                                    success_count += 1
                                else:
                                    failed_count += 1
                        
                        st.write(f"Successfully trained {success_count} models. Failed: {failed_count}")
                        
                        # Create a table of model metrics
                        metrics_data = []
                        
                        for symbol, intervals_data in training_results.items():
                            for interval, result_data in intervals_data.items():
                                if result_data.get('status') == 'success' and 'metrics' in result_data:
                                    metrics = result_data['metrics']
                                    metrics_data.append({
                                        'Symbol': symbol,
                                        'Interval': interval,
                                        'Direction Accuracy': f"{metrics.get('direction_accuracy', 0) * 100:.1f}%",
                                        'RMSE': f"{metrics.get('rmse', 0):.6f}",
                                        'R2': f"{metrics.get('r2', 0):.3f}"
                                    })
                        
                        if metrics_data:
                            metrics_df = pd.DataFrame(metrics_data)
                            st.dataframe(metrics_df, use_container_width=True)
                        
                        # Alert if there were failures
                        if failed_count > 0:
                            with st.expander("View Training Failures"):
                                for symbol, intervals_data in training_results.items():
                                    for interval, result_data in intervals_data.items():
                                        if result_data.get('status') != 'success':
                                            st.error(f"{symbol}/{interval}: {result_data.get('error', 'Unknown error')}")
                else:
                    # Show spinner while training
                    st.info("Training in progress... please wait. This may take several minutes depending on the amount of data.")
    
    with col2:
        # Train all models
        if st.button("Train All Popular Models"):
            # Initialize session state for all-model training
            if 'ml_all_training_thread' not in st.session_state:
                st.session_state.ml_all_training_thread = None
                st.session_state.ml_all_training_results = None
            
            # Show progress message
            all_progress_placeholder = st.empty()
            all_progress_placeholder.info("Training models for all popular cryptocurrencies. This may take 10-15 minutes...")
            
            # Get popular symbols
            from backfill_database import get_popular_symbols
            popular_symbols = get_popular_symbols(limit=10)
            
            # Select common intervals
            all_intervals = ['1h', '4h', '1d']
            all_binance_intervals = [timeframe_to_interval(interval) for interval in all_intervals]
            
            # Define training thread function for all models
            def train_all_models_thread():
                try:
                    results = train_all_models(
                        popular_symbols, 
                        all_binance_intervals,
                        lookback_days=90,
                        retrain=retrain_existing
                    )
                    st.session_state.ml_all_training_results = results
                except Exception as e:
                    logging.error(f"All-models training error: {e}")
                    st.session_state.ml_all_training_results = {"error": str(e)}
            
            # Start training in background thread
            if st.session_state.ml_all_training_thread is None or not st.session_state.ml_all_training_thread.is_alive():
                st.session_state.ml_all_training_thread = threading.Thread(target=train_all_models_thread)
                st.session_state.ml_all_training_thread.daemon = True
                st.session_state.ml_all_training_thread.start()
                st.session_state.ml_all_training_results = None
            
            # Check if training is complete
            if st.session_state.ml_all_training_results is not None:
                all_progress_placeholder.success("Training complete for all popular models!")
                
                # Display training results
                training_results = st.session_state.ml_all_training_results
                
                if "error" in training_results:
                    st.error(f"Training error: {training_results['error']}")
                else:
                    # Create a summary of results
                    success_count = 0
                    failed_count = 0
                    
                    for symbol, intervals_data in training_results.items():
                        for interval, result_data in intervals_data.items():
                            if result_data.get('status') == 'success':
                                success_count += 1
                            else:
                                failed_count += 1
                    
                    st.write(f"Successfully trained {success_count} models. Failed: {failed_count}")
            else:
                # Show spinner while training
                st.info("Training in progress... please wait. This may take 10-15 minutes.")
    
    # Show existing models
    st.subheader("Available Trained Models")
    
    # Get list of models
    models_dir = "models"
    available_models = []
    
    if os.path.exists(models_dir):
        for filename in os.listdir(models_dir):
            if filename.endswith("_model.joblib"):
                model_id = filename.replace("_model.joblib", "")
                
                # Get metadata if available
                metadata = {}
                metadata_path = os.path.join(models_dir, f"{model_id}_metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    except:
                        pass
                
                # Add to list
                available_models.append({
                    'id': model_id,
                    'parts': model_id.split('_'),
                    'timestamp': metadata.get('timestamp', 'Unknown'),
                    'metrics': metadata.get('metrics', {})
                })
    
    if available_models:
        # Convert to DataFrame for display
        models_df = pd.DataFrame([
            {
                'Symbol': model['parts'][0],
                'Interval': model['parts'][1] if len(model['parts']) > 1 else 'Unknown',
                'Trained On': datetime.fromisoformat(model['timestamp']).strftime('%Y-%m-%d %H:%M') if model['timestamp'] != 'Unknown' else 'Unknown',
                'Direction Accuracy': f"{model['metrics'].get('direction_accuracy', 0) * 100:.1f}%" if 'direction_accuracy' in model['metrics'] else 'Unknown',
                'R2 Score': f"{model['metrics'].get('r2', 0):.3f}" if 'r2' in model['metrics'] else 'Unknown'
            }
            for model in available_models
        ])
        
        # Show the table
        st.dataframe(models_df, use_container_width=True)
    else:
        st.info("No trained models available. Train models using the controls above.")

def render_metrics_subtab():
    """
    Render the Performance Metrics subtab
    """
    st.subheader("Model Performance Metrics")
    
    # Get list of models
    models_dir = "models"
    available_models = []
    
    if os.path.exists(models_dir):
        for filename in os.listdir(models_dir):
            if filename.endswith("_metadata.json"):
                model_id = filename.replace("_metadata.json", "")
                parts = model_id.split('_')
                if len(parts) >= 2:
                    symbol = parts[0]
                    interval = parts[1]
                    
                    # Read metadata
                    try:
                        with open(os.path.join(models_dir, filename), 'r') as f:
                            metadata = json.load(f)
                        
                        available_models.append({
                            'id': model_id,
                            'symbol': symbol,
                            'interval': interval,
                            'metadata': metadata
                        })
                    except:
                        pass
    
    if not available_models:
        st.warning("No trained models available. Train models in the 'Model Training' tab first.")
        return
    
    # Model selection
    model_options = [f"{model['symbol']}/{model['interval']}" for model in available_models]
    selected_model = st.selectbox("Select Model", options=model_options)
    
    # Find the selected model
    selected_model_data = next((model for model in available_models 
                              if f"{model['symbol']}/{model['interval']}" == selected_model), None)
    
    if selected_model_data:
        metadata = selected_model_data['metadata']
        symbol = selected_model_data['symbol']
        interval = selected_model_data['interval']
        
        # Display training details
        st.write(f"### Model for {symbol}/{interval}")
        
        if 'timestamp' in metadata:
            try:
                trained_on = datetime.fromisoformat(metadata['timestamp']).strftime('%Y-%m-%d %H:%M')
                st.write(f"**Trained on:** {trained_on}")
            except:
                st.write("**Trained on:** Unknown")
        
        # Feature window and prediction horizon
        st.write(f"**Feature Window:** {metadata.get('feature_window', 'Unknown')} periods")
        st.write(f"**Prediction Horizon:** {metadata.get('prediction_horizon', 'Unknown')} periods")
        
        # Show metrics
        if 'metrics' in metadata:
            metrics = metadata['metrics']
            
            # Performance metrics
            st.subheader("Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                direction_acc = metrics.get('direction_accuracy', 0) * 100
                st.metric("Direction Accuracy", f"{direction_acc:.1f}%")
            
            with col2:
                mse = metrics.get('mse', 0)
                st.metric("Mean Squared Error", f"{mse:.6f}")
            
            with col3:
                rmse = metrics.get('rmse', 0)
                st.metric("RMSE", f"{rmse:.6f}")
            
            with col4:
                r2 = metrics.get('r2', 0)
                st.metric("R² Score", f"{r2:.3f}")
            
            # Interpretation of metrics
            st.write("### Interpretation")
            
            if 'direction_accuracy' in metrics:
                direction_acc = metrics['direction_accuracy']
                if direction_acc > 0.7:
                    quality = "excellent"
                elif direction_acc > 0.6:
                    quality = "good"
                elif direction_acc > 0.55:
                    quality = "moderate"
                else:
                    quality = "poor"
                
                st.write(f"This model has **{quality} directional accuracy** ({direction_acc*100:.1f}%), "
                       f"meaning it correctly predicts the direction of price movement {direction_acc*100:.1f}% of the time.")
            
            if 'r2' in metrics:
                r2 = metrics['r2']
                if r2 > 0.7:
                    r2_quality = "excellent"
                elif r2 > 0.5:
                    r2_quality = "good"
                elif r2 > 0.3:
                    r2_quality = "moderate"
                else:
                    r2_quality = "weak"
                
                st.write(f"The R² score of {r2:.3f} indicates a **{r2_quality} fit** to the data.")
            
            # Feature importance
            if 'feature_importance' in metrics:
                st.subheader("Feature Importance")
                
                feature_imp = metrics['feature_importance']
                
                # Convert to DataFrame for display
                fi_df = pd.DataFrame({
                    'Feature': list(feature_imp.keys()),
                    'Importance': list(feature_imp.values())
                })
                
                # Sort by importance
                fi_df = fi_df.sort_values('Importance', ascending=False)
                
                # Create bar chart for top features
                top_features = fi_df.head(15)  # Top 15 features
                
                fig = go.Figure(go.Bar(
                    x=top_features['Importance'],
                    y=top_features['Feature'],
                    orientation='h',
                    marker_color='darkblue'
                ))
                
                fig.update_layout(
                    title="Top Features by Importance",
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature categories analysis
                st.subheader("Feature Category Analysis")
                
                # Group features by category
                categories = {
                    'Price Lags': [f for f in fi_df['Feature'] if 'close_lag' in f or 'open_lag' in f],
                    'Volume Features': [f for f in fi_df['Feature'] if 'volume' in f],
                    'Moving Averages': [f for f in fi_df['Feature'] if 'ma_' in f],
                    'Price Changes': [f for f in fi_df['Feature'] if 'price_change' in f],
                    'Technical Indicators': [f for f in fi_df['Feature'] if any(ind in f for ind in ['rsi', 'macd', 'bb_'])],
                    'Other': []
                }
                
                # Calculate importance by category
                category_importance = {}
                
                for category, features in categories.items():
                    if features:
                        category_df = fi_df[fi_df['Feature'].isin(features)]
                        category_importance[category] = category_df['Importance'].sum()
                
                # Add remaining features to 'Other'
                categorized_features = [f for cat_features in categories.values() for f in cat_features]
                other_features = [f for f in fi_df['Feature'] if f not in categorized_features]
                categories['Other'] = other_features
                
                if other_features:
                    other_df = fi_df[fi_df['Feature'].isin(other_features)]
                    category_importance['Other'] = other_df['Importance'].sum()
                
                # Create pie chart
                cat_fig = go.Figure(data=[go.Pie(
                    labels=list(category_importance.keys()),
                    values=list(category_importance.values()),
                    hole=.4
                )])
                
                cat_fig.update_layout(
                    title="Feature Importance by Category",
                    height=400
                )
                
                st.plotly_chart(cat_fig, use_container_width=True)
        else:
            st.warning("No metrics available for this model.")
    else:
        st.error("Selected model not found.")

def render_market_regime_subtab():
    """
    Render the Market Regime Detection subtab
    """
    st.subheader("Market Regime Detection")
    
    st.markdown("""
    Market regime detection identifies the current market conditions (trending, ranging, volatile) 
    to help optimize trading strategies and predictions for different market environments.
    """)
    
    # Sidebar for controls
    st.sidebar.header("Market Regime Settings")
    
    # Get available symbols
    available_symbols = get_available_symbols()
    default_symbol = "BTCUSDT"
    if default_symbol not in available_symbols and available_symbols:
        default_symbol = available_symbols[0]
    
    # Symbol selection
    symbol = st.sidebar.selectbox(
        "Select Cryptocurrency",
        options=available_symbols,
        index=available_symbols.index(default_symbol) if default_symbol in available_symbols else 0,
        key="regime_symbol"
    )
    
    # Timeframe selection
    timeframe_options = get_timeframe_options()
    interval = st.sidebar.selectbox(
        "Select Timeframe",
        options=list(timeframe_options.keys()),
        index=4,  # Default to 4h
        key="regime_interval"
    )
    binance_interval = timeframe_to_interval(interval)
    
    # Actions
    detect_regime = st.sidebar.button("Detect Current Regime")
    train_model = st.sidebar.button("Train Regime Model")
    
    # Check if model exists
    model_path = os.path.join('models', f"{symbol}_{binance_interval}_regime_model.joblib")
    model_exists = os.path.exists(model_path)
    
    if not model_exists:
        st.warning(f"No trained regime detection model exists for {symbol}/{binance_interval}. Please train a model first.")
    
    # Main content area
    if detect_regime and model_exists:
        with st.spinner(f"Detecting market regime for {symbol}/{interval}..."):
            try:
                # Initialize regime detector
                detector = MarketRegimeDetector(symbol, binance_interval)
                
                # Detect current regime
                regime_info = detector.detect_current_regime()
                
                if regime_info:
                    # Display current regime
                    current_regime = regime_info['current_regime']['name']
                    
                    # Set regime color based on type
                    if current_regime == "RANGING":
                        regime_color = "blue"
                    elif current_regime == "TRENDING_UP":
                        regime_color = "green"
                    elif current_regime == "TRENDING_DOWN":
                        regime_color = "red"
                    elif current_regime == "VOLATILE":
                        regime_color = "purple"
                    else:
                        regime_color = "gray"
                    
                    # Display regime info
                    st.success("Market regime detected successfully!")
                    
                    # Show current regime
                    st.markdown(f"### Current Market Regime: <span style='color:{regime_color}'>{current_regime}</span>", unsafe_allow_html=True)
                    
                    # Display regime stability
                    stability = regime_info['current_regime']['stability']
                    st.markdown(f"Regime stability: **{stability}** periods")
                    
                    # Display price info
                    price_info = regime_info['price_info']
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Current Price",
                            f"${price_info['current_price']:.2f}",
                            delta=None
                        )
                    
                    with col2:
                        if price_info['price_change_24h'] is not None:
                            st.metric(
                                "24h Change",
                                "",
                                delta=f"{price_info['price_change_24h']*100:.2f}%",
                                delta_color="normal" if price_info['price_change_24h'] >= 0 else "inverse"
                            )
                        else:
                            st.metric("24h Change", "N/A", delta=None)
                    
                    with col3:
                        st.metric(
                            "Volatility",
                            f"{price_info['volatility']:.2f}%",
                            delta=None
                        )
                    
                    # Display next likely regime
                    next_regime = regime_info['next_likely_regime']
                    if next_regime['name'] != "UNKNOWN" and next_regime['probability'] is not None:
                        st.markdown(f"### Next Likely Regime")
                        st.markdown(f"**{next_regime['name']}** (Probability: {next_regime['probability']*100:.1f}%)")
                    
                    # Get trading strategy for this regime
                    strategy = detector.get_trading_strategy_for_regime(current_regime)
                    
                    # Display recommended trading strategy
                    st.markdown("### Recommended Trading Strategy")
                    st.markdown(f"**{strategy['name']}**")
                    st.markdown(f"{strategy['description']}")
                    
                    # Display strategy details
                    with st.expander("Strategy Details"):
                        st.markdown("#### Key Indicators")
                        st.write(", ".join(strategy['indicators']))
                        
                        st.markdown("#### Recommended Timeframes")
                        st.write(", ".join(strategy['timeframes']))
                        
                        st.markdown("#### Risk Level")
                        st.write(strategy['risk_level'])
                        
                        st.markdown("#### Stop Loss")
                        st.write(strategy['stop_loss'])
                        
                        st.markdown("#### Take Profit")
                        st.write(strategy['take_profit'])
                    
                    # Display regime history chart
                    st.markdown("### Recent Regime History")
                    
                    # Get regime history
                    regime_history = regime_info['current_regime']['recent_history']
                    
                    # Map regime labels to names
                    regime_names = []
                    for label in regime_history:
                        if hasattr(detector, 'metadata') and 'regime_mapping' in detector.metadata:
                            regime_name = detector.metadata['regime_mapping'].get(
                                str(label), detector.REGIME_TYPES.get(0, "UNKNOWN")
                            )
                        else:
                            regime_name = detector.REGIME_TYPES.get(label, "UNKNOWN")
                        regime_names.append(regime_name)
                    
                    # Create a bar chart of recent regimes
                    fig = go.Figure()
                    
                    # Define colors for regimes
                    colors = {
                        "RANGING": "blue",
                        "TRENDING_UP": "green",
                        "TRENDING_DOWN": "red",
                        "VOLATILE": "purple",
                        "UNKNOWN": "gray"
                    }
                    
                    # Add bars for regime history
                    for i, regime in enumerate(reversed(regime_names)):
                        fig.add_trace(go.Bar(
                            x=[i],
                            y=[1],
                            name=regime,
                            marker_color=colors.get(regime, "gray"),
                            text=regime,
                            hoverinfo="text"
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title="Recent Regime History (Most Recent on Right)",
                        showlegend=False,
                        height=200,
                        yaxis=dict(showticklabels=False, showgrid=False),
                        xaxis=dict(showticklabels=False, showgrid=False)
                    )
                    
                    # Show chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tips for this regime
                    st.markdown("### Tips for Current Regime")
                    
                    if current_regime == "RANGING":
                        st.markdown("""
                        - Look for overbought/oversold conditions using RSI and Bollinger Bands
                        - Enter trades near support and resistance levels
                        - Use tighter stop losses and lower profit targets
                        - Consider mean reversion strategies
                        """)
                    elif current_regime == "TRENDING_UP":
                        st.markdown("""
                        - Focus on long positions in the direction of the trend
                        - Look for pullbacks as entry opportunities
                        - Use trailing stops to lock in profits
                        - Consider trend-following indicators like moving averages and MACD
                        """)
                    elif current_regime == "TRENDING_DOWN":
                        st.markdown("""
                        - Be cautious with long positions against the trend
                        - Look for bounces as potential short opportunities
                        - Use tighter stop losses for counter-trend trades
                        - Watch for bearish divergences in indicators
                        """)
                    elif current_regime == "VOLATILE":
                        st.markdown("""
                        - Reduce position sizes due to increased risk
                        - Look for breakout setups with momentum
                        - Use wider stop losses to avoid being stopped out by volatility
                        - Consider waiting for volatility to settle before entering new positions
                        """)
                else:
                    st.error("Failed to detect market regime. There may not be enough data.")
            except Exception as e:
                st.error(f"Error detecting market regime: {str(e)}")
    
    if train_model:
        with st.spinner(f"Training market regime detection model for {symbol}/{interval}..."):
            try:
                # Initialize regime detector
                detector = MarketRegimeDetector(symbol, binance_interval)
                
                # Train model
                success = detector.train_regime_model(lookback_days=90)
                
                if success:
                    st.success(f"Successfully trained market regime detection model for {symbol}/{interval}!")
                    
                    # Detect current regime with new model
                    regime_info = detector.detect_current_regime()
                    
                    if regime_info:
                        current_regime = regime_info['current_regime']['name']
                        st.info(f"Current market regime: {current_regime}")
                else:
                    st.error("Failed to train market regime detection model. There may not be enough data.")
            except Exception as e:
                st.error(f"Error training market regime detection model: {str(e)}")
    
    # Information about market regimes
    with st.expander("About Market Regimes"):
        st.markdown("""
        ### Understanding Market Regimes
        
        Market regimes are distinct market environments with different characteristics that require different trading approaches.
        
        #### Main Regime Types:
        
        **Ranging Markets**
        - Prices move sideways between support and resistance levels
        - Mean reversion strategies work well
        - Lower volatility and more predictable boundaries
        
        **Trending Markets (Up/Down)**
        - Prices show a consistent directional movement
        - Trend-following strategies work well
        - Pullbacks offer entry opportunities in the trend direction
        
        **Volatile Markets**
        - Rapid price movements with high volatility
        - Increased risk requires smaller position sizes
        - Breakout strategies can be effective
        
        #### How Regime Detection Works:
        
        Our regime detection model uses machine learning to analyze multiple technical indicators and price patterns to identify the current market regime. It looks at:
        
        - Trend strength indicators like ADX
        - Volatility measures like ATR and Bollinger Band Width
        - Price momentum and moving average relationships
        - Support/resistance interactions
        
        By understanding the current regime, you can adapt your trading strategy to current market conditions rather than using a one-size-fits-all approach.
        """)

def render_sentiment_analysis_subtab():
    """
    Render the Sentiment Analysis subtab
    """
    st.subheader("Sentiment Analysis Integration")
    
    st.markdown("""
    This feature integrates market sentiment data with technical price predictions to create 
    more comprehensive forecasts that factor in both market sentiment and price patterns.
    """)
    
    # Sidebar for controls
    st.sidebar.header("Sentiment Settings")
    
    # Get available symbols
    available_symbols = get_available_symbols()
    default_symbol = "BTCUSDT"
    if default_symbol not in available_symbols and available_symbols:
        default_symbol = available_symbols[0]
    
    # Symbol selection
    symbol = st.sidebar.selectbox(
        "Select Cryptocurrency",
        options=available_symbols,
        index=available_symbols.index(default_symbol) if default_symbol in available_symbols else 0,
        key="sentiment_symbol"
    )
    
    # Timeframe selection
    timeframe_options = get_timeframe_options()
    interval = st.sidebar.selectbox(
        "Select Timeframe",
        options=list(timeframe_options.keys()),
        index=3,  # Default to 1h
        key="sentiment_interval"
    )
    binance_interval = timeframe_to_interval(interval)
    
    # Include news option
    include_news = st.sidebar.checkbox("Include News Sentiment", value=True)
    
    # Actions
    analyze_sentiment = st.sidebar.button("Analyze Sentiment")
    
    # Check if model exists
    model_path = os.path.join('models', f"{symbol}_{binance_interval}_model.joblib")
    model_exists = os.path.exists(model_path)
    
    if not model_exists:
        st.warning(f"No trained prediction model exists for {symbol}/{binance_interval}. Please train a model first.")
    
    # Main content area
    if analyze_sentiment and model_exists:
        with st.spinner(f"Analyzing sentiment for {symbol}/{interval}..."):
            try:
                # Initialize sentiment integrator
                integrator = SentimentIntegrator(symbol, binance_interval)
                
                # Fetch sentiment data
                sentiment_df = integrator.fetch_sentiment_data(days_back=7)
                
                if sentiment_df is not None and not sentiment_df.empty:
                    # Get sentiment summary
                    sentiment_summary = integrator.sentiment_summary
                    
                    # Get news sentiment if requested
                    if include_news:
                        news_sentiment = integrator.get_news_sentiment(max_articles=5)
                    else:
                        news_sentiment = None
                    
                    # Initialize predictor and get a prediction
                    predictor = MLPredictor(symbol, binance_interval)
                    prediction_result = predictor.predict()
                    
                    if prediction_result and 'predicted_change' in prediction_result and 'confidence' in prediction_result:
                        # Get technical prediction
                        predicted_change = prediction_result['predicted_change']
                        confidence = prediction_result['confidence']
                        
                        # Adjust prediction with sentiment
                        adjusted_prediction, adjusted_confidence, sentiment_info = integrator.adjust_prediction(
                            predicted_change, confidence, include_news=include_news
                        )
                        
                        # Get explanation
                        explanation = integrator.get_sentiment_explanation(sentiment_info)
                        
                        # Display results
                        st.success("Sentiment analysis completed successfully!")
                        
                        # Display sentiment metrics
                        st.markdown("### Current Sentiment Metrics")
                        
                        # Determine sentiment state
                        if sentiment_summary['recent_sentiment'] > 0.5:
                            sentiment_state = "Very Bullish"
                            sentiment_color = "green"
                        elif sentiment_summary['recent_sentiment'] > 0.1:
                            sentiment_state = "Bullish"
                            sentiment_color = "lightgreen"
                        elif sentiment_summary['recent_sentiment'] > -0.1:
                            sentiment_state = "Neutral"
                            sentiment_color = "gray"
                        elif sentiment_summary['recent_sentiment'] > -0.5:
                            sentiment_state = "Bearish"
                            sentiment_color = "lightcoral"
                        else:
                            sentiment_state = "Very Bearish"
                            sentiment_color = "red"
                        
                        # Display sentiment state
                        st.markdown(f"#### Overall Sentiment: <span style='color:{sentiment_color}'>{sentiment_state}</span>", unsafe_allow_html=True)
                        
                        # Display sentiment metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Recent Sentiment",
                                f"{sentiment_summary['recent_sentiment']*100:.1f}%",
                                delta=f"{sentiment_summary['sentiment_trend']*100:.1f}%",
                                delta_color="normal" if sentiment_summary['sentiment_trend'] >= 0 else "inverse"
                            )
                        
                        with col2:
                            st.metric(
                                "Weighted Sentiment",
                                f"{sentiment_summary['volume_weighted_sentiment']*100:.1f}%",
                                delta=None
                            )
                        
                        with col3:
                            st.metric(
                                "Data Points",
                                f"{sentiment_summary['data_points']}",
                                delta=None
                            )
                        
                        # Display news sentiment if available
                        if news_sentiment:
                            st.markdown("### News Sentiment")
                            
                            # Display news sentiment metric
                            st.metric(
                                "News Sentiment Score",
                                f"{news_sentiment['average_sentiment']*100:.1f}%",
                                delta=None
                            )
                            
                            # Show most interesting headlines
                            if news_sentiment['most_positive']:
                                st.markdown("#### Most Bullish Headline")
                                st.markdown(f"**{news_sentiment['most_positive']['title']}**")
                                st.markdown(f"Source: {news_sentiment['most_positive']['source']} | Sentiment: {news_sentiment['most_positive']['sentiment']*100:.1f}%")
                                st.markdown(f"[Read Article]({news_sentiment['most_positive']['url']})")
                            
                            if news_sentiment['most_negative']:
                                st.markdown("#### Most Bearish Headline")
                                st.markdown(f"**{news_sentiment['most_negative']['title']}**")
                                st.markdown(f"Source: {news_sentiment['most_negative']['source']} | Sentiment: {news_sentiment['most_negative']['sentiment']*100:.1f}%")
                                st.markdown(f"[Read Article]({news_sentiment['most_negative']['url']})")
                        
                        # Display prediction comparison
                        st.markdown("### Prediction Comparison")
                        
                        # Create comparison table
                        comparison_df = pd.DataFrame({
                            'Metric': ['Predicted Change', 'Confidence'],
                            'Technical Only': [f"{predicted_change*100:.2f}%", f"{confidence*100:.1f}%"],
                            'With Sentiment': [f"{adjusted_prediction*100:.2f}%", f"{adjusted_confidence*100:.1f}%"],
                            'Difference': [f"{(adjusted_prediction-predicted_change)*100:.2f}%", f"{(adjusted_confidence-confidence)*100:.1f}%"]
                        })
                        
                        # Display table
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Display sentiment explanation
                        st.markdown("### Sentiment Impact Explanation")
                        st.markdown(explanation)
                        
                        # Create sentiment visualization
                        # Sentiment Timeline
                        if not sentiment_df.empty and 'timestamp' in sentiment_df.columns and 'sentiment_score' in sentiment_df.columns:
                            st.markdown("### Sentiment Timeline")
                            
                            # Prepare data for visualization
                            sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
                            sentiment_df = sentiment_df.sort_values('timestamp')
                            
                            # Create figure
                            fig = go.Figure()
                            
                            # Add sentiment score line
                            fig.add_trace(go.Scatter(
                                x=sentiment_df['timestamp'],
                                y=sentiment_df['sentiment_score'],
                                mode='lines+markers',
                                name='Sentiment Score',
                                line=dict(color='blue', width=2),
                                marker=dict(
                                    size=6,
                                    color=sentiment_df['sentiment_score'].apply(
                                        lambda x: 'green' if x > 0 else 'red'
                                    )
                                )
                            ))
                            
                            # Add zero line
                            fig.add_shape(
                                type="line",
                                x0=sentiment_df['timestamp'].min(),
                                y0=0,
                                x1=sentiment_df['timestamp'].max(),
                                y1=0,
                                line=dict(color="gray", width=1, dash="dash")
                            )
                            
                            # Update layout
                            fig.update_layout(
                                title="Sentiment Score Over Time",
                                xaxis_title="Date",
                                yaxis_title="Sentiment Score",
                                height=400
                            )
                            
                            # Show chart
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Could not generate a prediction for sentiment adjustment.")
                else:
                    st.warning("No sentiment data available for this cryptocurrency.")
            except Exception as e:
                st.error(f"Error analyzing sentiment: {str(e)}")
    
    # Information about sentiment analysis
    with st.expander("About Sentiment Analysis"):
        st.markdown("""
        ### Understanding Sentiment Analysis
        
        Sentiment analysis measures the overall market mood toward a particular cryptocurrency by analyzing data from various sources:
        
        #### Data Sources:
        - Social media (Twitter, Reddit, etc.)
        - News articles and headlines
        - Trading forums and discussion boards
        - Market commentary
        
        #### How Sentiment Affects Prices:
        
        Market sentiment often drives cryptocurrency prices, sometimes more than technical factors. Strong positive sentiment can push prices higher, while negative sentiment can accelerate downtrends.
        
        #### Key Sentiment Metrics:
        
        **Recent Sentiment**  
        The average sentiment score from the most recent data points (last 24 hours).
        
        **Sentiment Trend**  
        The change in sentiment over time (positive means improving sentiment).
        
        **Volume-Weighted Sentiment**  
        Sentiment weighted by the volume of mentions/discussions, giving more importance to widely-discussed sentiment.
        
        #### Integration with Technical Predictions:
        
        Our sentiment integration adjusts technical price predictions based on current market sentiment:
        
        - When sentiment agrees with technical indicators, confidence increases
        - When sentiment contradicts technical indicators, confidence decreases
        - Strong sentiment can adjust the magnitude of predicted price movements
        
        This combined approach provides a more holistic view of potential price movements by considering both technical patterns and market psychology.
        """)

def render_backtesting_subtab():
    """
    Render the Backtesting subtab
    """
    st.subheader("ML Prediction Backtesting")
    
    st.markdown("""
    Backtest ML prediction performance on historical data to validate model accuracy
    and understand how different models perform in various market conditions.
    """)
    
    # Sidebar for controls
    st.sidebar.header("Backtesting Settings")
    
    # Get available symbols
    available_symbols = get_available_symbols()
    default_symbol = "BTCUSDT"
    if default_symbol not in available_symbols and available_symbols:
        default_symbol = available_symbols[0]
    
    # Symbol selection
    symbol = st.sidebar.selectbox(
        "Select Cryptocurrency",
        options=available_symbols,
        index=available_symbols.index(default_symbol) if default_symbol in available_symbols else 0,
        key="backtest_symbol"
    )
    
    # Timeframe selection
    timeframe_options = get_timeframe_options()
    interval = st.sidebar.selectbox(
        "Select Timeframe",
        options=list(timeframe_options.keys()),
        index=3,  # Default to 1h
        key="backtest_interval"
    )
    binance_interval = timeframe_to_interval(interval)
    
    # Backtest range
    days_back = st.sidebar.slider(
        "Backtest Period (days)",
        min_value=30,
        max_value=180,
        value=90,
        step=30,
        help="Number of days to backtest"
    )
    
    # Include additional features
    include_regimes = st.sidebar.checkbox("Include Market Regimes", value=True)
    include_sentiment = st.sidebar.checkbox("Include Sentiment Analysis", value=True)
    
    # Actions
    run_backtest = st.sidebar.button("Run Backtest")
    
    # Check if model exists
    model_path = os.path.join('models', f"{symbol}_{binance_interval}_model.joblib")
    model_exists = os.path.exists(model_path)
    
    ensemble_path = os.path.join('models', f"{symbol}_{binance_interval}_ensemble_model.joblib")
    ensemble_exists = os.path.exists(ensemble_path)
    
    if not model_exists and not ensemble_exists:
        st.warning(f"No trained models exist for {symbol}/{binance_interval}. Please train a model first.")
    
    # Main content area
    if run_backtest and (model_exists or ensemble_exists):
        with st.spinner(f"Running backtest for {symbol}/{interval} over {days_back} days..."):
            try:
                # Initialize backtester
                backtester = MLBacktester(symbol, binance_interval)
                
                # Run backtest
                results = backtester.run_backtest(
                    days_back=days_back,
                    include_regimes=include_regimes,
                    include_sentiment=include_sentiment
                )
                
                if results:
                    # Display results
                    st.success("Backtest completed successfully!")
                    
                    # Display visualizations directly in a container
                    backtester.create_performance_visualizations(render_to=st)
                    
                    # Get performance metrics table
                    metrics_tables = backtester.get_performance_metrics_table()
                    
                    if metrics_tables and 'overall' in metrics_tables:
                        st.markdown("### Performance Metrics Comparison")
                        st.dataframe(metrics_tables['overall'], use_container_width=True)
                    
                    if metrics_tables and 'by_regime' in metrics_tables and metrics_tables['by_regime'] is not None:
                        st.markdown("### Performance by Market Regime")
                        st.dataframe(metrics_tables['by_regime'], use_container_width=True)
                    
                    # Get model recommendations
                    recommendations = backtester.get_model_recommendations()
                    
                    if recommendations:
                        st.markdown("### Model Recommendations")
                        st.markdown(recommendations['explanation'])
                else:
                    st.error("Failed to run backtest. There may not be enough historical data.")
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")
    
    # Information about backtesting
    with st.expander("About Backtesting"):
        st.markdown("""
        ### Understanding ML Backtesting
        
        Backtesting evaluates how ML prediction models would have performed on historical data, which helps us understand their strengths and weaknesses.
        
        #### What Backtesting Measures:
        
        **Direction Accuracy**  
        The percentage of times the model correctly predicted the direction of price movement (up or down).
        
        **Mean Absolute Error (MAE)**  
        The average size of prediction errors, regardless of direction.
        
        **RMSE (Root Mean Squared Error)**  
        Similar to MAE but gives more weight to larger errors.
        
        **Model Confidence**  
        How confident the model was in its predictions and how this correlates with accuracy.
        
        #### Why Backtest Across Different Regimes:
        
        Different models perform better in different market conditions:
        
        - Some models excel in trending markets but perform poorly in ranging markets
        - Others may handle volatility well but struggle during strong trends
        - Ensemble approaches often provide more consistent performance across regimes
        
        #### How to Use Backtest Results:
        
        - Choose the best model for current market conditions
        - Adjust confidence thresholds based on backtest performance
        - Understand when to trust or be skeptical of predictions
        - Improve models by focusing on where they underperform
        
        Backtesting is not a guarantee of future performance, but it provides insights into how models are likely to behave in similar market conditions.
        """)

def render_trading_strategy_subtab():
    """
    Render the Trading Strategy subtab
    """
    st.subheader("ML-Based Trading Strategy Generator")
    
    st.markdown("""
    Generate optimized trading strategies based on ML predictions, adapted to current market regimes
    and sentiment analysis.
    """)
    
    # Sidebar for controls
    st.sidebar.header("Strategy Settings")
    
    # Get available symbols
    available_symbols = get_available_symbols()
    default_symbol = "BTCUSDT"
    if default_symbol not in available_symbols and available_symbols:
        default_symbol = available_symbols[0]
    
    # Symbol selection
    symbol = st.sidebar.selectbox(
        "Select Cryptocurrency",
        options=available_symbols,
        index=available_symbols.index(default_symbol) if default_symbol in available_symbols else 0,
        key="strategy_symbol"
    )
    
    # Timeframe selection
    timeframe_options = get_timeframe_options()
    interval = st.sidebar.selectbox(
        "Select Timeframe",
        options=list(timeframe_options.keys()),
        index=3,  # Default to 1h
        key="strategy_interval"
    )
    binance_interval = timeframe_to_interval(interval)
    
    # Risk level
    risk_level = st.sidebar.select_slider(
        "Risk Level",
        options=["low", "medium", "high"],
        value="medium",
        help="Low risk = smaller positions, tighter stops. High risk = larger positions, wider stops"
    )
    
    # Strategy options
    optimize = st.sidebar.checkbox("Optimize Parameters", value=True, help="Optimize strategy parameters based on backtesting")
    
    # Actions
    generate_strategy = st.sidebar.button("Generate Strategy")
    simulate_strategy = st.sidebar.button("Simulate Strategy")
    
    # Check if model exists
    model_path = os.path.join('models', f"{symbol}_{binance_interval}_model.joblib")
    model_exists = os.path.exists(model_path)
    
    ensemble_path = os.path.join('models', f"{symbol}_{binance_interval}_ensemble_model.joblib")
    ensemble_exists = os.path.exists(ensemble_path)
    
    if not model_exists and not ensemble_exists:
        st.warning(f"No trained models exist for {symbol}/{binance_interval}. Please train a model first.")
    
    # Main content area
    if generate_strategy and (model_exists or ensemble_exists):
        with st.spinner(f"Generating trading strategy for {symbol}/{interval}..."):
            try:
                # Initialize strategy generator
                generator = TradingStrategyGenerator(symbol, binance_interval)
                
                # Generate strategy
                strategy = generator.generate_strategy(
                    risk_level=risk_level,
                    optimize=optimize
                )
                
                if strategy:
                    # Display results
                    st.success("Trading strategy generated successfully!")
                    
                    # Display current market regime if available
                    if 'market_regime' in strategy and strategy['market_regime']:
                        st.markdown(f"### Current Market Regime: {strategy['market_regime']}")
                    
                    # Display strategy overview
                    st.markdown(f"### Trading Strategy for {symbol}/{interval}")
                    st.markdown(f"**Risk Level:** {risk_level.capitalize()}")
                    
                    # Display optimized parameters
                    with st.expander("Strategy Parameters"):
                        params = strategy['parameters']
                        
                        st.markdown("#### Entry Parameters")
                        st.markdown(f"- Entry threshold: **{params['entry_threshold']*100:.1f}%**")
                        st.markdown(f"- Confidence threshold: **{params['confidence_threshold']*100:.1f}%**")
                        
                        st.markdown("#### Position Sizing")
                        st.markdown(f"- Maximum position size: **{params['max_position_size']*100:.1f}%** of capital")
                        
                        st.markdown("#### Risk Management")
                        st.markdown(f"- Stop loss: **{params['stop_loss']*100:.1f}%**")
                        st.markdown(f"- Take profit: **{params['take_profit']*100:.1f}%**")
                        st.markdown(f"- Trailing stop: **{'Yes' if params['trailing_stop'] else 'No'}**")
                        
                        if params['trailing_stop']:
                            st.markdown(f"- Trailing distance: **{params['trailing_distance']*100:.1f}%**")
                        
                        st.markdown("#### Advanced Options")
                        st.markdown(f"- Use market regime: **{'Yes' if params['use_market_regime'] else 'No'}**")
                        st.markdown(f"- Use sentiment: **{'Yes' if params['use_sentiment'] else 'No'}**")
                        st.markdown(f"- Exit on opposite signal: **{'Yes' if params['exit_on_opposite_signal'] else 'No'}**")
                    
                    # Display trading plan
                    st.markdown("### Trading Plan")
                    st.markdown(strategy['rules']['trading_plan'])
                else:
                    st.error("Failed to generate trading strategy.")
            except Exception as e:
                st.error(f"Error generating trading strategy: {str(e)}")
    
    if simulate_strategy and (model_exists or ensemble_exists):
        with st.spinner(f"Simulating trading strategy for {symbol}/{interval}..."):
            try:
                # Initialize strategy generator
                generator = TradingStrategyGenerator(symbol, binance_interval)
                
                # Simulate strategy
                simulation = generator.simulate_strategy(days_back=30)
                
                if simulation:
                    # Display results
                    st.success("Strategy simulation completed successfully!")
                    
                    # Display performance visualizations
                    generator.create_strategy_visualizations(render_to=st)
                    
                    # Display trade details
                    if simulation['trades']:
                        st.markdown("### Trade Details")
                        
                        # Create trade details table
                        trades_df = pd.DataFrame([
                            {
                                'Type': trade['position_type'].capitalize(),
                                'Entry Time': pd.to_datetime(trade['entry_time']).strftime('%Y-%m-%d %H:%M'),
                                'Exit Time': pd.to_datetime(trade['exit_time']).strftime('%Y-%m-%d %H:%M'),
                                'Entry Price': f"${trade['entry_price']:.2f}",
                                'Exit Price': f"${trade['exit_price']:.2f}",
                                'P/L': f"${trade['profit_loss']:.2f}",
                                'P/L %': f"{trade['profit_loss_pct']:.2f}%",
                                'Duration (hrs)': f"{trade['duration']:.1f}"
                            }
                            for trade in simulation['trades']
                        ])
                        
                        # Display table
                        st.dataframe(trades_df, use_container_width=True)
                else:
                    st.error("Failed to simulate trading strategy.")
            except Exception as e:
                st.error(f"Error simulating trading strategy: {str(e)}")
    
    # Information about trading strategies
    with st.expander("About Trading Strategies"):
        st.markdown("""
        ### Understanding ML-Based Trading Strategies
        
        Our trading strategy generator creates optimized trading rules based on ML predictions, market regimes, and backtesting results.
        
        #### Key Components:
        
        **Entry Rules**  
        When to enter trades based on ML predictions, confidence levels, and market context.
        
        **Position Sizing**  
        How much capital to allocate to each trade based on prediction confidence and risk level.
        
        **Exit Rules**  
        When to exit trades using stop losses, take profits, trailing stops, and opposing signals.
        
        **Risk Management**  
        Rules to protect capital and manage drawdowns across your portfolio.
        
        #### How Strategies Are Optimized:
        
        - Backtest results identify which models perform best in different market regimes
        - Stop loss and take profit levels are adjusted based on historical volatility
        - Position sizing is scaled based on model confidence and historical accuracy
        - Entry thresholds are optimized based on model prediction magnitude
        
        #### Strategy Simulation:
        
        Strategy simulation shows how the trading rules would have performed on recent historical data, including:
        
        - Trade-by-trade P&L analysis
        - Equity curve visualization
        - Win/loss ratio and average trade metrics
        - Maximum drawdown and other risk metrics
        
        Remember that past performance does not guarantee future results, and all trading strategies involve risk.
        """)

def render_continuous_learning_subtab():
    """
    Render the Continuous Learning subtab
    """
    st.subheader("Continuous Model Learning")
    
    st.write("""
    Continuous learning allows models to automatically retrain with new data at regular intervals.
    This keeps predictions up-to-date with the latest market patterns and trends.
    """)
    
    # Sidebar for continuous learning controls
    st.sidebar.header("Continuous Learning Settings")
    
    # Get available symbols
    available_symbols = get_available_symbols(limit=5)
    
    # Symbol selection (multi-select)
    selected_symbols = st.sidebar.multiselect(
        "Select Cryptocurrencies",
        options=available_symbols,
        default=["BTCUSDT"],
        key="ml_cont_symbols"
    )
    
    # Interval selection (multi-select)
    timeframe_options = get_timeframe_options()
    intervals = list(timeframe_options.keys())
    selected_intervals = st.sidebar.multiselect(
        "Select Timeframes",
        options=intervals,
        default=["1h"],
        key="ml_cont_intervals"
    )
    
    # Convert to Binance intervals
    binance_intervals = [timeframe_to_interval(interval) for interval in selected_intervals]
    
    # Retraining interval
    retraining_hours = st.sidebar.slider(
        "Retraining Interval (hours)",
        min_value=1,
        max_value=48,
        value=24,
        help="How often to retrain models with new data"
    )
    
    # Initialize session state for continuous learning
    if 'continuous_learning_active' not in st.session_state:
        st.session_state.continuous_learning_active = False
        st.session_state.continuous_learning_thread = None
    
    # Status indicator
    st.write("### Continuous Learning Status")
    
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        if st.session_state.continuous_learning_active:
            st.success("Continuous learning is active")
        else:
            st.warning("Continuous learning is not active")
    
    with status_col2:
        if 'continuous_learning_start_time' in st.session_state:
            st.info(f"Started at: {st.session_state.continuous_learning_start_time.strftime('%Y-%m-%d %H:%M')}")
    
    # Controls
    control_col1, control_col2 = st.columns(2)
    
    with control_col1:
        if not st.session_state.continuous_learning_active:
            if st.button("Start Continuous Learning"):
                if not selected_symbols:
                    st.error("Please select at least one cryptocurrency.")
                elif not selected_intervals:
                    st.error("Please select at least one timeframe.")
                else:
                    # Define continuous learning thread function
                    def continuous_learning_thread():
                        try:
                            continuous_learning_cycle(
                                selected_symbols,
                                binance_intervals,
                                retraining_interval_hours=retraining_hours
                            )
                        except Exception as e:
                            logging.error(f"Continuous learning error: {e}")
                            st.session_state.continuous_learning_active = False
                    
                    # Start continuous learning thread
                    st.session_state.continuous_learning_thread = threading.Thread(
                        target=continuous_learning_thread
                    )
                    st.session_state.continuous_learning_thread.daemon = True
                    st.session_state.continuous_learning_thread.start()
                    
                    # Update state
                    st.session_state.continuous_learning_active = True
                    st.session_state.continuous_learning_start_time = datetime.now()
                    st.session_state.continuous_learning_symbols = selected_symbols
                    st.session_state.continuous_learning_intervals = selected_intervals
                    
                    st.success("Continuous learning started successfully!")
                    st.rerun()  # Refresh UI
    
    with control_col2:
        if st.session_state.continuous_learning_active:
            if st.button("Stop Continuous Learning"):
                # Set flag to stop (thread will exit on next iteration)
                st.session_state.continuous_learning_active = False
                st.session_state.continuous_learning_thread = None
                
                st.info("Continuous learning will stop after current cycle completes.")
                st.rerun()  # Refresh UI
    
    # Display active learning info
    if st.session_state.continuous_learning_active:
        st.subheader("Active Learning Configuration")
        
        st.write(f"**Cryptocurrencies:** {', '.join(st.session_state.continuous_learning_symbols)}")
        st.write(f"**Timeframes:** {', '.join(st.session_state.continuous_learning_intervals)}")
        st.write(f"**Retraining Interval:** Every {retraining_hours} hours")
        
        # Next retraining time estimate
        if 'continuous_learning_start_time' in st.session_state:
            next_time = st.session_state.continuous_learning_start_time + timedelta(hours=retraining_hours)
            st.write(f"**Next retraining cycle:** Approximately {next_time.strftime('%Y-%m-%d %H:%M')}")
    
    # Information about continuous learning
    with st.expander("About Continuous Learning"):
        st.write("""
        ### How Continuous Learning Works
        
        The continuous learning process automatically retrains models at the specified interval
        using the latest market data. This helps the model adapt to changing market conditions
        and improves prediction accuracy over time.
        
        ### Benefits
        
        - **Adaptation:** Models adapt to changing market patterns and regimes
        - **Improved Accuracy:** Performance gradually improves as more data becomes available
        - **Reduced Drift:** Prevents model performance degradation due to market changes
        
        ### Recommended Settings
        
        - For short timeframes (15m, 1h): Retrain every 24 hours
        - For longer timeframes (4h, 1d): Retrain every 2-3 days
        - Include both short-term (BTCUSDT, ETHUSDT) and trending coins for better insights
        """)


def get_available_symbols(limit=None):
    """
    Get list of available symbols for selection
    
    Args:
        limit: Optional limit on number of symbols
        
    Returns:
        List of symbol strings
    """
    from binance_api import get_available_symbols as get_symbols
    
    symbols = get_symbols()
    
    if limit and len(symbols) > limit:
        # Ensure BTC and ETH are included if they were originally present
        priority_symbols = ["BTCUSDT", "ETHUSDT"]
        filtered_symbols = [s for s in priority_symbols if s in symbols]
        
        # Add other symbols up to the limit
        other_symbols = [s for s in symbols if s not in priority_symbols]
        filtered_symbols.extend(other_symbols[:limit-len(filtered_symbols)])
        
        return filtered_symbols
    
    return symbols