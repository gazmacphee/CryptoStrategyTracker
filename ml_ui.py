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