"""
UI Components for Machine Learning Predictions

This module provides Streamlit UI components for:
1. Training ML models
2. Displaying predictions
3. Showing model performance metrics
4. Managing continuous learning
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import threading
import os
import glob
import joblib
import logging

# Import local modules
from ml_prediction import MLPredictor, train_all_models, predict_for_all, continuous_learning_cycle

def render_ml_predictions_tab():
    """
    Render the ML Predictions tab in the Streamlit UI
    """
    st.header("Machine Learning Price Predictions")
    
    # Create tabs for different ML functionalities
    ml_tabs = st.tabs(["Predictions", "Model Training", "Performance Metrics", "Continuous Learning"])
    
    # Initialize session state for ML
    if 'ml_training_running' not in st.session_state:
        st.session_state.ml_training_running = False
    if 'ml_continuous_running' not in st.session_state:
        st.session_state.ml_continuous_running = False
    if 'ml_predictions' not in st.session_state:
        st.session_state.ml_predictions = {}
    if 'selected_symbol_interval' not in st.session_state:
        st.session_state.selected_symbol_interval = ('BTCUSDT', '1h')
    
    # Tab 1: Predictions
    with ml_tabs[0]:
        render_predictions_subtab()
    
    # Tab 2: Model Training
    with ml_tabs[1]:
        render_training_subtab()
    
    # Tab 3: Performance Metrics
    with ml_tabs[2]:
        render_metrics_subtab()
    
    # Tab 4: Continuous Learning
    with ml_tabs[3]:
        render_continuous_learning_subtab()

def render_predictions_subtab():
    """
    Render the Predictions subtab
    """
    st.subheader("Price Movement Predictions")
    
    # Symbol and interval selection
    col1, col2 = st.columns(2)
    with col1:
        symbols = get_available_symbols()
        symbol = st.selectbox("Select Symbol", symbols, index=symbols.index(st.session_state.selected_symbol_interval[0]) if st.session_state.selected_symbol_interval[0] in symbols else 0)
    
    with col2:
        intervals = ['15m', '30m', '1h', '4h', '1d']
        interval = st.selectbox("Select Interval", intervals, index=intervals.index(st.session_state.selected_symbol_interval[1]) if st.session_state.selected_symbol_interval[1] in intervals else 2)
    
    st.session_state.selected_symbol_interval = (symbol, interval)
    
    # Prediction horizon selection
    col1, col2 = st.columns(2)
    with col1:
        horizon = st.selectbox("Prediction Horizon", [1, 6, 12, 24, 48, 96], index=2)
    
    with col2:
        predict_button = st.button("Generate Prediction")
    
    # Get model status
    model_filename = f"models/{symbol}_{interval}_predictor.joblib"
    model_exists = os.path.exists(model_filename)
    
    if model_exists:
        try:
            model_data = joblib.load(model_filename)
            last_updated = datetime.fromisoformat(model_data.get('updated_at', '2000-01-01T00:00:00'))
            st.info(f"Model last updated: {last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
        except:
            st.warning("Model file exists but could not be loaded properly.")
    else:
        st.warning("No trained model exists for this symbol/interval. Please train a model first.")
    
    # Generate predictions when button is clicked
    if predict_button:
        with st.spinner(f"Generating predictions for {symbol}/{interval}..."):
            try:
                predictor = MLPredictor(symbol, interval, prediction_horizon=horizon)
                
                if not model_exists:
                    st.info("Training new model since none exists...")
                    success = predictor.train_model()
                    if not success:
                        st.error("Failed to train model. Please check logs.")
                        return
                
                predictions = predictor.predict_future(periods=horizon)
                if predictions is not None and not predictions.empty:
                    st.session_state.ml_predictions[(symbol, interval)] = predictions
                    st.success("Predictions generated successfully!")
                else:
                    st.error("Failed to generate predictions. Please check logs.")
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
    
    # Display predictions if available
    if (symbol, interval) in st.session_state.ml_predictions:
        predictions = st.session_state.ml_predictions[(symbol, interval)]
        
        # Show prediction summary
        st.subheader("Prediction Summary")
        avg_movement = predictions['predicted_movement'].mean()
        direction = "bullish 📈" if avg_movement > 0 else "bearish 📉"
        confidence = predictions['confidence'].mean()
        
        # Format metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Price Direction", direction)
        col2.metric("Predicted Change", f"{avg_movement:.2%}")
        col3.metric("Confidence", f"{confidence:.2%}")
        
        # Display the prediction table
        st.subheader("Detailed Predictions")
        display_df = predictions.copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['predicted_movement'] = display_df['predicted_movement'].apply(lambda x: f"{x:.2%}")
        display_df['predicted_price'] = display_df['predicted_price'].apply(lambda x: f"${x:.2f}")
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2%}")
        display_df.columns = ['Timestamp', 'Predicted Movement', 'Predicted Price', 'Confidence']
        st.dataframe(display_df)
        
        # Create prediction chart
        st.subheader("Price Prediction Chart")
        
        # Get historical data for context
        from database import get_historical_data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=3)  # 3 days of historical data
        historical_df = get_historical_data(symbol, interval, start_time, end_time)
        
        if not historical_df.empty:
            # Create plotly figure
            fig = go.Figure()
            
            # Add historical prices
            fig.add_trace(go.Scatter(
                x=historical_df['timestamp'],
                y=historical_df['close'],
                mode='lines',
                name='Historical Price',
                line=dict(color='blue')
            ))
            
            # Add prediction
            last_price = historical_df['close'].iloc[-1]
            last_timestamp = historical_df['timestamp'].iloc[-1]
            
            # Create prediction line
            pred_timestamps = [last_timestamp] + predictions['timestamp'].tolist()
            pred_prices = [last_price] + predictions['predicted_price'].tolist()
            
            fig.add_trace(go.Scatter(
                x=pred_timestamps,
                y=pred_prices,
                mode='lines+markers',
                name='Predicted Price',
                line=dict(color='green', dash='dot'),
                marker=dict(size=8)
            ))
            
            # Add confidence interval
            upper_bound = []
            lower_bound = []
            
            for i, row in predictions.iterrows():
                error_margin = (1 - row['confidence']) * row['predicted_price'] * 0.1
                upper_bound.append(row['predicted_price'] + error_margin)
                lower_bound.append(max(0, row['predicted_price'] - error_margin))
            
            fig.add_trace(go.Scatter(
                x=predictions['timestamp'],
                y=upper_bound,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=predictions['timestamp'],
                y=lower_bound,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0, 255, 0, 0.1)',
                name='Confidence Interval'
            ))
            
            # Update layout
            fig.update_layout(
                title=f"{symbol} Price Prediction ({interval})",
                xaxis_title="Date/Time",
                yaxis_title="Price (USDT)",
                hovermode="x unified",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No historical data available to display chart context.")

def render_training_subtab():
    """
    Render the Model Training subtab
    """
    st.subheader("Train Prediction Models")
    
    # Options for training
    col1, col2 = st.columns(2)
    
    with col1:
        train_option = st.radio(
            "Training Scope",
            ["Single Model", "Multiple Models", "All Available Models"],
            index=0
        )
    
    with col2:
        lookback_days = st.slider("Training Data Lookback (days)", 
                                min_value=30, 
                                max_value=365, 
                                value=90, 
                                step=30)
        retrain = st.checkbox("Retrain Existing Models", value=True)
    
    # Different options based on selection
    if train_option == "Single Model":
        # Symbol and interval selection
        col1, col2 = st.columns(2)
        with col1:
            symbols = get_available_symbols()
            symbol = st.selectbox("Symbol", symbols, key="train_single_symbol")
        
        with col2:
            intervals = ['15m', '30m', '1h', '4h', '1d']
            interval = st.selectbox("Interval", intervals, key="train_single_interval")
        
        train_button = st.button("Train Model")
        
        if train_button:
            with st.spinner(f"Training model for {symbol}/{interval}..."):
                try:
                    predictor = MLPredictor(symbol, interval)
                    success = predictor.train_model(lookback_days=lookback_days, retrain=retrain)
                    
                    if success:
                        st.success(f"Successfully trained model for {symbol}/{interval}")
                    else:
                        st.error(f"Failed to train model for {symbol}/{interval}. Check logs for details.")
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
    
    elif train_option == "Multiple Models":
        # Multiple symbol and interval selection
        col1, col2 = st.columns(2)
        with col1:
            all_symbols = get_available_symbols()
            symbols = st.multiselect("Symbols", all_symbols, default=all_symbols[:3])
        
        with col2:
            all_intervals = ['15m', '30m', '1h', '4h', '1d']
            intervals = st.multiselect("Intervals", all_intervals, default=['1h', '4h'])
        
        if not symbols:
            st.warning("Please select at least one symbol.")
        elif not intervals:
            st.warning("Please select at least one interval.")
        else:
            train_button = st.button("Train Selected Models")
            
            if train_button:
                if not st.session_state.ml_training_running:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Set flag to indicate training is running
                    st.session_state.ml_training_running = True
                    
                    # Start training in a separate thread
                    def train_models_thread():
                        try:
                            total_models = len(symbols) * len(intervals)
                            completed = 0
                            
                            results = {}
                            for symbol in symbols:
                                results[symbol] = {}
                                for interval in intervals:
                                    if not st.session_state.ml_training_running:
                                        break
                                        
                                    status_text.text(f"Training model for {symbol}/{interval}...")
                                    predictor = MLPredictor(symbol, interval)
                                    success = predictor.train_model(lookback_days=lookback_days, retrain=retrain)
                                    results[symbol][interval] = success
                                    
                                    completed += 1
                                    progress_bar.progress(completed / total_models)
                                    
                                if not st.session_state.ml_training_running:
                                    break
                            
                            if st.session_state.ml_training_running:
                                # Count successes and failures
                                successes = sum(1 for sym in results for intv in results[sym] if results[sym][intv])
                                failures = total_models - successes
                                
                                status_text.text(f"Training completed. {successes} models successful, {failures} failed.")
                            else:
                                status_text.text("Training cancelled.")
                                
                            # Reset flag
                            st.session_state.ml_training_running = False
                            
                        except Exception as e:
                            status_text.text(f"Error in training: {str(e)}")
                            st.session_state.ml_training_running = False
                    
                    # Start the training thread
                    threading.Thread(target=train_models_thread).start()
                else:
                    st.warning("Training is already running.")
            
            if st.session_state.ml_training_running:
                if st.button("Cancel Training"):
                    st.session_state.ml_training_running = False
                    st.warning("Cancelling training... This may take a moment.")
    
    else:  # All Available Models
        train_button = st.button("Train All Available Models")
        
        if train_button:
            if not st.session_state.ml_training_running:
                symbols = get_available_symbols()
                intervals = ['15m', '30m', '1h', '4h', '1d']
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Set flag to indicate training is running
                st.session_state.ml_training_running = True
                
                # Start training in a separate thread
                def train_all_models_thread():
                    try:
                        total_models = len(symbols) * len(intervals)
                        completed = 0
                        
                        results = {}
                        for symbol in symbols:
                            results[symbol] = {}
                            for interval in intervals:
                                if not st.session_state.ml_training_running:
                                    break
                                    
                                status_text.text(f"Training model for {symbol}/{interval}...")
                                predictor = MLPredictor(symbol, interval)
                                success = predictor.train_model(lookback_days=lookback_days, retrain=retrain)
                                results[symbol][interval] = success
                                
                                completed += 1
                                progress_bar.progress(completed / total_models)
                                
                            if not st.session_state.ml_training_running:
                                break
                        
                        if st.session_state.ml_training_running:
                            # Count successes and failures
                            successes = sum(1 for sym in results for intv in results[sym] if results[sym][intv])
                            failures = total_models - successes
                            
                            status_text.text(f"Training completed. {successes} models successful, {failures} failed.")
                        else:
                            status_text.text("Training cancelled.")
                            
                        # Reset flag
                        st.session_state.ml_training_running = False
                        
                    except Exception as e:
                        status_text.text(f"Error in training: {str(e)}")
                        st.session_state.ml_training_running = False
                
                # Start the training thread
                threading.Thread(target=train_all_models_thread).start()
            else:
                st.warning("Training is already running.")
        
        if st.session_state.ml_training_running:
            if st.button("Cancel Training"):
                st.session_state.ml_training_running = False
                st.warning("Cancelling training... This may take a moment.")

def render_metrics_subtab():
    """
    Render the Performance Metrics subtab
    """
    st.subheader("Model Performance Metrics")
    
    # Get all available models
    model_files = glob.glob("models/*_predictor.joblib")
    
    if not model_files:
        st.warning("No trained models found. Train models first to view metrics.")
        return
    
    # Extract symbols and intervals from model filenames
    model_info = []
    for file in model_files:
        try:
            filename = os.path.basename(file)
            parts = filename.replace("_predictor.joblib", "").split("_")
            if len(parts) >= 2:
                symbol = parts[0]
                interval = parts[1]
                
                # Load model to get metrics
                model_data = joblib.load(file)
                metrics_history = model_data.get('metrics_history', [])
                
                if metrics_history:
                    latest_metrics = metrics_history[-1]['metrics']
                    updated_at = datetime.fromisoformat(model_data.get('updated_at', '2000-01-01T00:00:00'))
                    
                    model_info.append({
                        'symbol': symbol,
                        'interval': interval,
                        'mse': latest_metrics.get('mse', 0),
                        'mae': latest_metrics.get('mae', 0),
                        'r2': latest_metrics.get('r2', 0),
                        'correct_direction': latest_metrics.get('correct_direction', 0),
                        'updated_at': updated_at
                    })
        except Exception as e:
            logging.error(f"Error loading model metrics from {file}: {e}")
    
    if not model_info:
        st.warning("Could not load metrics from models. Try retraining models.")
        return
    
    # Convert to DataFrame for display
    metrics_df = pd.DataFrame(model_info)
    
    # Add formatting
    display_df = metrics_df.copy()
    display_df['mse'] = display_df['mse'].apply(lambda x: f"{x:.6f}")
    display_df['mae'] = display_df['mae'].apply(lambda x: f"{x:.6f}")
    display_df['r2'] = display_df['r2'].apply(lambda x: f"{x:.4f}")
    display_df['correct_direction'] = display_df['correct_direction'].apply(lambda x: f"{x:.2%}")
    display_df['updated_at'] = display_df['updated_at'].dt.strftime('%Y-%m-%d %H:%M')
    
    # Rename columns for display
    display_df.columns = ['Symbol', 'Interval', 'MSE', 'MAE', 'R²', 'Direction Accuracy', 'Last Updated']
    
    # Sort by symbol and interval
    display_df = display_df.sort_values(['Symbol', 'Interval'])
    
    # Display metrics table
    st.dataframe(display_df, use_container_width=True)
    
    # Select model for detailed metrics
    st.subheader("Model Metrics History")
    
    col1, col2 = st.columns(2)
    with col1:
        symbols = sorted(metrics_df['symbol'].unique())
        symbol = st.selectbox("Symbol", symbols, key="metrics_symbol")
    
    with col2:
        intervals_for_symbol = sorted(metrics_df[metrics_df['symbol'] == symbol]['interval'].unique())
        interval = st.selectbox("Interval", intervals_for_symbol, key="metrics_interval")
    
    # Load detailed metrics history for selected model
    model_file = f"models/{symbol}_{interval}_predictor.joblib"
    if os.path.exists(model_file):
        try:
            model_data = joblib.load(model_file)
            metrics_history = model_data.get('metrics_history', [])
            
            if metrics_history:
                history_data = []
                for entry in metrics_history:
                    history_data.append({
                        'timestamp': entry['timestamp'],
                        'mse': entry['metrics'].get('mse', 0),
                        'mae': entry['metrics'].get('mae', 0),
                        'r2': entry['metrics'].get('r2', 0),
                        'correct_direction': entry['metrics'].get('correct_direction', 0)
                    })
                
                history_df = pd.DataFrame(history_data)
                
                # Create history chart
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('MSE', 'MAE', 'R²', 'Direction Accuracy'),
                    vertical_spacing=0.15
                )
                
                # Add MSE trace
                fig.add_trace(
                    go.Scatter(x=history_df['timestamp'], y=history_df['mse'], mode='lines+markers'),
                    row=1, col=1
                )
                
                # Add MAE trace
                fig.add_trace(
                    go.Scatter(x=history_df['timestamp'], y=history_df['mae'], mode='lines+markers'),
                    row=1, col=2
                )
                
                # Add R² trace
                fig.add_trace(
                    go.Scatter(x=history_df['timestamp'], y=history_df['r2'], mode='lines+markers'),
                    row=2, col=1
                )
                
                # Add direction accuracy trace
                fig.add_trace(
                    go.Scatter(x=history_df['timestamp'], y=history_df['correct_direction'], mode='lines+markers'),
                    row=2, col=2
                )
                
                # Update layout
                fig.update_layout(
                    title=f"{symbol}/{interval} Model Metrics History",
                    showlegend=False,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No metrics history available for this model.")
        except Exception as e:
            st.error(f"Error loading metrics history: {str(e)}")
    else:
        st.warning(f"Model file {model_file} not found.")

def render_continuous_learning_subtab():
    """
    Render the Continuous Learning subtab
    """
    st.subheader("Continuous Learning Management")
    
    st.write("""
    Continuous learning automatically retrains models at regular intervals using the latest data.
    This helps models adapt to changing market conditions and improve prediction accuracy over time.
    """)
    
    # Continuous learning configuration
    col1, col2 = st.columns(2)
    
    with col1:
        symbols = get_available_symbols()
        selected_symbols = st.multiselect("Symbols to Monitor", symbols, default=symbols[:3])
    
    with col2:
        intervals = ['15m', '30m', '1h', '4h', '1d']
        selected_intervals = st.multiselect("Intervals to Monitor", intervals, default=['1h', '4h'])
        retraining_interval = st.number_input("Retraining Interval (hours)", min_value=1, max_value=72, value=24)
    
    # Start/Stop continuous learning
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Continuous Learning"):
            if not st.session_state.ml_continuous_running:
                if not selected_symbols:
                    st.warning("Please select at least one symbol.")
                elif not selected_intervals:
                    st.warning("Please select at least one interval.")
                else:
                    st.session_state.ml_continuous_running = True
                    
                    # Start continuous learning in a separate thread
                    def continuous_learning_thread():
                        try:
                            st.session_state.ml_continuous_status = "Running"
                            st.session_state.ml_continuous_last_run = datetime.now()
                            
                            while st.session_state.ml_continuous_running:
                                # Run training cycle
                                results = {}
                                for symbol in selected_symbols:
                                    results[symbol] = {}
                                    for interval in selected_intervals:
                                        if not st.session_state.ml_continuous_running:
                                            break
                                            
                                        st.session_state.ml_continuous_current = f"{symbol}/{interval}"
                                        predictor = MLPredictor(symbol, interval)
                                        success = predictor.train_model(retrain=True)
                                        results[symbol][interval] = success
                                        
                                    if not st.session_state.ml_continuous_running:
                                        break
                                
                                # Update status
                                st.session_state.ml_continuous_last_run = datetime.now()
                                st.session_state.ml_continuous_results = results
                                
                                # Wait for next cycle
                                if st.session_state.ml_continuous_running:
                                    next_run = datetime.now() + timedelta(hours=retraining_interval)
                                    st.session_state.ml_continuous_next_run = next_run
                                    
                                    # Sleep in smaller chunks to allow for cancellation
                                    sleep_until = time.time() + retraining_interval * 3600
                                    while time.time() < sleep_until and st.session_state.ml_continuous_running:
                                        time.sleep(10)  # Check every 10 seconds
                            
                            st.session_state.ml_continuous_status = "Stopped"
                            
                        except Exception as e:
                            st.session_state.ml_continuous_status = f"Error: {str(e)}"
                            st.session_state.ml_continuous_running = False
                    
                    # Initialize status
                    st.session_state.ml_continuous_status = "Starting"
                    st.session_state.ml_continuous_current = "N/A"
                    st.session_state.ml_continuous_last_run = None
                    st.session_state.ml_continuous_next_run = None
                    st.session_state.ml_continuous_results = {}
                    
                    # Start the thread
                    threading.Thread(target=continuous_learning_thread).start()
            else:
                st.warning("Continuous learning is already running.")
    
    with col2:
        if st.button("Stop Continuous Learning"):
            if st.session_state.ml_continuous_running:
                st.session_state.ml_continuous_running = False
                st.warning("Stopping continuous learning... This may take a moment.")
            else:
                st.info("Continuous learning is not running.")
    
    # Show status
    st.subheader("Continuous Learning Status")
    
    if hasattr(st.session_state, 'ml_continuous_status'):
        status_color = {
            "Starting": "blue",
            "Running": "green",
            "Stopped": "orange"
        }
        
        color = status_color.get(st.session_state.ml_continuous_status, "red")
        
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<p style='color:{color}; font-weight:bold;'>Status: {st.session_state.ml_continuous_status}</p>", unsafe_allow_html=True)
        
        if st.session_state.ml_continuous_running:
            col2.write(f"Current model: {st.session_state.ml_continuous_current}")
            
            if st.session_state.ml_continuous_last_run:
                col3.write(f"Last run: {st.session_state.ml_continuous_last_run.strftime('%Y-%m-%d %H:%M')}")
            
            if st.session_state.ml_continuous_next_run:
                now = datetime.now()
                next_run = st.session_state.ml_continuous_next_run
                
                if next_run > now:
                    time_left = next_run - now
                    hours = int(time_left.total_seconds() // 3600)
                    minutes = int((time_left.total_seconds() % 3600) // 60)
                    
                    st.info(f"Next retraining cycle in {hours}h {minutes}m ({next_run.strftime('%Y-%m-%d %H:%M')})")
            
            # Show last results if available
            if st.session_state.ml_continuous_results:
                with st.expander("Last Training Cycle Results"):
                    results = st.session_state.ml_continuous_results
                    
                    # Count successes and failures
                    total = sum(1 for sym in results for intv in results[sym])
                    successes = sum(1 for sym in results for intv in results[sym] if results[sym][intv])
                    failures = total - successes
                    
                    st.write(f"Total models: {total}, Successful: {successes}, Failed: {failures}")
                    
                    # Display detailed results
                    details = []
                    for symbol in results:
                        for interval in results[symbol]:
                            details.append({
                                'Symbol': symbol,
                                'Interval': interval,
                                'Status': "✅ Success" if results[symbol][interval] else "❌ Failed"
                            })
                    
                    if details:
                        st.dataframe(pd.DataFrame(details))
    else:
        st.info("Continuous learning has not been started yet.")

def get_available_symbols(limit=None):
    """
    Get list of available symbols for selection
    
    Args:
        limit: Optional limit on number of symbols
        
    Returns:
        List of symbol strings
    """
    # Use function from existing modules if available
    try:
        from backfill_database import get_popular_symbols
        return get_popular_symbols(limit=limit)
    except:
        # Fallback to hardcoded popular symbols
        symbols = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT",
            "SOLUSDT", "DOTUSDT", "LTCUSDT", "LINKUSDT", "MATICUSDT", "AVAXUSDT"
        ]
        
        if limit and limit < len(symbols):
            return symbols[:limit]
        return symbols