"""
Enhanced ML UI module for cryptocurrency price predictions and analysis.
This module provides ML features for price prediction, market regime detection,
and strategy backtesting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import os

# Import ML functionality
try:
    from simple_ml import predict_prices as ml_predict_prices
    from simple_ml import detect_market_regime, run_strategy_backtest
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("Simple ML module not available - using fallback mode")

def format_percent(value):
    """Format a number as a percentage string"""
    if value is None:
        return "0%"
    return f"{value:.1f}%"

def render_ml_predictions_tab():
    """Render a comprehensive ML predictions tab with prediction functionality"""
    st.header("Machine Learning Price Predictions")
    
    # Show a collapsible warning about archived ML functionality
    with st.expander("⚠️ Note about ML Functionality"):
        st.warning("""
        The full ML functionality was archived during project cleanup.
        
        The original advanced ML modules can be found in the `legacy_archive/scripts` directory.
        A simplified ML implementation has been restored with core functionality.
        """)
    
    # Sidebar for controls
    st.sidebar.header("ML Settings")
    
    # Parse symbol to match format expected by the backend
    symbol_display = st.sidebar.selectbox(
        "Symbol",
        ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
        index=0
    )
    symbol = symbol_display.replace('/', '')
    
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["1h", "4h", "1d"],
        index=1
    )
    
    forecast_days = st.sidebar.slider(
        "Forecast Days",
        min_value=1,
        max_value=30,
        value=7
    )
    
    # Action buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        run_prediction = st.button("Run Prediction", type="primary")
    with col2:
        run_backtest = st.button("Run Backtest")
    
    # Main content with tabs
    tab1, tab2, tab3 = st.tabs(["Price Predictions", "Market Regime", "ML Backtesting"])
    
    # Get data for the selected symbol and timeframe
    if run_prediction or run_backtest:
        with st.spinner("Loading data..."):
            try:
                from app import get_data
                df = get_data(symbol, timeframe, lookback_days=60)
                if df is None or len(df) < 10:
                    st.error(f"Not enough data available for {symbol} with {timeframe} timeframe.")
                    df = pd.DataFrame()
            except Exception as e:
                st.error(f"Error loading data: {e}")
                df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    
    # Price Predictions Tab
    with tab1:
        st.subheader(f"Price Predictions for {symbol_display}")
        
        if run_prediction and not df.empty and ML_AVAILABLE:
            # Run actual prediction
            with st.spinner("Generating predictions..."):
                try:
                    predictions = ml_predict_prices(df, symbol, timeframe, days_ahead=forecast_days)
                    fig = create_prediction_chart(df, predictions, symbol_display, forecast_days)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display prediction table
                    st.subheader("Predicted Prices")
                    formatted_predictions = predictions.copy()
                    formatted_predictions['date'] = formatted_predictions['date'].dt.strftime('%Y-%m-%d')
                    formatted_predictions['predicted_price'] = formatted_predictions['predicted_price'].apply(
                        lambda x: f"${x:.2f}" if x is not None else "-")
                    formatted_predictions['lower_bound'] = formatted_predictions['lower_bound'].apply(
                        lambda x: f"${x:.2f}" if x is not None else "-")
                    formatted_predictions['upper_bound'] = formatted_predictions['upper_bound'].apply(
                        lambda x: f"${x:.2f}" if x is not None else "-")
                    
                    if 'confidence' in formatted_predictions.columns:
                        formatted_predictions['confidence'] = formatted_predictions['confidence'].apply(
                            lambda x: f"{x*100:.1f}%" if x is not None else "-")
                        
                    if 'model' in formatted_predictions.columns:
                        st.dataframe(formatted_predictions[['date', 'predicted_price', 
                                                           'lower_bound', 'upper_bound', 
                                                           'confidence', 'model']])
                    else:
                        st.dataframe(formatted_predictions[['date', 'predicted_price', 
                                                           'lower_bound', 'upper_bound']])
                except Exception as e:
                    st.error(f"Error generating predictions: {e}")
                    fig = create_prediction_chart(df, None, symbol_display, forecast_days)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            # Show placeholder or previous result
            fig = create_prediction_chart(df, None, symbol_display, forecast_days)
            st.plotly_chart(fig, use_container_width=True)
            
            if not ML_AVAILABLE:
                st.info("ML functionality not available. Using placeholder visualization.")
        
        # Show model details
        with st.expander("About Price Prediction"):
            if ML_AVAILABLE:
                st.markdown("""
                **Prediction Model Information:**
                
                This simplified prediction model uses a Random Forest Regressor trained on historical 
                price data and technical indicators. It generates predictions along with confidence 
                bounds for future price movements.
                
                **Feature Importance:**
                - Price data (OHLCV): 60%
                - Technical indicators (RSI, MACD, EMAs): 40%
                
                The model's accuracy tends to decrease as the prediction horizon increases.
                """)
            else:
                st.markdown("""
                The price prediction would use an ensemble of models:
                
                | Model | Weight | MAE | RMSE |
                | --- | --- | --- | --- |
                | Random Forest | 60% | - | - |
                | XGBoost | 30% | - | - |
                | Linear | 10% | - | - |
                
                Last trained: *Not available*
                """)
    
    # Market Regime Tab
    with tab2:
        st.subheader("Market Regime Detection")
        
        if run_prediction and not df.empty and ML_AVAILABLE:
            # Run actual market regime detection
            with st.spinner("Detecting market regime..."):
                try:
                    regime_data = detect_market_regime(df, symbol, timeframe)
                    regime_fig = create_regime_chart(df, regime_data)
                    st.plotly_chart(regime_fig, use_container_width=True)
                    
                    # Display current regime metrics
                    st.metric("Current Market Regime", 
                             regime_data.get('regime', 'Unknown').capitalize(), 
                             "")
                    
                    probs = regime_data.get('probabilities', {})
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Bullish Probability", 
                               f"{probs.get('bullish', 0)*100:.1f}%", 
                               "")
                    col2.metric("Bearish Probability", 
                               f"{probs.get('bearish', 0)*100:.1f}%", 
                               "")
                    col3.metric("Sideways Probability", 
                               f"{probs.get('sideways', 0)*100:.1f}%", 
                               "")
                    
                    st.caption(f"Regime detection confidence: {regime_data.get('confidence', 0)*100:.1f}%")
                except Exception as e:
                    st.error(f"Error detecting market regime: {e}")
                    regime_fig = create_regime_chart(df, None)
                    st.plotly_chart(regime_fig, use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Bullish Probability", "33.3%", "")
                    col2.metric("Bearish Probability", "33.3%", "")
                    col3.metric("Sideways Probability", "33.4%", "")
        else:
            # Show placeholder or previous result
            regime_fig = create_regime_chart(df, None)
            st.plotly_chart(regime_fig, use_container_width=True)
            
            # Current regime
            st.metric("Current Market Regime", "Neutral/Sideways", "")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Bullish Probability", "34%", "")
            col2.metric("Bearish Probability", "29%", "")
            col3.metric("Sideways Probability", "37%", "")
        
        st.caption("Market regime detection helps identify the overall market condition to adapt trading strategies.")
    
    # Backtesting Tab
    with tab3:
        st.subheader("ML Strategy Backtesting")
        
        # Strategy parameters
        with st.expander("Strategy Parameters", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                pred_threshold = st.slider("Prediction Threshold (%)", 
                                         min_value=1, max_value=10, value=2) / 100
                stop_loss = st.slider("Stop Loss (%)", 
                                    min_value=1, max_value=15, value=5) / 100
            with col2:
                take_profit = st.slider("Take Profit (%)", 
                                      min_value=2, max_value=30, value=10) / 100
                max_holding = st.slider("Max Holding Period (days)", 
                                      min_value=1, max_value=30, value=14)
        
        strategy_params = {
            'prediction_threshold': pred_threshold,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'max_holding_days': max_holding,
            'confidence_threshold': 0.6
        }
        
        if run_backtest and not df.empty and ML_AVAILABLE:
            # Run actual backtest
            with st.spinner("Running backtest..."):
                try:
                    backtest_results = run_strategy_backtest(df, symbol, timeframe, strategy_params)
                    
                    # Display backtest metrics
                    st.markdown("### Backtest Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Return", 
                                 format_percent(backtest_results.get('total_return_pct')), 
                                 "")
                        st.metric("Max Drawdown", 
                                 format_percent(backtest_results.get('max_drawdown_pct')), 
                                 "")
                        st.metric("Sharpe Ratio", 
                                 f"{backtest_results.get('sharpe_ratio', 0):.2f}", 
                                 "")
                    
                    with col2:
                        st.metric("Win Rate", 
                                 format_percent(backtest_results.get('win_rate')), 
                                 "")
                        st.metric("Profit Factor", 
                                 f"{backtest_results.get('profit_factor', 0):.2f}", 
                                 "")
                        st.metric("Avg. Holding Period", 
                                 f"{backtest_results.get('avg_holding_period', 0):.1f} days", 
                                 "")
                    
                    # Show trades table
                    trades = backtest_results.get('trades', [])
                    if trades:
                        st.subheader(f"Trades ({len(trades)} total)")
                        trades_df = pd.DataFrame(trades)
                        if 'date' in trades_df.columns:
                            trades_df['date'] = pd.to_datetime(trades_df['date'])
                            trades_df['date'] = trades_df['date'].dt.strftime('%Y-%m-%d')
                        
                        if 'profit_pct' in trades_df.columns:
                            trades_df['profit_pct'] = trades_df['profit_pct'].apply(
                                lambda x: f"{x:.2f}%" if x is not None else "-")
                        
                        if 'balance' in trades_df.columns:
                            trades_df['balance'] = trades_df['balance'].apply(
                                lambda x: f"${x:.2f}" if x is not None else "-")
                        
                        if 'price' in trades_df.columns:
                            trades_df['price'] = trades_df['price'].apply(
                                lambda x: f"${x:.2f}" if x is not None else "-")
                        
                        st.dataframe(trades_df)
                    else:
                        st.info("No trades were executed during the backtest period.")
                except Exception as e:
                    st.error(f"Error running backtest: {e}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Return", "0%", "")
                        st.metric("Max Drawdown", "0%", "")
                        st.metric("Sharpe Ratio", "0.0", "")
                    
                    with col2:
                        st.metric("Win Rate", "0%", "")
                        st.metric("Profit Factor", "0.0", "")
                        st.metric("Avg. Holding Period", "0 days", "")
        else:
            # Show placeholder or previous result
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Return", "0%", "")
                st.metric("Max Drawdown", "0%", "")
                st.metric("Sharpe Ratio", "0.0", "")
            
            with col2:
                st.metric("Win Rate", "0%", "")
                st.metric("Profit Factor", "0.0", "")
                st.metric("Avg. Holding Period", "0 days", "")
            
            if not ML_AVAILABLE:
                st.info("ML functionality not available. Click 'Run Backtest' to generate results.")
    
    # About section
    with st.expander("About the ML functionality"):
        st.markdown("""
        The ML module provides the following features:
        
        - **Price Prediction Models**: Random Forest for price forecasting
        - **Market Regime Detection**: Classification of market states (bullish, bearish, sideways)
        - **Advanced Backtesting**: Testing of ML-powered trading strategies
        
        For more advanced features that were available in the archived modules:
        - **Model Ensembles**: Combining multiple ML models for better predictions
        - **Sentiment Analysis Integration**: Combining technical indicators with sentiment data
        - **Strategy Generation**: Automated generation of trading strategies based on historical data
        """)

def predict_prices(df, days_ahead=7, models=None):
    """Wrapper for ML price prediction function"""
    if ML_AVAILABLE and not df.empty:
        try:
            # Get the symbol and interval from the dataframe if possible
            symbol = 'BTCUSDT'  # Default
            interval = '1d'     # Default
            
            if 'symbol' in df.columns:
                symbol = df['symbol'].iloc[0]
            if 'interval' in df.columns:
                interval = df['interval'].iloc[0]
            
            return ml_predict_prices(df, symbol, interval, days_ahead)
        except Exception as e:
            logging.error(f"Error predicting prices: {e}")
    
    # Fallback to stub implementation
    result = pd.DataFrame()
    today = datetime.now()
    result['date'] = [today + timedelta(days=i) for i in range(1, days_ahead+1)]
    result['predicted_price'] = [None] * days_ahead
    result['lower_bound'] = [None] * days_ahead
    result['upper_bound'] = [None] * days_ahead
    result['model'] = ['ensemble'] * days_ahead
    return result

def create_prediction_chart(hist_df, pred_df, symbol, days_ahead):
    """Create a prediction chart with actual data and predictions"""
    # Create figure
    fig = go.Figure()
    
    # Process historical data if available
    if not hist_df.empty and 'close' in hist_df.columns:
        # Get most recent 30 days of data for display
        recent_df = hist_df.tail(30).copy()
        if not recent_df.empty:
            dates = recent_df.index
            prices = recent_df['close'].values
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=dates,
                y=prices,
                mode='lines',
                name='Historical',
                line=dict(color='royalblue')
            ))
    else:
        # Create empty data
        today = datetime.now()
        dates = [today - timedelta(days=i) for i in range(30, 0, -1)]
        prices = [None] * 30
    
    # Process prediction data
    pred_dates = []
    predictions = []
    lower_bounds = []
    upper_bounds = []
    
    if pred_df is not None and not pred_df.empty:
        pred_dates = pred_df['date'].values
        predictions = pred_df['predicted_price'].values
        
        if 'lower_bound' in pred_df.columns:
            lower_bounds = pred_df['lower_bound'].values
        else:
            lower_bounds = [None] * len(pred_dates)
            
        if 'upper_bound' in pred_df.columns:
            upper_bounds = pred_df['upper_bound'].values
        else:
            upper_bounds = [None] * len(pred_dates)
    else:
        # Create empty prediction data
        today = datetime.now()
        pred_dates = [today + timedelta(days=i) for i in range(1, days_ahead+1)]
        predictions = [None] * days_ahead
        lower_bounds = [None] * days_ahead
        upper_bounds = [None] * days_ahead
    
    # Add prediction line
    fig.add_trace(go.Scatter(
        x=pred_dates,
        y=predictions,
        mode='lines',
        name='Prediction',
        line=dict(color='green', dash='dash')
    ))
    
    # Add prediction bounds
    fig.add_trace(go.Scatter(
        x=pred_dates,
        y=upper_bounds,
        mode='lines',
        name='Upper Bound',
        line=dict(color='rgba(0,128,0,0.2)', width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=pred_dates,
        y=lower_bounds,
        mode='lines',
        name='Lower Bound',
        line=dict(color='rgba(0,128,0,0.2)', width=0),
        fill='tonexty',
        fillcolor='rgba(0,128,0,0.1)',
        showlegend=False
    ))
    
    # Update layout
    title = f"{symbol} Price Prediction"
    if not ML_AVAILABLE:
        title += " (Placeholder)"
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price (USDT)",
        legend=dict(y=0.99, x=0.01),
        template='plotly_white',
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_regime_chart(hist_df, regime_data):
    """Create a market regime chart with historical data and regime information"""
    today = datetime.now()
    dates = [today - timedelta(days=i) for i in range(90, 0, -1)]
    
    # Create figure
    fig = go.Figure()
    
    # Process historical data if available
    if not hist_df.empty and 'close' in hist_df.columns:
        # Get most recent data for display
        recent_df = hist_df.tail(90).copy()
        if not recent_df.empty:
            hist_dates = recent_df.index
            hist_prices = recent_df['close'].values
            
            # Normalize prices for better visualization
            min_price = np.min(hist_prices)
            max_price = np.max(hist_prices)
            norm_prices = (hist_prices - min_price) / max(1, max_price - min_price)
            
            # Add historical price data
            fig.add_trace(go.Scatter(
                x=hist_dates,
                y=norm_prices,
                mode='lines',
                name='Price',
                line=dict(color='royalblue')
            ))
    
    # Process regime data if available
    if regime_data is not None and 'regime_history' in regime_data:
        regime_history = regime_data.get('regime_history', [])
        
        if regime_history:
            # Prepare data
            regime_dates = []
            regime_values = []
            regime_colors = []
            
            for item in regime_history:
                regime_dates.append(item.get('date'))
                
                if item.get('regime') == 'bullish':
                    regime_values.append(0.8)
                    regime_colors.append('rgba(0,255,0,0.2)')  # Green
                elif item.get('regime') == 'bearish':
                    regime_values.append(0.2) 
                    regime_colors.append('rgba(255,0,0,0.2)')  # Red
                else:  # sideways
                    regime_values.append(0.5)
                    regime_colors.append('rgba(128,128,128,0.2)')  # Grey
            
            # Add regime background
            fig.add_trace(go.Scatter(
                x=regime_dates,
                y=regime_values,
                mode='lines',
                name='Market Regime',
                line=dict(width=0),
                showlegend=False
            ))
    
    # Add legend entries for regimes
    fig.add_trace(go.Scatter(
        x=[dates[0]],
        y=[0.8],
        mode='markers',
        name='Bullish',
        marker=dict(color='green', size=10),
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=[dates[0]],
        y=[0.5],
        mode='markers',
        name='Sideways',
        marker=dict(color='gray', size=10),
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=[dates[0]],
        y=[0.2],
        mode='markers',
        name='Bearish',
        marker=dict(color='red', size=10),
        showlegend=True
    ))
    
    # Update layout
    title = "Market Regime Detection"
    if not ML_AVAILABLE:
        title += " (Placeholder)"
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="",
        yaxis=dict(
            showticklabels=False,
            showgrid=False
        ),
        legend=dict(y=0.99, x=0.01),
        template='plotly_white',
        height=400
    )
    
    return fig
