"""
Enhanced ML UI module for graceful fallback when ML features are not available.
This module provides stub implementations of ML features to prevent errors
and displays informative placeholders.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

def render_ml_predictions_tab():
    """Render a comprehensive ML predictions tab with information about archived ML modules"""
    st.header("Machine Learning Price Predictions")
    
    # Status info about ML modules
    st.warning("""
    ⚠️ ML functionality has been moved to the legacy archive during project cleanup.
    
    The original ML modules can be found in the `legacy_archive/scripts` directory:
    - ml_backtesting.py
    - ml_ensemble.py
    - ml_market_regime.py
    - ml_prediction.py
    - ml_sentiment_integration.py
    - ml_trading_strategy.py
    """)
    
    # Restoration instructions
    st.info("""
    To restore ML functionality:
    1. Copy the ML modules from legacy_archive/scripts back to the main directory
    2. Update imports in app.py as needed
    """)
    
    # Sidebar for controls
    st.sidebar.header("ML Settings")
    symbol = st.sidebar.selectbox(
        "Symbol",
        ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
        index=0
    )
    
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
    
    # Main content with tabs
    tab1, tab2, tab3 = st.tabs(["Price Predictions", "Market Regime", "ML Backtesting"])
    
    with tab1:
        st.subheader(f"Price Predictions for {symbol}")
        
        # Display placeholder chart
        fig = create_prediction_chart(symbol, forecast_days)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show model details
        with st.expander("Model Details"):
            st.markdown("""
            The price prediction would use an ensemble of models:
            
            | Model | Weight | MAE | RMSE |
            | --- | --- | --- | --- |
            | LSTM | 40% | - | - |
            | XGBoost | 30% | - | - |
            | Prophet | 20% | - | - |
            | Linear | 10% | - | - |
            
            Last trained: *Not available*
            """)
    
    with tab2:
        st.subheader("Market Regime Detection")
        
        # Display placeholder regime chart
        regime_fig = create_regime_chart()
        st.plotly_chart(regime_fig, use_container_width=True)
        
        # Current regime
        st.metric("Current Market Regime", "Neutral/Sideways", "")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Bullish Probability", "34%", "")
        col2.metric("Bearish Probability", "29%", "")
        col3.metric("Sideways Probability", "37%", "")
        
        st.caption("Market regime detection helps identify the overall market condition to adapt trading strategies.")
    
    with tab3:
        st.subheader("ML Strategy Backtesting")
        
        # Show placeholder backtest results
        st.markdown("### Backtest Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Return", "0%", "")
            st.metric("Max Drawdown", "0%", "")
            st.metric("Sharpe Ratio", "0.0", "")
        
        with col2:
            st.metric("Win Rate", "0%", "")
            st.metric("Profit Factor", "0.0", "")
            st.metric("Avg. Holding Period", "0 days", "")
        
        st.markdown("Backtesting not available. Restore ML modules to enable this feature.")
    
    # About section
    with st.expander("About the ML functionality"):
        st.markdown("""
        The archived ML modules provided the following features:
        
        - **Price Prediction Models**: LSTM, XGBoost, and ensemble models for price forecasting
        - **Market Regime Detection**: Classification of market states (bullish, bearish, sideways)
        - **Sentiment Analysis Integration**: Combining technical indicators with sentiment data
        - **Advanced Backtesting**: Comprehensive testing of ML-powered trading strategies
        - **Strategy Generation**: Automated generation of trading strategies based on historical data
        """)

def predict_prices(df, days_ahead=7, models=None):
    """Stub for price prediction function that returns empty predictions"""
    # Return empty DataFrame with expected structure
    result = pd.DataFrame()
    today = datetime.now()
    result['date'] = [today + timedelta(days=i) for i in range(1, days_ahead+1)]
    result['predicted_price'] = [None] * days_ahead
    result['lower_bound'] = [None] * days_ahead
    result['upper_bound'] = [None] * days_ahead
    result['model'] = ['ensemble'] * days_ahead
    return result

def create_prediction_chart(symbol, days_ahead):
    """Create a placeholder prediction chart"""
    # Create empty data
    today = datetime.now()
    dates = [today - timedelta(days=i) for i in range(30, 0, -1)] + [today + timedelta(days=i) for i in range(1, days_ahead+1)]
    
    # Historical data (placeholder)
    historical_prices = [None] * 30
    
    # Prediction and bounds are None
    predictions = [None] * days_ahead
    lower_bounds = [None] * days_ahead
    upper_bounds = [None] * days_ahead
    
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=dates[:30],
        y=historical_prices,
        mode='lines',
        name='Historical',
        line=dict(color='royalblue')
    ))
    
    # Add prediction line
    fig.add_trace(go.Scatter(
        x=dates[30:],
        y=predictions,
        mode='lines',
        name='Prediction',
        line=dict(color='green', dash='dash')
    ))
    
    # Add prediction bounds
    fig.add_trace(go.Scatter(
        x=dates[30:],
        y=upper_bounds,
        mode='lines',
        name='Upper Bound',
        line=dict(color='rgba(0,128,0,0.2)', width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=dates[30:],
        y=lower_bounds,
        mode='lines',
        name='Lower Bound',
        line=dict(color='rgba(0,128,0,0.2)', width=0),
        fill='tonexty',
        fillcolor='rgba(0,128,0,0.1)',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} Price Prediction (ML feature archived)",
        xaxis_title="Date",
        yaxis_title="Price (USDT)",
        legend=dict(y=0.99, x=0.01),
        template='plotly_white',
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_regime_chart():
    """Create a placeholder market regime chart"""
    # Create empty data for the chart
    today = datetime.now()
    dates = [today - timedelta(days=i) for i in range(90, 0, -1)]
    
    # Create figure
    fig = go.Figure()
    
    # Dummy regimes (0=bear, 1=neutral, 2=bull)
    regimes = [None] * 90
    
    # Add shaded areas for regimes
    fig.add_trace(go.Scatter(
        x=dates,
        y=[0] * 90,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title="Market Regime Detection (ML feature archived)",
        xaxis_title="Date",
        yaxis_title="",
        yaxis=dict(
            showticklabels=False,
            showgrid=False
        ),
        legend=dict(y=0.99, x=0.01),
        template='plotly_white',
        height=300
    )
    
    return fig
