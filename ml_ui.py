"""
Simplified ML UI module for graceful fallback when ML features are not available.
This module provides stub implementations of ML features to prevent errors.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

def render_ml_predictions_tab():
    """Render a simple ML predictions tab with information about archived ML modules"""
    st.header("Machine Learning Price Predictions")
    
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
    
    st.info("""
    To restore ML functionality:
    1. Copy the ML modules from legacy_archive/scripts back to the main directory
    2. Update imports in app.py as needed
    """)
    
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
    """Stub for price prediction function"""
    # Return empty DataFrame with expected structure
    result = pd.DataFrame()
    today = datetime.now()
    result['date'] = [today + timedelta(days=i) for i in range(1, days_ahead+1)]
    result['predicted_price'] = [None] * days_ahead
    result['lower_bound'] = [None] * days_ahead
    result['upper_bound'] = [None] * days_ahead
    return result
