"""
Utilities module for data fetching and processing shared between different UI modules.
This avoids circular imports between app.py and specialized UI modules.
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
from binance_api import get_klines_data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_data(symbol, interval, lookback_days=30, start_date=None, end_date=None):
    """
    Fetch data for specified symbol and interval
    
    Args:
        symbol: Trading pair symbol (e.g., BTCUSDT)
        interval: Time interval for candles
        lookback_days: Default number of days to look back
        start_date: Optional specific start date (overrides lookback_days)
        end_date: Optional specific end date (defaults to now)
    """
    try:
        # If start_date is provided as string, convert to datetime
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        
        # If end_date is provided as string, convert to datetime
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
        # If no start_date provided, calculate from lookback_days
        if start_date is None:
            start_date = datetime.now() - timedelta(days=lookback_days)
            
        # If no end_date provided, use current time
        if end_date is None:
            end_date = datetime.now()
            
        # Get data from API
        df = get_klines_data(symbol, interval, start_date, end_date)
        
        return df
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None