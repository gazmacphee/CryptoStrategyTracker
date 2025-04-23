"""
Economic Indicators Module

This module provides functionality to fetch and store economic indicators data such as:
1. US Dollar Index (DXY) - Measures the value of the US dollar against a basket of foreign currencies
2. Global Liquidity Metrics - Measures of money supply and credit availability worldwide

These indicators can be used to analyze correlations with cryptocurrency prices.
"""

import os
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv
from decimal import Decimal

import database
from database import get_db_connection, save_dxy_data, save_liquidity_data, get_liquidity_data

def ensure_float_values(df):
    """
    Ensures all numeric columns in a DataFrame are converted to float type.
    This prevents issues with decimal.Decimal values.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        DataFrame with all numeric columns as float
    """
    if df is None or df.empty:
        return df
        
    for col in df.select_dtypes(include=['number']).columns:
        # Check if column contains any Decimal objects
        has_decimal = False
        for x in df[col].dropna().head():
            if isinstance(x, Decimal):
                has_decimal = True
                break
                
        if has_decimal:
            df[col] = df[col].astype(float)
    
    return df

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# FRED API key for economic data
# Make sure to strip any whitespace or special characters
FRED_API_KEY = os.environ.get('FRED_API_KEY', '').strip()

# Alpha Vantage API key for financial data
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', '').strip()

# Yahoo Finance is used as a fallback (no API key required)

# Constants
DXY_SYMBOL = 'DX-Y.NYB'  # Yahoo Finance symbol for US Dollar Index
GLOBAL_LIQUIDITY_INDICATORS = {
    'M2': 'WM2NS',  # M2 Money Supply (US) - FRED series ID
    'WALCL': 'WALCL',  # Federal Reserve Total Assets - FRED series ID
    'BOGMBASE': 'BOGMBASE',  # Monetary Base - FRED series ID
}

def create_economic_indicator_tables():
    """Create tables to store economic indicator data if they don't exist"""
    conn = get_db_connection()
    
    with conn.cursor() as cursor:
        # Table for US Dollar Index (DXY)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS dollar_index (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            close DECIMAL(18,8) NOT NULL,
            open DECIMAL(18,8),
            high DECIMAL(18,8),
            low DECIMAL(18,8),
            volume DECIMAL(18,8),
            UNIQUE(timestamp)
        );
        """)
        
        # Table for global liquidity indicators
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS global_liquidity (
            id SERIAL PRIMARY KEY,
            indicator_name VARCHAR(50) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            value DECIMAL(22,8) NOT NULL,
            UNIQUE(indicator_name, timestamp)
        );
        """)
    
    conn.commit()
    conn.close()
    
    logger.info("Economic indicator tables created successfully")

def fetch_dxy_data_yahoo(start_date=None, end_date=None):
    """
    Fetch US Dollar Index (DXY) data from Yahoo Finance
    
    Args:
        start_date: Start date for data (datetime or string YYYY-MM-DD)
        end_date: End date for data (datetime or string YYYY-MM-DD)
        
    Returns:
        DataFrame with DXY data
    """
    try:
        # Default to last 5 years if no dates provided
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Convert to string format if datetime objects
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
            
        # Convert dates to UNIX timestamps (Yahoo Finance API format)
        start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        
        # Yahoo Finance API URL
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{DXY_SYMBOL}"
        params = {
            "period1": start_timestamp,
            "period2": end_timestamp,
            "interval": "1d",  # Daily data
            "events": "history"
        }
        
        # Add user agent to avoid blocking
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        data = response.json()
        
        # Parse the response
        chart_data = data['chart']['result'][0]
        timestamps = chart_data['timestamp']
        quote = chart_data['indicators']['quote'][0]
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': [datetime.fromtimestamp(ts) for ts in timestamps],
            'open': quote['open'],
            'high': quote['high'],
            'low': quote['low'],
            'close': quote['close'],
            'volume': quote.get('volume', [None] * len(timestamps))
        })
        
        # Clean up any NaN values
        df = df.dropna(subset=['close'])
        
        logger.info(f"Successfully fetched DXY data from Yahoo Finance, {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching DXY data from Yahoo Finance: {e}")
        return pd.DataFrame()

def fetch_dxy_data_alpha_vantage():
    """
    Fetch US Dollar Index data from Alpha Vantage API
    This is used as a backup method if Yahoo Finance fails
    
    Returns:
        DataFrame with DXY data
    """
    if not ALPHA_VANTAGE_API_KEY:
        logger.warning("Alpha Vantage API key not found. Cannot fetch DXY data from Alpha Vantage.")
        return pd.DataFrame()
    
    try:
        # Alpha Vantage API endpoint for forex data (using EUR/USD as proxy for DXY)
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'FX_DAILY',
            'from_symbol': 'USD',
            'to_symbol': 'EUR',  # Inverse of EUR/USD is a rough proxy for DXY
            'outputsize': 'full',
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Check if we received valid data
        if 'Time Series FX (Daily)' not in data:
            logger.error(f"Invalid data format from Alpha Vantage: {data}")
            return pd.DataFrame()
        
        # Parse time series data
        time_series = data['Time Series FX (Daily)']
        records = []
        
        for date, values in time_series.items():
            # For USD/EUR, we need to invert to get something similar to DXY trend
            close_value = 1 / float(values['4. close'])
            records.append({
                'timestamp': datetime.strptime(date, '%Y-%m-%d'),
                'close': close_value,
                'open': 1 / float(values['1. open']),
                'high': 1 / float(values['2. high']),
                'low': 1 / float(values['3. low']),
                # No volume data available for forex
            })
        
        df = pd.DataFrame(records)
        logger.info(f"Successfully fetched USD/EUR data from Alpha Vantage, {len(df)} records")
        
        # Note: This is not exactly DXY, but it follows similar trends
        return df
        
    except Exception as e:
        logger.error(f"Error fetching USD/EUR data from Alpha Vantage: {e}")
        return pd.DataFrame()

def fetch_fred_data(series_id, start_date=None, end_date=None):
    """
    Fetch economic data from FRED (Federal Reserve Economic Data)
    
    Args:
        series_id: FRED series identifier (e.g., 'WM2NS' for M2 Money Supply)
        start_date: Start date for data (YYYY-MM-DD)
        end_date: End date for data (YYYY-MM-DD)
        
    Returns:
        DataFrame with the time series data
    """
    if not FRED_API_KEY:
        logger.warning(f"FRED API key not found. Cannot fetch {series_id} data from FRED.")
        return pd.DataFrame()
    
    try:
        # Default to last 10 years if no dates provided
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # FRED API endpoint
        url = 'https://api.stlouisfed.org/fred/series/observations'
        params = {
            'series_id': series_id,
            'api_key': FRED_API_KEY,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date,
            'frequency': 'm',  # Monthly data
            'units': 'lin'  # Linear units (not percent change)
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Parse observations
        observations = data.get('observations', [])
        records = []
        
        for obs in observations:
            # Skip missing values
            if obs['value'] == '.':
                continue
                
            records.append({
                'timestamp': datetime.strptime(obs['date'], '%Y-%m-%d'),
                'value': float(obs['value'])
            })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df['indicator_name'] = series_id
            logger.info(f"Successfully fetched {series_id} data from FRED, {len(df)} records")
        else:
            logger.warning(f"No data fetched for {series_id} from FRED")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching {series_id} data from FRED: {e}")
        return pd.DataFrame()

def save_dxy_to_database(df):
    """
    Save US Dollar Index data to database using the centralized database function
    
    Args:
        df: DataFrame with DXY data
        
    Returns:
        Boolean indicating success or number of records saved
    """
    if df.empty:
        logger.warning("No DXY data to save to database")
        return 0
    
    try:
        # Use the centralized database function to save DXY data
        success = save_dxy_data(df)
        
        if success:
            logger.info(f"Successfully saved {len(df)} DXY data points")
            return len(df)
        else:
            logger.error("Failed to save DXY data")
            return 0
    except Exception as e:
        logger.error(f"Error saving DXY data: {e}")
        return 0

def save_liquidity_to_database(df):
    """
    Save global liquidity indicator data to database using centralized database function
    
    Args:
        df: DataFrame with global liquidity data 
            (must have indicator_name, timestamp, and value columns)
            
    Returns:
        Boolean indicating success or number of records saved
    """
    if df.empty:
        logger.warning("No liquidity data to save to database")
        return 0
    
    try:
        # Use the centralized database function to save liquidity data
        success = save_liquidity_data(df)
        
        if success:
            logger.info(f"Successfully saved {len(df)} liquidity data points")
            return len(df)
        else:
            logger.error("Failed to save liquidity data")
            return 0
    except Exception as e:
        logger.error(f"Error saving liquidity data: {e}")
        return 0

def get_dxy_data_from_api(start_date=None, end_date=None):
    """
    Get US Dollar Index data directly from external API
    
    Args:
        start_date: Start date for data (datetime or string YYYY-MM-DD)
        end_date: End date for data (datetime or string YYYY-MM-DD)
        
    Returns:
        DataFrame with DXY data
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Default dates if not provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)
    if end_date is None:
        end_date = datetime.now()
    
    # Try Yahoo Finance first
    df = fetch_dxy_data_yahoo(start_date, end_date)
    
    # If Yahoo fails, try Alpha Vantage
    if df.empty and ALPHA_VANTAGE_API_KEY:
        df = fetch_dxy_data_alpha_vantage()
    
    # Save to database if we got data
    if not df.empty:
        save_dxy_data(df)
    
    return df

def get_dxy_data_from_database(start_date=None, end_date=None):
    """
    Get US Dollar Index data directly from database using centralized function
    
    Args:
        start_date: Start date for data (datetime or string YYYY-MM-DD)
        end_date: End date for data (datetime or string YYYY-MM-DD)
        
    Returns:
        DataFrame with DXY data
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Default dates if not provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)
    if end_date is None:
        end_date = datetime.now()
    
    # Use the aliased database function that we imported
    from database import get_dxy_data as db_get_dxy_data
    df = db_get_dxy_data(start_time=start_date, end_time=end_date)
    
    # Log the result
    if not df.empty:
        logger.info(f"Retrieved {len(df)} DXY records from database")
    else:
        logger.warning("No DXY records found in database for specified date range")
    
    return df

def get_dxy_data(start_date=None, end_date=None):
    """
    Get US Dollar Index data from database or fetch from external source if missing
    
    Args:
        start_date: Start date for data (datetime or string YYYY-MM-DD)
        end_date: End date for data (datetime or string YYYY-MM-DD)
        
    Returns:
        DataFrame with DXY data
    """
    # This is the main function that should be used by the UI
    return get_full_dxy_data(start_date, end_date)

def get_full_dxy_data(start_date=None, end_date=None):
    """
    Get US Dollar Index data from database or fetch from external source if missing
    
    Args:
        start_date: Start date for data (datetime or string YYYY-MM-DD)
        end_date: End date for data (datetime or string YYYY-MM-DD)
        
    Returns:
        DataFrame with DXY data
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Default dates if not provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)
    if end_date is None:
        end_date = datetime.now()
    
    # First try to get data from database
    # Call the database function using the parameter names it expects (start_time, end_time)
    df = get_dxy_data_from_database(start_date=start_date, end_date=end_date)
    
    # If we have no data or incomplete data, fetch from external source
    if df.empty or len(df) < (end_date - start_date).days * 0.7:  # If less than 70% of expected daily data
        logger.info(f"Insufficient DXY data in database, fetching from external source")
        api_df = get_dxy_data_from_api(start_date, end_date)
        
        if not api_df.empty:
            # Combine with existing data
            df = pd.concat([df, api_df]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    
    logger.info(f"Retrieved {len(df)} DXY records")
        # Convert any decimal.Decimal columns to float
    df = ensure_float_values(df)
    
    return df

def get_liquidity_data_from_database(indicator=None, start_date=None, end_date=None):
    """
    Get global liquidity data directly from database
    
    Args:
        indicator: Specific indicator name (or None for all)
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        DataFrame with liquidity data
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Default dates if not provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365*5)  # 5 years
    if end_date is None:
        end_date = datetime.now()
    
    # Access database directly to avoid circular reference
    conn = get_db_connection()
    df = pd.DataFrame()
    
    try:
        # Build query based on whether a specific indicator was requested
        query = """
        SELECT indicator_name, timestamp, value
        FROM global_liquidity
        WHERE timestamp BETWEEN %s AND %s
        """
        params = [start_date, end_date]
        
        if indicator:
            query += " AND indicator_name = %s"
            params.append(indicator)
        
        query += " ORDER BY indicator_name, timestamp"
        
        df = pd.read_sql_query(query, conn, params=params)
        
        # Log the result
        if not df.empty:
            logger.info(f"Retrieved {len(df)} liquidity records from database")
        else:
            logger.warning("No liquidity records found in database for specified criteria")
    except Exception as e:
        logger.error(f"Error getting liquidity data from database: {e}")
    finally:
        if conn:
            conn.close()
    
    return df

def get_liquidity_data_from_fred(indicator=None, start_date=None, end_date=None):
    """
    Fetch global liquidity data from FRED API
    
    Args:
        indicator: Specific indicator name (or None for all)
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        DataFrame with liquidity data
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Default dates if not provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365*5)  # 5 years
    if end_date is None:
        end_date = datetime.now()
    
    all_data = []
    indicators_to_fetch = [indicator] if indicator else GLOBAL_LIQUIDITY_INDICATORS.keys()
    
    for ind in indicators_to_fetch:
        if ind in GLOBAL_LIQUIDITY_INDICATORS:
            series_id = GLOBAL_LIQUIDITY_INDICATORS[ind]
            df = fetch_fred_data(series_id, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            if not df.empty:
                all_data.append(df)
                # Save to database
                save_liquidity_data(df)
    
    if all_data:
        return pd.concat(all_data)
    else:
        return pd.DataFrame()

def get_liquidity_data(indicator=None, start_date=None, end_date=None):
    """
    Get global liquidity data from database or fetch from external source if missing
    
    Args:
        indicator: Specific indicator name (or None for all)
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        DataFrame with liquidity data
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Default dates if not provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365*5)  # 5 years
    if end_date is None:
        end_date = datetime.now()
    
    # First try to get data from database
    df = get_liquidity_data_from_database(indicator, start_date, end_date)
    
    # If we have no data or incomplete data, fetch from external source
    indicators_to_check = [indicator] if indicator else GLOBAL_LIQUIDITY_INDICATORS.keys()
    need_external_data = False
    
    for ind in indicators_to_check:
        # Check if we have enough records for this indicator
        ind_df = df[df['indicator_name'] == ind] if not df.empty else pd.DataFrame()
        
        # Monthly data should have approximately 30% of days in range (allowing for weekends, holidays)
        expected_min_records = (end_date - start_date).days * 0.3 / 30  
        
        if len(ind_df) < expected_min_records:
            logger.info(f"Insufficient data for {ind}, fetching from FRED")
            need_external_data = True
            break
    
    if need_external_data:
        external_df = get_liquidity_data_from_fred(indicator, start_date, end_date)
        
        if not external_df.empty:
            # Re-query database to get all data
            df = get_liquidity_data_from_database(indicator, start_date, end_date)
    
        # Convert any decimal.Decimal columns to float
    df = ensure_float_values(df)
    
    return df

def update_economic_indicators():
    """
    Update all economic indicators in the database with latest data
    This function is intended to be run periodically (e.g., daily or weekly)
    """
    try:
        # Ensure tables exist
        create_economic_indicator_tables()
        
        # Update DXY - Get data for the last 6 months
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        dxy_df = fetch_dxy_data_yahoo(start_date, end_date)
        if not dxy_df.empty:
            # Use centralized database function to save DXY data
            save_dxy_data(dxy_df)
        
        # Update liquidity indicators
        for indicator, series_id in GLOBAL_LIQUIDITY_INDICATORS.items():
            df = fetch_fred_data(series_id, start_date, end_date)
            if not df.empty:
                # Use centralized database function to save liquidity data
                save_liquidity_data(df)
                
        logger.info("Economic indicators updated successfully")
        return True
    except Exception as e:
        logger.error(f"Error updating economic indicators: {e}")
        return False

def calculate_correlation(crypto_df, dxy_df=None, liquidity_df=None):
    """
    Calculate correlation between cryptocurrency prices and economic indicators
    
    Args:
        crypto_df: DataFrame with cryptocurrency prices
        dxy_df: DataFrame with DXY data (optional)
        liquidity_df: DataFrame with liquidity data (optional)
        
    Returns:
        DataFrame with correlation coefficients
    """
    if crypto_df.empty:
        return pd.DataFrame()
    
    # Ensure we have required columns in crypto_df
    if 'timestamp' not in crypto_df.columns or 'close' not in crypto_df.columns:
        logger.error("Crypto DataFrame must have 'timestamp' and 'close' columns")
        return pd.DataFrame()
    
    # Set timestamp as index for easier joining
    crypto_df = crypto_df.set_index('timestamp')
    
    # Create results dataframe
    results = []
    
    # Calculate DXY correlation if provided
    if dxy_df is not None and not dxy_df.empty:
        dxy_df = dxy_df.set_index('timestamp')
        
        # Align data on same dates
        joined = crypto_df[['close']].join(dxy_df[['close']], how='inner', lsuffix='_crypto', rsuffix='_dxy')
        
        if not joined.empty and len(joined) > 5:  # Need at least a few data points
            # Convert to float to avoid Decimal type issues
            correlation = joined['close_crypto'].astype(float).corr(joined['close_dxy'].astype(float))
            results.append({
                'indicator': 'US Dollar Index (DXY)',
                'correlation': correlation,
                'data_points': len(joined)
            })
    
    # Calculate liquidity correlations if provided
    if liquidity_df is not None and not liquidity_df.empty:
        # Group by indicator
        for indicator, group in liquidity_df.groupby('indicator_name'):
            group = group.set_index('timestamp')
            
            # Align data on same dates
            joined = crypto_df[['close']].join(group[['value']], how='inner')
            
            if not joined.empty and len(joined) > 5:
                # Convert to float to avoid Decimal type issues
                correlation = joined['close'].astype(float).corr(joined['value'].astype(float))
                results.append({
                    'indicator': indicator,
                    'correlation': correlation,
                    'data_points': len(joined)
                })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Create necessary tables
    create_economic_indicator_tables()
    
    # Update economic indicators
    update_economic_indicators()