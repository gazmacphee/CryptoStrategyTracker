"""
Utility module for fixing maximum recursion depth exceeded error in binance_api and database_extensions
"""

import pandas as pd
import numpy as np
from decimal import Decimal

def fix_recursion_depth_exceeded():
    """
    Apply a monkey patch to fix the recursion depth exceeded error.
    This is needed because database_extensions.get_historical_data and
    binance_api.get_historical_data call each other recursively, causing
    maximum recursion depth to be exceeded.
    """
    # Import the modules to patch
    import binance_api
    import database_extensions
    
    # Save the original function from binance_api
    original_binance_get_historical_data = binance_api.get_historical_data
    
    # Create a new function that doesn't redirect
    def new_binance_get_historical_data(symbol, interval, lookback_days=30, start_date=None, end_date=None):
        """
        Direct version of get_historical_data that doesn't redirect to database_extensions
        to avoid recursion issues.
        """
        # Skip the redirection to database_extensions and use the fallback code directly
        try:
            # Convert dates if provided
            start_time = None
            if start_date:
                if isinstance(start_date, str):
                    from datetime import datetime
                    start_time = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
                elif isinstance(start_date, datetime):
                    start_time = int(start_date.timestamp() * 1000)
            
            end_time = None    
            if end_date:
                if isinstance(end_date, str):
                    from datetime import datetime
                    end_time = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
                elif isinstance(end_date, datetime):
                    end_time = int(end_date.timestamp() * 1000)
            
            if not start_time and lookback_days:
                # Calculate start time based on lookback_days
                from datetime import datetime
                end_ts = end_time if end_time else int(datetime.now().timestamp() * 1000)
                start_time = end_ts - (lookback_days * 24 * 60 * 60 * 1000)
            
            # Call the get_klines_data function which doesn't redirect
            return binance_api.get_klines_data(symbol, interval, start_time, end_time)
            
        except Exception as e:
            print(f"Error in direct binance_api.get_historical_data: {e}")
            return pd.DataFrame()
    
    # Replace the function in binance_api with our new version
    binance_api.get_historical_data = new_binance_get_historical_data
    
    # Save the original function from database_extensions
    original_db_get_historical_data = database_extensions.get_historical_data
    
    # Create a new function that doesn't redirect
    def new_db_get_historical_data(symbol, interval, lookback_days=30, start_date=None, end_date=None):
        """
        Direct version of get_historical_data that doesn't redirect to binance_api
        to avoid recursion issues.
        """
        from datetime import datetime, timedelta
        import pandas as pd
        import database
        
        print(f"====== DATABASE_EXTENSIONS: get_historical_data called for {symbol}/{interval} =======")
        
        # Calculate date range
        if start_date is None:
            start_date = datetime.now() - timedelta(days=lookback_days)
        elif isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Try to get data from database first
        conn = database.get_db_connection()
        if conn is None:
            print("Could not connect to database")
            return pd.DataFrame()
            
        try:
            # Query the historical_data table directly, avoiding binance_api
            query = """
                SELECT * FROM historical_data
                WHERE symbol = %s AND interval = %s AND timestamp BETWEEN %s AND %s
                ORDER BY timestamp ASC
            """
            
            df = pd.read_sql_query(
                query, 
                conn, 
                params=(symbol, interval, start_date, end_date)
            )
            
            # Convert Decimal columns to float for calculations
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            
            if not df.empty:
                print(f"Found {len(df)} records for {symbol}/{interval}")
                return df
            else:
                print(f"No data found in database for {symbol}/{interval}")
                # Instead of falling back to binance_api, we return empty DataFrame
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching data from database: {e}")
            return pd.DataFrame()
            
        finally:
            conn.close()
    
    # Replace the function in database_extensions with our new version
    database_extensions.get_historical_data = new_db_get_historical_data

def convert_decimal_to_float(df):
    """
    Convert all Decimal columns in a DataFrame to float type
    
    Args:
        df: DataFrame that may contain Decimal columns
        
    Returns:
        DataFrame with Decimal columns converted to float
    """
    if df is None or df.empty:
        return df
        
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if any values in the column are Decimal
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(sample, Decimal):
                df[col] = df[col].astype(float)
    
    return df