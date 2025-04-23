"""
Database extensions module that adds missing functions needed for ML processing.
This module imports from database.py and adds additional functionality.
"""

import database
from database import get_db_connection
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal

# Helper functions for type conversion
def safe_float_convert(value):
    """Convert Decimal values to float safely"""
    if isinstance(value, Decimal):
        return float(value)
    return value

def ensure_float_df(df, columns=None):
    """Ensure specified columns in DataFrame contain only float values, not Decimal"""
    if df is None or df.empty:
        return df
        
    if columns is None:
        # Get all numeric columns
        numeric_columns = df.select_dtypes(include=['number', 'object']).columns
        columns = [col for col in numeric_columns]
    
    # Apply the conversion to each column
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(safe_float_convert)
    
    return df

def get_symbols_from_database(limit=None):
    """
    Get a list of all unique symbols in the database
    
    Args:
        limit: Optional limit on the number of symbols to return
    
    Returns:
        List of symbol strings
    """
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM historical_data")
            symbols = [row[0] for row in cursor.fetchall()]
            
            # Sort symbols and apply limit if provided
            symbols.sort()
            if limit and limit < len(symbols):
                symbols = symbols[:limit]
                
            return symbols
        except Exception as e:
            print(f"Error getting symbols from database: {e}")
            return []
        finally:
            conn.close()
    return []


def get_available_symbols(quote_asset="USDT", limit=30):
    """
    Get available trading pairs, prioritizing ones in our database
    Similar to the function in binance_api.py but directly from database

    Args:
        quote_asset: Quote currency (e.g., "USDT")
        limit: Maximum number of symbols to return

    Returns:
        List of symbol strings
    """
    # Default popular symbols to return if DB access fails
    popular_symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT",
        "SOLUSDT", "DOTUSDT", "LTCUSDT", "LINKUSDT", "MATICUSDT", "AVAXUSDT",
        "ATOMUSDT", "UNIUSDT", "FILUSDT", "AAVEUSDT", "NEARUSDT", "ALGOUSDT",
        "ICPUSDT", "XTZUSDT", "AXSUSDT", "FTMUSDT", "SANDUSDT", "MANAUSDT",
        "HBARUSDT", "THETAUSDT", "EGLDUSDT", "FLOWUSDT", "APUSDT", "CAKEUSDT"
    ]
    
    # First try to get symbols from our database
    db_symbols = get_symbols_from_database()
    
    if db_symbols:
        # Filter for the specified quote asset
        filtered_symbols = [s for s in db_symbols if s.endswith(quote_asset)]
        
        # Prioritize popular symbols that we have data for
        available_popular = [s for s in popular_symbols if s in filtered_symbols]
        remaining = [s for s in filtered_symbols if s not in popular_symbols]
        
        # Combine and limit
        result = available_popular + remaining
        if limit and limit < len(result):
            return result[:limit]
        return result
    
    # Fallback to default popular symbols
    return popular_symbols[:limit]

# Add a function to get historical data directly from database or binance_api
def get_historical_data(symbol, interval, lookback_days=30, start_date=None, end_date=None):
    """
    Get historical data for a symbol and interval.
    This is a wrapper that either uses binance_api's get_historical_data or gets data directly from the database.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '1d')
        lookback_days: Number of days to look back
        start_date: Optional specific start date (overrides lookback_days)
        end_date: Optional specific end date (defaults to now)
        
    Returns:
        DataFrame with OHLCV and timestamp data
    """
    print(f"====== DATABASE_EXTENSIONS: get_historical_data called for {symbol}/{interval} =======")
    
    # Check if we need to use 30m data for 1h interval
    use_30m = False
    if interval == '1h':
        # Query db to see if we have 1h data
        conn = database.get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM historical_data WHERE symbol = %s AND interval = %s", (symbol, interval))
                count = cursor.fetchone()[0]
                print(f"Found {count} records for {symbol}/{interval}")
                
                if count == 0:
                    # No 1h data found, check for 30m data
                    cursor.execute("SELECT COUNT(*) FROM historical_data WHERE symbol = %s AND interval = '30m'", (symbol,))
                    count_30m = cursor.fetchone()[0]
                    print(f"Found {count_30m} records for {symbol}/30m")
                    
                    if count_30m > 0:
                        print(f"Will use 30m data and resample to 1h")
                        use_30m = True
                        # Adjust interval for query
                        interval = '30m'
            except Exception as e:
                print(f"Error checking database for data: {e}")
            finally:
                conn.close()
    
    # Let's directly query the database for 30m data and resample it
    if interval == '1h':
        from datetime import datetime, timedelta
        import pandas as pd
        
        # Calculate date range if not provided
        if end_date is None:
            end_time = datetime.now()
        else:
            end_time = end_date if isinstance(end_date, datetime) else datetime.strptime(end_date, '%Y-%m-%d')
        
        if start_date is not None:
            start_time = start_date if isinstance(start_date, datetime) else datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start_time = end_time - timedelta(days=lookback_days)
            
        # Format timestamps for SQL query
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        conn = database.get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                # Query 30m data directly - first check if we have any data
                check_query = "SELECT COUNT(*) FROM historical_data WHERE symbol = %s AND interval = '30m'"
                cursor.execute(check_query, (symbol,))
                count = cursor.fetchone()[0]
                print(f"Found {count} total records of 30m data for {symbol}")
                
                if count > 0:
                    # Just get all the data and then filter in Python for simplicity (to avoid timestamp formatting issues)
                    query = """
                        SELECT * FROM historical_data 
                        WHERE symbol = %s AND interval = '30m'
                        ORDER BY timestamp ASC
                    """
                    cursor.execute(query, (symbol,))
                    rows = cursor.fetchall()
                    
                    # Get column names
                    columns = [desc[0] for desc in cursor.description]
                    
                    # Create DataFrame
                    df_30m = pd.DataFrame(rows, columns=columns)
                    
                    if not df_30m.empty:
                        print(f"Retrieved {len(df_30m)} rows of 30m data for {symbol}")
                        
                        # Convert any Decimal values to float
                        df_30m = ensure_float_df(df_30m)
                        
                        # Convert to datetime
                        if 'timestamp' in df_30m.columns:
                            df_30m['timestamp'] = pd.to_datetime(df_30m['timestamp'])
                        
                        # Filter to our date range if needed
                        if len(df_30m) > lookback_days * 48:  # 48 30-minute periods per day 
                            # Filter to the date range we want
                            df_30m = df_30m[(df_30m['timestamp'] >= start_time) & (df_30m['timestamp'] <= end_time)]
                            print(f"Filtered to {len(df_30m)} rows in date range")
                    
                    # Set timestamp as index for resampling
                    df_30m = df_30m.set_index('timestamp')
                    
                    # Resample to 1h
                    df_1h = df_30m.resample('1H').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })
                    
                    # Reset index to get timestamp as column again
                    df_1h = df_1h.reset_index()
                    
                    # Restore the original column order and add any missing columns
                    for col in columns:
                        if col not in df_1h.columns and col not in ['id', 'created_at']:
                            df_1h[col] = None
                    
                    # Set the interval to 1h
                    if 'interval' in df_1h.columns:
                        df_1h['interval'] = '1h'
                        
                    print(f"Successfully resampled 30m data to 1h: {len(df_1h)} rows")
                    return df_1h
            except Exception as e:
                print(f"Error querying 30m data: {e}")
            finally:
                conn.close()
                
    # If we couldn't convert 30m data to 1h, fall back to the original method
    try:
        # Now try to import from binance_api
        from binance_api import get_historical_data as binance_get_historical_data
        
        print(f"Falling back to binance_api.get_historical_data for {symbol}/{interval}")
        df = binance_get_historical_data(symbol, interval, lookback_days, start_date, end_date)
        
        if df is not None and not df.empty:
            print(f"Successfully retrieved {len(df)} rows from binance_api.get_historical_data")
        else:
            print(f"No data retrieved from binance_api.get_historical_data")
            
        return df
    except Exception as e:
        print(f"Error using binance_api.get_historical_data: {e}")
        print(f"Falling back to database query for {symbol}/{interval}")
        
        # Fall back to direct database query
        from datetime import datetime, timedelta
        import pandas as pd
        
        conn = database.get_db_connection()
        if conn is None:
            print("Failed to connect to database")
            return pd.DataFrame()
        
        try:
            # Calculate date range
            if end_date is None:
                end_time = datetime.now()
            else:
                end_time = end_date if isinstance(end_date, datetime) else datetime.strptime(end_date, '%Y-%m-%d')
            
            if start_date is not None:
                start_time = start_date if isinstance(start_date, datetime) else datetime.strptime(start_date, '%Y-%m-%d')
            else:
                start_time = end_time - timedelta(days=lookback_days)
            
            # Convert to timestamps for SQL query
            start_timestamp = int(start_time.timestamp() * 1000)
            end_timestamp = int(end_time.timestamp() * 1000)
            
            # Query the database
            cursor = conn.cursor()
            # Use a simplified query first to check what's in the database
            check_query = "SELECT COUNT(*) FROM historical_data WHERE symbol = %s AND interval = %s"
            cursor.execute(check_query, (symbol, interval))
            count = cursor.fetchone()[0]
            print(f"Found {count} records for {symbol}/{interval} in database")
            
            # Now query with date range
            query = """
                SELECT * FROM historical_data 
                WHERE symbol = %s AND interval = %s 
                AND timestamp BETWEEN to_timestamp(%s/1000) AND to_timestamp(%s/1000)
                ORDER BY timestamp ASC
            """
            cursor.execute(query, (symbol, interval, start_timestamp, end_timestamp))
            rows = cursor.fetchall()
            
            # Get column names from cursor
            columns = [desc[0] for desc in cursor.description]
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=columns)
            
            # Log the data retrieval results
            if df.empty:
                print(f"No data found in database for {symbol}/{interval} between {start_time} and {end_time}")
                # Print the query for debugging
                print(f"Query: {query}")
                print(f"Parameters: {symbol}, {interval}, {start_timestamp}, {end_timestamp}")
            else:
                print(f"Retrieved {len(df)} rows from database for {symbol}/{interval}")
            
            # Make sure timestamp is in datetime format
            if not df.empty and 'timestamp' in df.columns:
                # Convert to pandas datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as db_err:
            print(f"Error fetching data from database: {db_err}")
            return pd.DataFrame()
        finally:
            conn.close()

# Add the functions to the database module namespace
database.get_symbols_from_database = get_symbols_from_database
database.get_available_symbols = get_available_symbols
database.get_historical_data = get_historical_data