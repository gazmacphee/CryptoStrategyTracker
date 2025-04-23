"""
Database extensions module that adds missing functions needed for ML processing.
This module imports from database.py and adds additional functionality.
"""

import database
from database import get_db_connection
import pandas as pd

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
    try:
        # First try to import from binance_api
        from binance_api import get_historical_data as binance_get_historical_data
        
        print(f"Getting historical data for {symbol}/{interval} using binance_api function")
        return binance_get_historical_data(symbol, interval, lookback_days, start_date, end_date)
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
            # Fix the query to use timestamp instead of open_time
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