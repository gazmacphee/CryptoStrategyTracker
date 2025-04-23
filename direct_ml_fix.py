"""
Direct fix for maximum recursion depth exceeded errors in ML modules.
This script completely breaks the circular dependency between modules
by implementing a standalone database-only data retrieval system for ML.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ml_training.log')
    ]
)

# Global patched functions dictionary to store original functions
_original_functions = {}

def get_historical_data_direct_from_db(symbol, interval, lookback_days=30, start_date=None, end_date=None):
    """
    STANDALONE function to get historical data directly from database.
    This function does NOT depend on any other module's get_historical_data function.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '1d')
        lookback_days: Number of days to look back
        start_date: Optional specific start date (overrides lookback_days)
        end_date: Optional specific end date (defaults to now)
        
    Returns:
        DataFrame with OHLCV and timestamp data or empty DataFrame if no data
    """
    # Import database only when needed, to avoid circular imports
    import database
    
    logging.info(f"ML STANDALONE DB QUERY for {symbol}/{interval}")
    
    # Calculate date range
    if start_date is None:
        start_date = datetime.now() - timedelta(days=lookback_days)
    elif isinstance(start_date, str):
        try:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        except ValueError:
            try:
                start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                logging.warning(f"Could not parse start_date: {start_date}, using lookback_days instead")
                start_date = datetime.now() - timedelta(days=lookback_days)
        
    if end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        try:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            try:
                end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                logging.warning(f"Could not parse end_date: {end_date}, using current time instead")
                end_date = datetime.now()
    
    # Connect to database directly
    conn = database.get_db_connection()
    if conn is None:
        logging.error("ML STANDALONE: Could not connect to database")
        return pd.DataFrame()
        
    try:
        # Query the historical_data table directly
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
        
        if not df.empty:
            logging.info(f"ML STANDALONE: Found {len(df)} records for {symbol}/{interval} in database")
            
            # Convert Decimal columns to float for calculations
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    if df[col].dtype == 'object':
                        # Convert Decimal objects to float
                        df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
            
            # Make sure timestamp is a proper datetime
            if 'timestamp' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        else:
            logging.warning(f"ML STANDALONE: No data found in database for {symbol}/{interval}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
    except Exception as e:
        logging.error(f"ML STANDALONE: Error fetching data from database: {e}")
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
    finally:
        if conn:
            conn.close()

def apply_recursion_fix():
    """
    Apply direct fixes to break the recursion loop by completely replacing
    the get_historical_data functions in both modules with our standalone version
    """
    logging.info("Applying direct fix for maximum recursion depth errors")
    
    try:
        # Import modules that need patching
        import binance_api
        import database_extensions
        
        # Store the original functions for later restoration if needed
        global _original_functions
        _original_functions['binance_api.get_historical_data'] = binance_api.get_historical_data
        _original_functions['database_extensions.get_historical_data'] = database_extensions.get_historical_data
        
        # Replace both functions with our standalone version
        # This completely breaks the circular dependency
        binance_api.get_historical_data = get_historical_data_direct_from_db
        database_extensions.get_historical_data = get_historical_data_direct_from_db
        
        logging.info("Successfully applied recursion fix")
        return True
        
    except Exception as e:
        logging.error(f"Failed to apply recursion fix: {e}")
        return False

def restore_original_functions():
    """Restore the original functions if needed"""
    try:
        import binance_api
        import database_extensions
        
        if 'binance_api.get_historical_data' in _original_functions:
            binance_api.get_historical_data = _original_functions['binance_api.get_historical_data']
            
        if 'database_extensions.get_historical_data' in _original_functions:
            database_extensions.get_historical_data = _original_functions['database_extensions.get_historical_data']
            
        logging.info("Original functions restored")
        return True
    except Exception as e:
        logging.error(f"Failed to restore original functions: {e}")
        return False

def fix_ml_modules():
    """Apply the recursion fix to enable ML modules to work properly"""
    try:
        # Apply the patch to fix recursion issues
        if apply_recursion_fix():
            logging.info("ML modules patched successfully")
            return True
        else:
            logging.error("Failed to patch ML modules")
            return False
    except Exception as e:
        logging.error(f"Error fixing ML modules: {e}")
        return False

# Import additional modules needed for the ML operations
def get_ml_data(symbol, interval, lookback_days=30):
    """
    Get data specifically for ML operations, with proper feature preparation
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '1d')
        lookback_days: Number of days to look back
        
    Returns:
        DataFrame with OHLCV, indicators, and ML features
    """
    # Get raw historical data using our standalone function
    df = get_historical_data_direct_from_db(symbol=symbol, interval=interval, lookback_days=lookback_days)
    
    if df.empty:
        logging.warning(f"No data available for {symbol}/{interval}")
        return pd.DataFrame()
    
    try:
        # Add indicators - but import here to avoid circular imports
        from indicators import add_all_indicators
        
        # Add indicators to the dataframe
        df = add_all_indicators(df)
        logging.info(f"Added indicators to {symbol}/{interval} data")
        
        return df
    except Exception as e:
        logging.error(f"Error adding indicators: {e}")
        return df  # Return raw data without indicators if error


if __name__ == "__main__":
    print("Applying direct ML fix to resolve maximum recursion depth errors")
    success = fix_ml_modules()
    if success:
        print("Successfully applied ML fixes. ML modules should now work correctly.")
    else:
        print("Failed to apply ML fixes. Check ml_training.log for details.")