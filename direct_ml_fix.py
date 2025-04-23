"""
Direct fix for maximum recursion depth exceeded errors in ML modules.
This script directly patches the functions causing the issue and provides
a clean interface for ML modules.
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

def apply_recursion_fix():
    """
    Apply direct fixes to break the recursion loop between
    database_extensions.get_historical_data and binance_api.get_historical_data
    """
    logging.info("Applying direct fix for maximum recursion depth errors")
    
    # Import modules that need patching only after setting up logging
    import database
    import binance_api
    import database_extensions
    
    # Create direct database query function
    def direct_db_query(symbol, interval, start_date=None, end_date=None, lookback_days=30):
        """
        Directly query database for historical data without any redirections
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Time interval (e.g., '1h', '1d')
            start_date: Optional start date
            end_date: Optional end date
            lookback_days: Number of days to look back if start_date not provided
            
        Returns:
            DataFrame with historical price data
        """
        logging.info(f"Direct DB query for {symbol}/{interval}")
        
        # Calculate date range
        if start_date is None:
            start_date = datetime.now() - timedelta(days=lookback_days)
        elif isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Direct database query
        conn = database.get_db_connection()
        if conn is None:
            logging.error("Could not connect to database")
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
                logging.info(f"Found {len(df)} records for {symbol}/{interval} in database")
                
                # Convert Decimal columns to float for calculations
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        if df[col].dtype == 'object':
                            # Sample first value to check if it's Decimal
                            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                            if isinstance(sample, Decimal):
                                logging.info(f"Converting Decimal column {col} to float")
                                df[col] = df[col].astype(float)
                
                return df
            else:
                logging.warning(f"No data found in database for {symbol}/{interval}")
                return pd.DataFrame()
                
        except Exception as e:
            logging.error(f"Error fetching data from database: {e}")
            return pd.DataFrame()
            
        finally:
            if conn:
                conn.close()
    
    # Patch the ML get_historical_data function to use our direct database query ONLY
    def ml_get_historical_data(symbol, interval, lookback_days=30, start_date=None, end_date=None):
        """
        Safe version of get_historical_data for ML modules that avoids recursion
        and ONLY uses database data without attempting Binance API calls
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Time interval (e.g., '1h', '1d')
            lookback_days: Number of days to look back
            start_date: Optional specific start date (overrides lookback_days)
            end_date: Optional specific end date (defaults to now)
            
        Returns:
            DataFrame with OHLCV and timestamp data or empty DataFrame if no data in database
        """
        logging.info(f"ML get_historical_data called for {symbol}/{interval}")
        
        # Only query the database, never try to download from Binance
        df = direct_db_query(symbol, interval, start_date, end_date, lookback_days)
        
        if df is not None and not df.empty:
            logging.info(f"Successfully retrieved {len(df)} records from database")
            return df
        
        # If no data in database, return empty DataFrame (never attempt direct download)
        logging.warning(f"No data available in database for {symbol}/{interval}")
        return pd.DataFrame()
    
    # Replace the functions that cause recursion
    # This is essentially monkey patching to break the infinite recursion
    binance_api.get_historical_data = ml_get_historical_data
    database_extensions.get_historical_data = ml_get_historical_data
    
    logging.info("Successfully applied recursion fix")
    return True


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


if __name__ == "__main__":
    print("Applying direct ML fix to resolve maximum recursion depth errors")
    success = fix_ml_modules()
    if success:
        print("Successfully applied ML fixes. ML modules should now work correctly.")
    else:
        print("Failed to apply ML fixes. Check ml_training.log for details.")