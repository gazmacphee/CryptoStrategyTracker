import os
import pandas as pd
import requests
from datetime import datetime, timedelta, date
import time
import numpy as np
import json
import random
import hmac
import hashlib
from urllib.parse import urlencode
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Binance API endpoint
BASE_URL = "https://api.binance.com/api/v3"  # Global endpoint as requested by user

# API endpoint paths
KLINES_ENDPOINT = "/klines"
EXCHANGE_INFO_ENDPOINT = "/exchangeInfo"
TICKER_PRICE_ENDPOINT = "/ticker/price"
TICKER_24HR_ENDPOINT = "/ticker/24hr"

# Get API keys from environment variables
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Flag to check if we can access Binance API
BINANCE_API_ACCESSIBLE = False
# Flag to check if we have valid API credentials
BINANCE_API_AUTH = False
# Flag to check if we're in a geo-restricted region for Binance
BINANCE_API_GEO_RESTRICTED = False

# Helper function to generate API signatures for authenticated requests
def get_binance_signature(data, secret):
    """Generate signature for Binance API authentication"""
    signature = hmac.new(
        bytes(secret, 'utf-8'),
        msg=bytes(data, 'utf-8'),
        digestmod=hashlib.sha256
    ).hexdigest()
    return signature

# Function to try Binance API endpoints
def try_binance_endpoints():
    global BASE_URL, BINANCE_API_ACCESSIBLE, BINANCE_API_AUTH, BINANCE_API_GEO_RESTRICTED
    
    endpoints_to_try = [BASE_URL]  # Only using the main endpoint as requested
    
    for endpoint in endpoints_to_try:
        try:
            print(f"Trying Binance API endpoint: {endpoint}")
            response = requests.get(f"{endpoint}/ping")
            
            if response.status_code == 200:
                BINANCE_API_ACCESSIBLE = True
                
                # Set the working endpoint as our base URL
                BASE_URL = endpoint
                
                print(f"Successfully connected to Binance API at {endpoint}")
                
                # Check if API credentials are valid by making an authenticated request
                if API_KEY and API_SECRET:
                    # Current timestamp for the request
                    timestamp = int(time.time() * 1000)
                    
                    # Request parameters
                    params = {
                        'timestamp': timestamp
                    }
                    
                    # Generate signature
                    query_string = urlencode(params)
                    signature = get_binance_signature(query_string, API_SECRET)
                    params['signature'] = signature
                    
                    # Add API key to headers
                    headers = {
                        'X-MBX-APIKEY': API_KEY
                    }
                    
                    # Make a simple authenticated request to test credentials
                    try:
                        auth_response = requests.get(
                            f"{endpoint}/account",
                            params=params,
                            headers=headers
                        )
                        
                        if auth_response.status_code == 200:
                            BINANCE_API_AUTH = True
                            print(f"API credentials are valid on {endpoint}")
                            return True  # Successfully connected and authenticated
                        elif auth_response.status_code == 451 or ('restricted location' in auth_response.text.lower()):
                            # This is a geographical restriction error
                            # Use globals() to modify global variables from within a function
                            globals()['BINANCE_API_GEO_RESTRICTED'] = True
                            print(f"Binance API is not available in your region. The application will use backfilled historical data.")
                            globals()['BINANCE_API_ACCESSIBLE'] = False
                            globals()['BINANCE_API_AUTH'] = False
                            return False
                        else:
                            print(f"API credentials could not be verified on {endpoint}: {auth_response.text}")
                    except Exception as auth_err:
                        print(f"Error checking API credentials on {endpoint}: {auth_err}")
                
                # Return True even if auth failed (we still have connectivity)
                return True
            elif response.status_code == 451 or ('restricted location' in str(response.text).lower()):
                # This is a geographical restriction error
                # Use globals() to modify global variables from within a function
                globals()['BINANCE_API_GEO_RESTRICTED'] = True
                print(f"Binance API is not available in your region. The application will use backfilled historical data.")
                return False
            else:
                print(f"Could not access Binance API at {endpoint}: {response.status_code}")
        
        except Exception as e:
            print(f"Error checking Binance API at {endpoint}: {e}")
    
    # If we get here, all endpoints failed
    print("All Binance API endpoints failed. Using alternative data source.")
    return False

# Try to access Binance API to check if it's accessible
try_binance_endpoints()

# Force use of Binance API if we have API keys, ONLY if we're not in a geo-restricted region
if API_KEY and API_SECRET and not BINANCE_API_GEO_RESTRICTED:
    print("API keys found - forcing use of real Binance API data")
    BINANCE_API_ACCESSIBLE = True
    BINANCE_API_AUTH = True
elif BINANCE_API_GEO_RESTRICTED:
    print("API keys found but we're in a geo-restricted region. Using backfilled data instead.")
    BINANCE_API_ACCESSIBLE = False
    BINANCE_API_AUTH = False

def get_available_symbols(quote_asset="USDT", limit=30):
    """Get available trading pairs from Binance"""
    # Default popular symbols to return if API is not accessible
    popular_symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT",
        "SOLUSDT", "DOTUSDT", "LTCUSDT", "LINKUSDT", "MATICUSDT", "AVAXUSDT",
        "ATOMUSDT", "UNIUSDT", "FILUSDT", "AAVEUSDT", "NEARUSDT", "ALGOUSDT",
        "ICPUSDT", "XTZUSDT", "AXSUSDT", "FTMUSDT", "SANDUSDT", "MANAUSDT",
        "HBARUSDT", "THETAUSDT", "EGLDUSDT", "FLOWUSDT", "APUSDT", "CAKEUSDT"
    ]
    
    # Return popular symbols if API is not accessible OR we can't get proper symbols
    if not BINANCE_API_ACCESSIBLE or True:  # Always use popular symbols for consistent processing
        print("Using pre-defined list of popular symbols")
        return popular_symbols[:limit]
    
    try:
        response = requests.get(f"{BASE_URL}{EXCHANGE_INFO_ENDPOINT}")
        
        if response.status_code != 200:
            print(f"Error getting exchange info: {response.text}")
            return popular_symbols[:limit]
        
        data = response.json()
        symbols = [s["symbol"] for s in data["symbols"] 
                  if s["symbol"].endswith(quote_asset) and s["status"] == "TRADING"]
        
        # Sort by symbol name and limit if requested
        symbols.sort()
        if limit and limit < len(symbols):
            # Filter popular from symbols
            available_popular = [s for s in popular_symbols if s in symbols]
            remaining = [s for s in symbols if s not in popular_symbols]
            
            # Combine lists and limit
            result = available_popular + remaining
            return result[:limit]
        
        return symbols
    except Exception as e:
        print(f"Error fetching available symbols: {e}")
        return popular_symbols[:limit]

def generate_synthetic_candle_data(symbol, interval, start_time, end_time, limit=1000):
    """
    This function is deprecated and will not be used.
    We now only use real data from Binance API.
    
    Returns an empty DataFrame with expected structure.
    """
    print(f"WARNING: Synthetic data generation requested but disabled. Only real API data will be used.")
    
    # Return empty DataFrame with expected structure
    return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

def get_current_prices(symbols=None):
    """
    Get current prices for one or multiple symbols
    
    Args:
        symbols: Single symbol (string) or list of symbols. If None, fetches all available prices.
    
    Returns:
        Dictionary with symbol as key and current price as value or empty dict if data cannot be fetched
    """
    # Check if backfill is still running
    try:
        from os.path import exists
        if exists('.backfill_lock'):
            print("Backfill is still running. Cannot retrieve current prices until backfill completes.")
            return {}
    except Exception as e:
        print(f"Error checking backfill status: {e}")
    
    # Check if we have API credentials - if not, return empty dict
    if not API_KEY or not API_SECRET:
        print("ERROR: No API credentials available. Cannot retrieve current prices.")
        return {}
    
    # Skip API calls if we're in a geo-restricted region
    if BINANCE_API_GEO_RESTRICTED:
        print("Skipping API calls for current prices because we're in a geo-restricted region.")
        return {}
    
    # Use only the main endpoint as requested
    endpoints_to_try = [BASE_URL]
    
    for endpoint in endpoints_to_try:
        try:
            print(f"Fetching current prices from {endpoint}...")
            
            params = {}
            # If specific symbols are requested, format them for the API
            if symbols:
                if isinstance(symbols, str):
                    # Single symbol
                    params["symbol"] = symbols
                else:
                    # Multiple symbols - use the price endpoint for a specific symbol to avoid rate limits
                    prices = {}
                    for symbol in symbols:
                        # Add timestamp and signature for authentication
                        timestamp = int(time.time() * 1000)
                        request_params = {
                            "symbol": symbol,
                            "timestamp": timestamp
                        }
                        
                        # Generate signature
                        query_string = urlencode(request_params)
                        signature = get_binance_signature(query_string, API_SECRET)
                        request_params['signature'] = signature
                        
                        # Add API key to headers
                        headers = {'X-MBX-APIKEY': API_KEY}
                        
                        # Get price for each symbol with individual API calls
                        single_response = requests.get(
                            f"{endpoint}{TICKER_PRICE_ENDPOINT}", 
                            params=request_params,
                            headers=headers
                        )
                        
                        if single_response.status_code == 200:
                            price_data = single_response.json()
                            prices[price_data["symbol"]] = float(price_data["price"])
                            
                            # Update base URL if this endpoint worked
                            if endpoint != BASE_URL:
                                # Update global variables at module level
                                global_vars = globals()
                                global_vars['BASE_URL'] = endpoint
                                print(f"Switching to {endpoint} for future API calls")
                        else:
                            print(f"Error fetching price for {symbol} from {endpoint}: {single_response.text}")
                        
                        # Add small delay to avoid rate limiting
                        time.sleep(0.05)
                        
                    # If we got any prices, return them
                    if prices:
                        return prices
                    # Otherwise continue to try the next endpoint
            
            # Get all prices
            # Add timestamp and signature for authentication
            timestamp = int(time.time() * 1000)
            request_params = {
                "timestamp": timestamp
            }
            
            # Add any existing params
            for key, value in params.items():
                request_params[key] = value
            
            # Generate signature
            query_string = urlencode(request_params)
            signature = get_binance_signature(query_string, API_SECRET)
            request_params['signature'] = signature
            
            # Add API key to headers
            headers = {'X-MBX-APIKEY': API_KEY}
            
            response = requests.get(
                f"{endpoint}{TICKER_PRICE_ENDPOINT}", 
                params=request_params,
                headers=headers
            )
            
            if response.status_code == 200:
                price_data = response.json()
                
                # Update base URL if this endpoint worked
                if endpoint != BASE_URL:
                    # Update global variables at module level
                    global_vars = globals()
                    global_vars['BASE_URL'] = endpoint
                    print(f"Switching to {endpoint} for future API calls")
                
                # Format response based on whether it's a single symbol or multiple
                if isinstance(price_data, dict):
                    # Single symbol
                    return {price_data["symbol"]: float(price_data["price"])}
                else:
                    # Multiple symbols
                    return {item["symbol"]: float(item["price"]) for item in price_data}
            else:
                print(f"Error fetching prices from {endpoint}: {response.text}")
                # Try the next endpoint
        
        except Exception as e:
            print(f"Error in get_current_prices with {endpoint}: {e}")
            # Try the next endpoint
    
    # If we get here, all endpoints failed
    print("All Binance API endpoints failed for current prices. Returning empty dict.")
    return {}

def get_recent_klines_from_api(symbol, interval, start_time=None, end_time=None, limit=1000):
    """
    Fetch the most recent klines directly from Binance API using API credentials.
    This is used for real-time/recent data that might not be available in the historical CSVs.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '1d')
        start_time: Start time as datetime or timestamp
        end_time: End time as datetime or timestamp
        limit: Maximum number of records to return
        
    Returns:
        DataFrame with OHLCV data or empty DataFrame if API call fails
    """
    # Skip if we don't have API credentials
    if not API_KEY or not API_SECRET:
        print("No API credentials available for direct Binance API access")
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Skip API calls if we're in a geo-restricted region
    if BINANCE_API_GEO_RESTRICTED:
        print("Skipping API calls for recent data because we're in a geo-restricted region.")
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Convert timestamps to milliseconds for API
    if start_time:
        if isinstance(start_time, datetime):
            start_ms = int(start_time.timestamp() * 1000)
        elif isinstance(start_time, int) and start_time > 1000000000000:  # Already milliseconds
            start_ms = start_time
        elif isinstance(start_time, int):  # Seconds
            start_ms = start_time * 1000
        else:
            # Default to 7 days ago for recent data
            start_ms = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
    else:
        # Default to 7 days ago for recent data
        start_ms = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
        
    if end_time:
        if isinstance(end_time, datetime):
            end_ms = int(end_time.timestamp() * 1000)
        elif isinstance(end_time, int) and end_time > 1000000000000:  # Already milliseconds
            end_ms = end_time
        elif isinstance(end_time, int):  # Seconds
            end_ms = end_time * 1000
        else:
            # Default to now
            end_ms = int(datetime.now().timestamp() * 1000)
    else:
        # Default to now
        end_ms = int(datetime.now().timestamp() * 1000)
    
    # Use only the main endpoint as requested
    endpoints_to_try = [BASE_URL]
    
    for endpoint in endpoints_to_try:
        try:
            print(f"Fetching recent data for {symbol} {interval} directly from Binance API at {endpoint}...")
            
            # Set up request parameters - klines endpoint is a public endpoint and doesn't need authentication
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_ms,
                'endTime': end_ms,
                'limit': limit
            }
            
            # For klines endpoint, we don't need to add timestamp and signature as it's a public endpoint
            # This prevents the "Too many parameters" error from the API
            
            # Set headers - we can still include the API key for rate limiting benefits
            headers = {'X-MBX-APIKEY': API_KEY}
            
            # Make request to Binance API
            response = requests.get(f"{endpoint}{KLINES_ENDPOINT}", params=params, headers=headers)
            
            if response.status_code == 200:
                # Parse response
                data = response.json()
                
                # Process the API response into DataFrame
                df_data = []
                for candle in data:
                    df_data.append({
                        'timestamp': datetime.fromtimestamp(candle[0] / 1000),  # Open time
                        'open': float(candle[1]),
                        'high': float(candle[2]),
                        'low': float(candle[3]),
                        'close': float(candle[4]),
                        'volume': float(candle[5])
                    })
                
                df = pd.DataFrame(df_data)
                print(f"Successfully fetched {len(df)} records directly from Binance API at {endpoint}")
                
                # Update the base URL if we're using a different endpoint than the current one
                if endpoint != BASE_URL:
                    # Update global variables at module level
                    global_vars = globals()
                    global_vars['BASE_URL'] = endpoint
                    print(f"Switching to {endpoint} for future API calls")
                
                return df
            elif response.status_code == 451 or ('restricted location' in response.text.lower()):
                # This is a geographical restriction error
                print(f"Binance API is not available in your region. The application will use backfilled historical data instead.")
                # Update the global flag to avoid repeated API attempts if we know we're in a restricted region
                # We need to use globals() to modify the global variable from within a function
                globals()['BINANCE_API_GEO_RESTRICTED'] = True
                # Continue to the next endpoint if available, though it's likely all will fail with the same error
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            else:
                print(f"Error fetching data from API at {endpoint}: {response.text}")
                # Continue to the next endpoint if available
        
        except Exception as e:
            print(f"Error fetching recent data from Binance API at {endpoint}: {e}")
            # Continue to the next endpoint if available
    
    # If we get here, all endpoints failed
    print("All Binance API endpoints failed for recent data. Returning empty dataset.")
    return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

def get_historical_data(symbol, interval, lookback_days=30, start_date=None, end_date=None):
    """
    Get historical data for a symbol and interval.
    This is a wrapper around get_klines_data to provide a consistent interface for ML modules.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '1d')
        lookback_days: Number of days to look back
        start_date: Optional specific start date (overrides lookback_days)
        end_date: Optional specific end date (defaults to now)
        
    Returns:
        DataFrame with OHLCV and timestamp data
    """
    # Calculate start and end times based on inputs
    if end_date is None:
        end_time = datetime.now()
    else:
        end_time = end_date if isinstance(end_date, datetime) else datetime.strptime(end_date, '%Y-%m-%d')
    
    if start_date is not None:
        start_time = start_date if isinstance(start_date, datetime) else datetime.strptime(start_date, '%Y-%m-%d')
    else:
        start_time = end_time - timedelta(days=lookback_days)
    
    # Get data from klines data function
    df = get_klines_data(symbol, interval, start_time, end_time)
    
    # Ensure we have the expected columns and format
    if not df.empty:
        # Make sure we have a timestamp column
        if 'timestamp' not in df.columns and 'open_time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        elif 'timestamp' not in df.columns:
            # Create timestamp from index if needed
            df['timestamp'] = df.index
    
    return df

def get_historical_data(symbol, interval, lookback_days=30, start_date=None, end_date=None):
    """
    Get historical data for a symbol and interval.
    This is a wrapper to provide a consistent interface for ML modules.
    
    This function now redirects to database_extensions.get_historical_data,
    which prioritizes pulling data from the database instead of re-downloading.
    
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
        # Import database_extensions to ensure it's available
        import database_extensions
        print(f"Redirecting historical data request for {symbol}/{interval} to database_extensions")
        return database_extensions.get_historical_data(symbol, interval, lookback_days, start_date, end_date)
    except ImportError:
        print("Warning: database_extensions module not found. Using direct method.")
        try:
            # Convert dates if provided
            start_time = None
            if start_date:
                if isinstance(start_date, str):
                    start_time = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
                elif isinstance(start_date, datetime):
                    start_time = int(start_date.timestamp() * 1000)
            
            end_time = None    
            if end_date:
                if isinstance(end_date, str):
                    end_time = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
                elif isinstance(end_date, datetime):
                    end_time = int(end_date.timestamp() * 1000)
            
            if not start_time and lookback_days:
                # Calculate start time based on lookback_days
                end_ts = end_time if end_time else int(datetime.now().timestamp() * 1000)
                start_time = end_ts - (lookback_days * 24 * 60 * 60 * 1000)
                
            return get_klines_data(symbol, interval, start_time, end_time)
        except Exception as e:
            print(f"Error getting historical data for {symbol}/{interval}: {e}")
            return None

def get_klines_data(symbol, interval, start_time=None, end_time=None, limit=1000):
    """
    Fetch klines (candlestick) data using a hybrid approach:
    1. Check database first for historical data
    2. If data is missing or for recent time periods, fetch directly from Binance API
    3. Fall back to backfilled CSV data downloads if needed
    
    Never uses synthetic data and returns empty DataFrame if real data cannot be retrieved.
    """
    import pandas as pd
    # Skip excluded intervals as requested
    if interval in ['1m', '3m', '5m']:
        print(f"Skipping {interval} interval as requested")
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Create empty DataFrame
    empty_df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Parse timestamps
    if start_time:
        if isinstance(start_time, str):
            # Parse string to datetime
            start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        # Convert milliseconds to datetime if needed
        elif isinstance(start_time, int) and start_time > 1000000000000:  # Likely a millisecond timestamp
            start_time = datetime.fromtimestamp(start_time / 1000)
        elif isinstance(start_time, int):  # Likely a second timestamp
            start_time = datetime.fromtimestamp(start_time)
    else:
        # Default to 30 days ago
        start_time = datetime.now() - timedelta(days=30)
        
    if end_time:
        if isinstance(end_time, str):
            # Parse string to datetime
            end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
        # Convert milliseconds to datetime if needed
        elif isinstance(end_time, int) and end_time > 1000000000000:  # Likely a millisecond timestamp
            end_time = datetime.fromtimestamp(end_time / 1000)
        elif isinstance(end_time, int):  # Likely a second timestamp
            end_time = datetime.fromtimestamp(end_time)
    else:
        # Default to now
        end_time = datetime.now()
    
    # Split time range into historical and recent
    current_time = datetime.now()
    recent_threshold = current_time - timedelta(days=7)  # Consider last 7 days as "recent"
    
    # If requesting recent data that includes today or yesterday, use direct API for those days
    need_recent_data = False
    recent_start = None
    if end_time > recent_threshold:
        recent_start = max(start_time, recent_threshold)
        historical_end = min(end_time, recent_threshold)
        need_recent_data = True
    
    try:
        # Step 1: Try to use our database data first (direct query)
        from database import get_db_connection
        import pandas as pd
        
        conn = get_db_connection()
        df_from_db = pd.DataFrame()
        
        if conn:
            try:
                query = """
                SELECT timestamp, open, high, low, close, volume 
                FROM historical_data 
                WHERE symbol = %s AND interval = %s
                AND timestamp BETWEEN %s AND %s
                ORDER BY timestamp
                """
                
                df_from_db = pd.read_sql_query(
                    query, 
                    conn, 
                    params=(symbol, interval, start_time, end_time)
                )
                
                if not df_from_db.empty and len(df_from_db) > 10:  # If we have some reasonable amount of data
                    print(f"Using {len(df_from_db)} records from database for {symbol} {interval}")
                    
                    # Check if we need to supplement with recent API data
                    if need_recent_data and API_KEY and API_SECRET:
                        # Find the most recent timestamp in our database data
                        if not df_from_db.empty:
                            last_db_time = df_from_db['timestamp'].max()
                            # Only get API data if we need more recent data
                            if last_db_time < end_time:
                                print(f"Supplementing with recent data from API after {last_db_time}")
                                df_recent = get_recent_klines_from_api(
                                    symbol, interval, 
                                    start_time=last_db_time + timedelta(minutes=1), 
                                    end_time=end_time
                                )
                                
                                # Combine the database data with the API data
                                if 'df_recent' in locals() and not df_recent.empty:
                                    df_combined = pd.concat([df_from_db, df_recent]).drop_duplicates(subset=['timestamp'])
                                    print(f"Combined dataset now has {len(df_combined)} records")
                                    return df_combined
                    
                    # If we didn't need API data or couldn't get it, return the database data
                    return df_from_db
            except Exception as db_err:
                print(f"Error querying database: {db_err}")
            finally:
                conn.close()
        
        # Step 2: If we couldn't get enough data from the database, try direct API for recent data
        if need_recent_data and API_KEY and API_SECRET and recent_start is not None:
            print("Trying direct Binance API for recent data...")
            df_recent = get_recent_klines_from_api(symbol, interval, start_time=recent_start, end_time=end_time)
            
            # If we got some data but need historical data too, we'll get that next
            if 'df_recent' in locals() and not df_recent.empty and start_time < recent_threshold:
                # We still need historical data - continue to step 3
                historical_data_needed = True
            else:
                # We only needed recent data or we got all we need
                if 'df_recent' in locals():
                    return df_recent
        
        # Step 3: Database didn't have sufficient data, try downloading from Binance Data Vision
        print(f"Trying to download from Binance Data Vision for {symbol} {interval}")
        
        # Use the single pair download function which we know works
        from download_single_pair import download_and_process
        all_dfs = []
        
        # Adjust dates for CSV API 
        # Data on Binance is only available up to a certain date
        MAX_AVAILABLE_DATE = date(2024, 4, 17)  # Or whatever the current availability is
        
        # Ensure we're not requesting future data
        start_date = start_time.date()
        end_date = min(end_time.date(), MAX_AVAILABLE_DATE)
        
        # Download month by month
        current_date = start_date
        while current_date <= end_date:
            year = current_date.year
            month = current_date.month
            
            print(f"Downloading {symbol} {interval} for {year}-{month}")
            
            # Get data for this month
            download_and_process(symbol, interval, year, month)
            
            # Move to next month
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
            
            current_date = date(year, month, 1)
        
        # Now get the data from the database
        conn = get_db_connection()
        if conn:
            try:
                query = """
                SELECT timestamp, open, high, low, close, volume 
                FROM historical_data 
                WHERE symbol = %s AND interval = %s
                AND timestamp BETWEEN %s AND %s
                ORDER BY timestamp
                """
                
                df_historical = pd.read_sql_query(
                    query, 
                    conn, 
                    params=(symbol, interval, start_time, end_time)
                )
                
                if not df_historical.empty:
                    print(f"Successfully retrieved {len(df_historical)} historical records")
                    
                    # If we also have recent data from API, combine them
                    if need_recent_data and 'df_recent' in locals() and not df_recent.empty:
                        df_combined = pd.concat([df_historical, df_recent]).drop_duplicates(subset=['timestamp'])
                        print(f"Combined dataset has {len(df_combined)} records")
                        return df_combined
                    
                    return df_historical
            except Exception as db_err:
                print(f"Error querying database after download: {db_err}")
            finally:
                conn.close()
        
        # If we got recent data but no historical, still return the recent data
        if need_recent_data and 'df_recent' in locals() and not df_recent.empty:
            return df_recent
        
        # If we get here, we couldn't get data
        print(f"No data available for {symbol} on {interval} timeframe")
        return empty_df
            
    except Exception as e:
        print(f"Error fetching data for {symbol} {interval}: {e}")
        return empty_df
