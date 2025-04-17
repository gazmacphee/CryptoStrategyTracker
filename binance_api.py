import os
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import numpy as np
import json
import random
import hmac
import hashlib
from urllib.parse import urlencode

# Binance API endpoints
BASE_URL = "https://api.binance.com/api/v3"
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

# Helper function to generate API signatures for authenticated requests
def get_binance_signature(data, secret):
    """Generate signature for Binance API authentication"""
    signature = hmac.new(
        bytes(secret, 'utf-8'),
        msg=bytes(data, 'utf-8'),
        digestmod=hashlib.sha256
    ).hexdigest()
    return signature

# Try to access Binance API to check if it's accessible
try:
    response = requests.get(f"{BASE_URL}/ping")
    
    if response.status_code == 200:
        BINANCE_API_ACCESSIBLE = True
        print("Successfully connected to Binance API")
        
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
            # We're using account endpoint which requires authentication
            try:
                auth_response = requests.get(
                    f"{BASE_URL}/account",
                    params=params,
                    headers=headers
                )
                
                if auth_response.status_code == 200:
                    BINANCE_API_AUTH = True
                    print("API credentials are valid")
                else:
                    print(f"API credentials could not be verified: {auth_response.text}")
            except Exception as auth_err:
                print(f"Error checking API credentials: {auth_err}")
    else:
        print("Binance API is restricted in this location. Using alternative data source.")
except Exception as e:
    print(f"Error checking Binance API: {e}")
    print("Using alternative data source.")

# Force use of Binance API if we have API keys, regardless of location restrictions
if API_KEY and API_SECRET:
    print("API keys found - forcing use of real Binance API data")
    BINANCE_API_ACCESSIBLE = True
    BINANCE_API_AUTH = True

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
    
    Returns an empty DataFrame to prevent any synthetic data from being generated.
    """
    print(f"WARNING: Synthetic data generation requested but disabled. Only real API data will be used.")
    
    # Return empty DataFrame with expected structure
    return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Default to 1h if interval not recognized
    seconds_per_candle = interval_seconds.get(interval, 3600)
    
    # Normalize start and end times to interval boundaries
    if isinstance(start_time, datetime):
        start_timestamp = start_time
    else:
        start_timestamp = datetime.fromtimestamp(start_time/1000) if isinstance(start_time, int) else datetime.now() - timedelta(days=30)
        
    if isinstance(end_time, datetime):
        end_timestamp = end_time
    else:
        end_timestamp = datetime.fromtimestamp(end_time/1000) if isinstance(end_time, int) else datetime.now()
    
    # Normalize timestamps to exact interval boundaries to avoid duplicates
    # For example, for 1h interval, ensure timestamps are at exact hour boundaries
    if interval in ['1h', '2h', '4h', '6h', '8h', '12h']:
        # For hour-based intervals, normalize to exact hours
        start_timestamp = start_timestamp.replace(minute=0, second=0, microsecond=0)
        end_timestamp = end_timestamp.replace(minute=0, second=0, microsecond=0)
    elif interval in ['1d', '3d', '1w']:
        # For day-based intervals, normalize to midnight
        start_timestamp = start_timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        end_timestamp = end_timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    elif interval == '1M':
        # For month interval, normalize to first day of month
        start_timestamp = start_timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_timestamp = end_timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        # For minute-based intervals, ensure timestamps are at exact minute boundaries
        start_timestamp = start_timestamp.replace(second=0, microsecond=0)
        end_timestamp = end_timestamp.replace(second=0, microsecond=0)
    
    # Calculate number of candles
    time_diff = (end_timestamp - start_timestamp).total_seconds()
    num_candles = min(int(time_diff / seconds_per_candle), limit)
    
    # Base price depends on the symbol (make popular coins have realistic starting prices)
    base_prices = {
        'BTCUSDT': 30000.0, 'ETHUSDT': 2000.0, 'BNBUSDT': 300.0, 'ADAUSDT': 0.5,
        'DOGEUSDT': 0.1, 'XRPUSDT': 0.5, 'SOLUSDT': 70.0, 'DOTUSDT': 8.0
    }
    base_price = base_prices.get(symbol, random.uniform(1.0, 100.0))
    
    # Generate data with some randomness but following trends
    data = []
    current_price = base_price
    timestamp = start_timestamp
    
    for i in range(num_candles):
        # Each candle has some randomness but maintains a trend
        change_percent = random.uniform(-2.0, 2.0)  # -2% to 2% change
        close_price = current_price * (1 + change_percent/100)
        
        # Generate high, low, open with realistic relationships
        high_price = max(current_price, close_price) * (1 + random.uniform(0.1, 1.0)/100)
        low_price = min(current_price, close_price) * (1 - random.uniform(0.1, 1.0)/100)
        open_price = current_price
        
        # Generate volume with some randomness
        volume = random.uniform(100, 1000) * (base_price / 10)
        
        # Add row to data
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        # Update for next candle
        current_price = close_price
        timestamp += timedelta(seconds=seconds_per_candle)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

def get_current_prices(symbols=None):
    """
    Get current prices for one or multiple symbols
    
    Args:
        symbols: Single symbol (string) or list of symbols. If None, fetches all available prices.
    
    Returns:
        Dictionary with symbol as key and current price as value or empty dict if data cannot be fetched
    """
    # Check if we have API credentials - if not, return empty dict
    if not API_KEY or not API_SECRET:
        print("ERROR: No API credentials available. Cannot retrieve current prices.")
        return {}
    
    try:
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
                    # Get price for each symbol with individual API calls
                    single_response = requests.get(f"{BASE_URL}{TICKER_PRICE_ENDPOINT}", params={"symbol": symbol})
                    if single_response.status_code == 200:
                        price_data = single_response.json()
                        prices[price_data["symbol"]] = float(price_data["price"])
                    else:
                        print(f"Error fetching price for {symbol}: {single_response.text}")
                    
                    # Add small delay to avoid rate limiting
                    time.sleep(0.05)
                    
                return prices
        
        # Get all prices
        response = requests.get(f"{BASE_URL}{TICKER_PRICE_ENDPOINT}", params=params)
        
        if response.status_code != 200:
            print(f"Error fetching prices: {response.text}")
            # Return empty dict, never use synthetic data
            return {}
        
        price_data = response.json()
        
        # Format response based on whether it's a single symbol or multiple
        if isinstance(price_data, dict):
            # Single symbol
            return {price_data["symbol"]: float(price_data["price"])}
        else:
            # Multiple symbols
            return {item["symbol"]: float(item["price"]) for item in price_data}
            
    except Exception as e:
        print(f"Error in get_current_prices: {e}")
        # Return empty dict, never use synthetic data
        return {}

def get_klines_data(symbol, interval, start_time=None, end_time=None, limit=1000):
    """
    Fetch klines (candlestick) data from Binance API.
    Never falls back to synthetic data.
    Returns empty DataFrame if real data cannot be retrieved.
    """
    # Skip excluded intervals as requested
    if interval in ['1m', '3m', '5m']:
        print(f"Skipping {interval} interval as requested")
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # If no API credentials, return empty DataFrame
    if not API_KEY or not API_SECRET:
        print(f"ERROR: No API credentials available. Cannot retrieve real data for {symbol} ({interval})")
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    try:
        # Build request parameters
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        
        # Add start and end time if provided
        if start_time:
            if isinstance(start_time, datetime):
                params["startTime"] = int(start_time.timestamp() * 1000)
            else:
                params["startTime"] = start_time
                
        if end_time:
            if isinstance(end_time, datetime):
                params["endTime"] = int(end_time.timestamp() * 1000)
            else:
                params["endTime"] = end_time
        
        # Add API key to headers if available
        headers = {}
        if API_KEY:
            headers['X-MBX-APIKEY'] = API_KEY
        
        # Make request
        response = requests.get(f"{BASE_URL}{KLINES_ENDPOINT}", params=params, headers=headers)
        
        if response.status_code != 200:
            print(f"Error fetching klines: {response.text}")
            # Return empty DataFrame instead of synthetic data
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Parse response
        data = response.json()
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert types
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        df['timestamp'] = df['open_time']  # More intuitive name
        
        # Convert string values to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # If we need more data than the limit allows, make multiple requests
        if limit == 1000 and start_time and end_time:
            start_dt = start_time if isinstance(start_time, datetime) else datetime.fromtimestamp(start_time/1000)
            end_dt = end_time if isinstance(end_time, datetime) else datetime.fromtimestamp(end_time/1000)
            
            # If time range is greater than what a single request can provide, fetch in chunks
            time_delta = end_dt - start_dt
            if time_delta.days > 30:  # Approximate limit for 1000 candles at 1h interval
                all_data = []
                current_start = start_dt
                
                while current_start < end_dt:
                    current_end = min(current_start + timedelta(days=30), end_dt)
                    chunk_df = get_klines_data(
                        symbol, interval, current_start, current_end, limit=1000
                    )
                    if not chunk_df.empty:
                        all_data.append(chunk_df)
                    current_start = current_end
                    
                    # Rate limiting to avoid API errors
                    time.sleep(0.5)
                
                if all_data:
                    df = pd.concat(all_data, ignore_index=True)
                    # Remove duplicates
                    df = df.drop_duplicates(subset=['timestamp'])
                    df = df.sort_values('timestamp')
        
        # Select relevant columns
        result_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return result_df
    
    except Exception as e:
        print(f"Error in get_klines_data: {e}")
        # Return empty DataFrame, never use synthetic data
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
