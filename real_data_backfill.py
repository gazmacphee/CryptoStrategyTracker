"""
Script to backfill the database with real data from Binance API only.
Never falls back to synthetic/simulated data.

This script retrieves data in small chunks (1 month at a time) for each symbol and interval,
starting 3 years ago and working forward to the present day.
"""

import os
import sys
import time
import requests
from datetime import datetime, timedelta
import pandas as pd
import hmac
import hashlib
from urllib.parse import urlencode

# Import our modules
from database import create_tables, save_historical_data, get_historical_data, save_indicators
import indicators
from strategy import evaluate_buy_sell_signals
from utils import timeframe_to_seconds

# Binance API endpoints
BASE_URL = "https://api.binance.com/api/v3"
KLINES_ENDPOINT = "/klines"
EXCHANGE_INFO_ENDPOINT = "/exchangeInfo"
TICKER_PRICE_ENDPOINT = "/ticker/price"

# Get API keys from environment variables
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Lock file to prevent multiple processes
LOCK_FILE = ".backfill_lock"

# Helper function to generate API signatures for authenticated requests
def get_binance_signature(data, secret):
    """Generate signature for Binance API authentication"""
    signature = hmac.new(
        bytes(secret, 'utf-8'),
        msg=bytes(data, 'utf-8'),
        digestmod=hashlib.sha256
    ).hexdigest()
    return signature

def get_available_symbols(quote_asset="USDT", limit=30):
    """Get available trading pairs from Binance"""
    # Default popular symbols to use
    popular_symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT",
        "SOLUSDT", "DOTUSDT", "LTCUSDT", "LINKUSDT", "MATICUSDT", "AVAXUSDT"
    ]
    
    # If no API key, return default list
    if not API_KEY:
        print("No API key provided. Using default symbol list.")
        return popular_symbols[:limit]
    
    try:
        # Add API key to headers
        headers = {'X-MBX-APIKEY': API_KEY}
        
        # Make request to get exchange info
        response = requests.get(f"{BASE_URL}{EXCHANGE_INFO_ENDPOINT}", headers=headers)
        
        if response.status_code != 200:
            print(f"Error getting exchange info: {response.text}")
            print("Using default symbol list")
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
        print("Using default symbol list")
        return popular_symbols[:limit]

def get_klines_data(symbol, interval, start_time, end_time, limit=1000):
    """
    Fetch klines (candlestick) data from Binance API.
    Only returns real data, never synthetic.
    
    Args:
        symbol: Trading pair symbol
        interval: Time interval (e.g., '1h', '1d')
        start_time: Start datetime
        end_time: End datetime
        limit: Maximum number of candles per request
        
    Returns:
        DataFrame with candlestick data or None if data cannot be fetched
    """
    # Skip excluded intervals as requested (1m, 3m, 5m)
    if interval in ['1m', '3m', '5m']:
        print(f"Skipping {interval} interval as requested")
        return None
    
    # Check if we have API credentials
    if not API_KEY or not API_SECRET:
        print("Error: No API credentials. Cannot fetch real data.")
        return None
    
    try:
        # Build request parameters
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        
        # Add start and end time
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
        
        # Add API key to headers
        headers = {'X-MBX-APIKEY': API_KEY}
        
        # Make request
        response = requests.get(f"{BASE_URL}{KLINES_ENDPOINT}", params=params, headers=headers)
        
        if response.status_code != 200:
            print(f"Error fetching klines: {response.text}")
            return None
        
        # Parse response
        data = response.json()
        
        # No data returned
        if not data:
            print(f"No data returned for {symbol} ({interval}) from {start_time} to {end_time}")
            return None
        
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
        
        # Select relevant columns
        result_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return result_df
    
    except Exception as e:
        print(f"Error in get_klines_data: {e}")
        return None

def process_symbol_interval_monthly(symbol, interval, total_months=36):
    """
    Process a single symbol and interval month by month, going back up to 36 months (3 years)
    
    Args:
        symbol: Trading pair symbol
        interval: Time interval
        total_months: Total number of months to look back
    """
    print(f"Processing {symbol} on {interval} timeframe...")
    
    # Get current time
    now = datetime.now()
    
    # Process month by month, starting 3 years ago
    for months_ago in range(total_months, 0, -1):
        # Calculate start and end time for this month
        end_time = now - timedelta(days=30 * (months_ago - 1))
        start_time = now - timedelta(days=30 * months_ago)
        
        print(f"  Fetching data from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
        
        # Get data for this period
        df = get_klines_data(symbol, interval, start_time, end_time)
        
        if df is None or df.empty:
            print(f"  No data available for this period")
            continue
        
        print(f"  Retrieved {len(df)} candles")
        
        # Save to database
        save_historical_data(df, symbol, interval)
        print(f"  Saved historical data to database")
        
        # Add technical indicators
        df = indicators.add_bollinger_bands(df)
        df = indicators.add_rsi(df)
        df = indicators.add_macd(df)
        df = indicators.add_ema(df)
        
        # Evaluate trading signals
        df = evaluate_buy_sell_signals(df)
        
        # Save indicators to database
        save_indicators(df, symbol, interval)
        print(f"  Saved indicators to database")
        
        # Sleep briefly to avoid rate limiting
        time.sleep(1)
    
    print(f"Completed processing {symbol} on {interval} timeframe")

def run_real_data_backfill():
    """Run a full backfill using only real data from Binance API"""
    print("Starting real data backfill process...")
    
    # Make sure database tables exist
    create_tables()
    
    # Define timeframes (excluding 1m, 3m, and 5m as requested)
    timeframes = [
        "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"
    ]
    
    # Get available symbols
    symbols = get_available_symbols(limit=12)  # Get top 12 symbols
    
    print(f"Processing {len(symbols)} symbols with {len(timeframes)} timeframes")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Timeframes: {', '.join(timeframes)}")
    
    # Process each symbol and timeframe
    for symbol in symbols:
        for timeframe in timeframes:
            # Process this symbol-timeframe combination month by month
            process_symbol_interval_monthly(symbol, timeframe)
            
            # Sleep between symbol-timeframe combinations to avoid rate limiting
            time.sleep(2)
    
    print("Real data backfill completed successfully!")

if __name__ == "__main__":
    # Check if database reset is requested
    reset_db = "--reset" in sys.argv
    
    if reset_db:
        from clean_reset_database import reset_database
        reset_database()
    
    # Check if lock file exists
    if os.path.exists(LOCK_FILE):
        print("A backfill process is already running. Use clear_backfill_lock.py to remove it if needed.")
        sys.exit(1)
    
    # Create lock file
    with open(LOCK_FILE, 'w') as f:
        f.write(f"Backfill started at {datetime.now()}")
    
    try:
        run_real_data_backfill()
    finally:
        # Remove lock file when done
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)