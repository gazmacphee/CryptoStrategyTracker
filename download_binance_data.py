"""
Script to download historical cryptocurrency data from Binance Data Vision repository
and populate the database with real data.

This script will:
1. Download data for specified symbols and intervals going back 3 years
2. Process and normalize the data
3. Save it to the database
4. Calculate and save technical indicators

Data source: https://data.binance.vision/
"""

import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import zipfile
import io
from calendar import monthrange
import logging

# Import project modules
from database import create_tables, save_historical_data, save_indicators
import indicators
from strategy import evaluate_buy_sell_signals

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('binance_data_download.log')
    ]
)

# Binance Data Vision base URL
BASE_URL = "https://data.binance.vision/data/spot"

# Lock file to prevent multiple processes
LOCK_FILE = ".backfill_lock"

# Symbols and intervals to download
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT",
    "SOLUSDT", "DOTUSDT", "LTCUSDT", "LINKUSDT", "MATICUSDT", "AVAXUSDT"
]

# Intervals to download (excluding 1m, 3m, 5m as requested)
INTERVALS = [
    "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"
]

def download_file(url):
    """Download a file from a URL and return its content"""
    logging.info(f"Downloading from: {url}")
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.content
        else:
            logging.error(f"Failed to download {url}. Status code: {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")
        return None

def process_kline_data(data_content, symbol, interval):
    """Process kline data from a ZIP file content"""
    if not data_content:
        return None
    
    try:
        # Extract ZIP file to memory
        with zipfile.ZipFile(io.BytesIO(data_content)) as zip_file:
            # CSV files typically have the same name as the ZIP file
            csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
            if not csv_files:
                logging.warning(f"No CSV files found in the ZIP for {symbol} {interval}")
                return None
            
            # Extract data from CSV
            with zip_file.open(csv_files[0]) as csv_file:
                # Binance kline data columns
                columns = [
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ]
                
                df = pd.read_csv(csv_file, header=None, names=columns)
                
                # Convert timestamp columns
                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
                df['timestamp'] = df['open_time']  # More intuitive name
                
                # Convert string values to float
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                
                # Select relevant columns and return
                result_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                return result_df
    
    except Exception as e:
        logging.error(f"Error processing kline data for {symbol} {interval}: {e}")
        return None

def download_monthly_klines(symbol, interval, year, month):
    """Download klines for a specific month"""
    # Format month with leading zero
    month_str = f"{month:02d}"
    
    # Determine number of days in month
    days_in_month = monthrange(year, month)[1]
    
    # Monthly data URL
    monthly_url = f"{BASE_URL}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month_str}.zip"
    
    # Try to download monthly data first
    content = download_file(monthly_url)
    if content:
        df = process_kline_data(content, symbol, interval)
        if df is not None and not df.empty:
            logging.info(f"Successfully downloaded monthly data for {symbol} {interval} {year}-{month_str}")
            return df
    
    # If monthly data not available, try daily data for each day in month
    logging.info(f"Monthly data not available, trying daily downloads for {symbol} {interval} {year}-{month_str}")
    
    daily_dfs = []
    for day in range(1, days_in_month + 1):
        day_str = f"{day:02d}"
        daily_url = f"{BASE_URL}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month_str}-{day_str}.zip"
        
        content = download_file(daily_url)
        if content:
            df = process_kline_data(content, symbol, interval)
            if df is not None and not df.empty:
                daily_dfs.append(df)
                logging.info(f"Successfully downloaded daily data for {symbol} {interval} {year}-{month_str}-{day_str}")
        
        # Add small delay to avoid overwhelming the server
        time.sleep(0.5)
    
    if daily_dfs:
        # Combine all daily dataframes for this month
        combined_df = pd.concat(daily_dfs, ignore_index=True)
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['timestamp'])
        # Sort by timestamp
        combined_df = combined_df.sort_values('timestamp')
        return combined_df
    
    logging.warning(f"No data available for {symbol} {interval} {year}-{month_str}")
    return None

def download_symbol_interval_data(symbol, interval, start_date, end_date=None):
    """
    Download all available data for a symbol and interval between start_date and end_date
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '1d')
        start_date: Start date (datetime or date object)
        end_date: End date (datetime or date object, default=today)
    
    Returns:
        DataFrame with all data or None if no data available
    """
    # Convert to date objects if datetime
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    
    if end_date is None:
        end_date = date.today()
    elif isinstance(end_date, datetime):
        end_date = end_date.date()
    
    logging.info(f"Downloading data for {symbol} {interval} from {start_date} to {end_date}")
    
    all_dfs = []
    current_date = start_date
    
    while current_date <= end_date:
        year = current_date.year
        month = current_date.month
        
        df = download_monthly_klines(symbol, interval, year, month)
        if df is not None and not df.empty:
            all_dfs.append(df)
        
        # Move to next month
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        
        current_date = date(year, month, 1)
    
    if all_dfs:
        # Combine all monthly dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['timestamp'])
        # Sort by timestamp
        combined_df = combined_df.sort_values('timestamp')
        return combined_df
    
    return None

def calculate_and_save_indicators(df, symbol, interval):
    """Calculate technical indicators and save to database"""
    if df is None or df.empty:
        return
    
    # Add technical indicators
    df = indicators.add_bollinger_bands(df)
    df = indicators.add_rsi(df)
    df = indicators.add_macd(df)
    df = indicators.add_ema(df)
    
    # Evaluate trading signals
    df = evaluate_buy_sell_signals(df)
    
    # Save indicators to database
    save_indicators(df, symbol, interval)
    logging.info(f"Saved indicators for {symbol} {interval}")

def backfill_symbol_interval(symbol, interval, lookback_years=3):
    """Backfill data for a specific symbol and interval going back specified years"""
    # Calculate start date
    end_date = date.today()
    start_date = date(end_date.year - lookback_years, end_date.month, 1)
    
    logging.info(f"Backfilling {symbol} {interval} from {start_date} to {end_date}")
    
    # Download data
    df = download_symbol_interval_data(symbol, interval, start_date, end_date)
    
    if df is not None and not df.empty:
        # Number of candles downloaded
        candle_count = len(df)
        logging.info(f"Downloaded {candle_count} candles for {symbol} {interval}")
        
        # Save to database
        save_historical_data(df, symbol, interval)
        logging.info(f"Saved historical data to database for {symbol} {interval}")
        
        # Calculate and save indicators
        calculate_and_save_indicators(df, symbol, interval)
        
        return candle_count
    else:
        logging.warning(f"No data available for {symbol} {interval}")
        return 0

def run_backfill(symbols=None, intervals=None, lookback_years=3):
    """Run backfill for specified symbols and intervals"""
    if symbols is None:
        symbols = SYMBOLS
    
    if intervals is None:
        intervals = INTERVALS
    
    # Make sure database tables exist
    create_tables()
    
    logging.info(f"Starting backfill with {len(symbols)} symbols and {len(intervals)} intervals")
    logging.info(f"Symbols: {', '.join(symbols)}")
    logging.info(f"Intervals: {', '.join(intervals)}")
    
    total_candles = 0
    
    # Process each symbol and interval
    for symbol in symbols:
        for interval in intervals:
            try:
                candles = backfill_symbol_interval(symbol, interval, lookback_years)
                total_candles += candles
                
                # Sleep to avoid overwhelming the server
                time.sleep(1)
            except Exception as e:
                logging.error(f"Error processing {symbol} {interval}: {e}")
    
    logging.info(f"Backfill completed. Total candles downloaded: {total_candles}")
    return total_candles

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Download historical cryptocurrency data from Binance Data Vision')
    parser.add_argument('--reset', action='store_true', help='Reset database before backfill')
    parser.add_argument('--years', type=int, default=3, help='Number of years to look back')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to download')
    parser.add_argument('--intervals', nargs='+', help='Specific intervals to download')
    
    args = parser.parse_args()
    
    # Reset database if requested
    if args.reset:
        from clean_reset_database import reset_database
        reset_database()
    
    # Check if lock file exists
    if os.path.exists(LOCK_FILE):
        logging.warning("A backfill process is already running. Use clear_backfill_lock.py to remove it if needed.")
        sys.exit(1)
    
    # Create lock file
    with open(LOCK_FILE, 'w') as f:
        f.write(f"Backfill started at {datetime.now()}")
    
    try:
        run_backfill(
            symbols=args.symbols,
            intervals=args.intervals,
            lookback_years=args.years
        )
    finally:
        # Remove lock file when done
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)