"""
Windows-compatible Script for Downloading Cryptocurrency Data Directly from Binance

This script is specifically designed for Windows users who may have issues with the Node.js
dependency required by the main backfill process. It downloads data directly from the
Binance Data Vision service without requiring Node.js or the binance-historical-data package.

Usage:
    python windows_data_download.py
"""

import os
import sys
import time
import json
import requests
import io
import zipfile
import calendar
import logging
from datetime import datetime, timedelta
import pandas as pd

from database import save_historical_data, get_db_connection, save_indicators
import indicators
from strategy import evaluate_buy_sell_signals

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                   filename='windows_data_download.log', filemode='a')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Default symbols and intervals to download
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT"
]

DEFAULT_INTERVALS = ['1h', '4h', '1d']

def download_monthly_klines(symbol, interval, year, month):
    """
    Download monthly kline data directly from Binance Data Vision.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '4h', '1d')
        year: Year to download
        month: Month to download
        
    Returns:
        DataFrame with OHLCV data or None if download fails
    """
    try:
        # Build URL for monthly file
        base_url = "https://data.binance.vision/data/spot/monthly/klines"
        filename = f"{symbol.upper()}-{interval}-{year}-{month:02d}.zip"
        url = f"{base_url}/{symbol.upper()}/{interval}/{filename}"
        
        print(f"    └─ Downloading {symbol}/{interval} for {year}-{month:02d}...")
        logging.info(f"Trying to download from Binance Data Vision for {symbol} {interval}")
        
        # Try to download the file
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            # File downloaded successfully
            print(f"       ✓ Monthly file downloaded successfully")
            
            # Process the ZIP file in memory
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                csv_file = zip_file.namelist()[0]  # Get the first file in the ZIP
                
                # Read the CSV into DataFrame with explicit dtypes
                with zip_file.open(csv_file) as f:
                    df = pd.read_csv(f, header=None, names=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                        'taker_buy_quote_volume', 'ignore'
                    ], dtype={
                        'open_time': 'int64',  # Force this to be int64
                        'close_time': 'int64'  # Force this to be int64 too
                    })
                    
                    # Convert numeric data types
                    df['open'] = pd.to_numeric(df['open'])
                    df['high'] = pd.to_numeric(df['high'])
                    df['low'] = pd.to_numeric(df['low'])
                    df['close'] = pd.to_numeric(df['close'])
                    df['volume'] = pd.to_numeric(df['volume'])
                    
                    # Convert timestamps
                    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', errors='coerce')
                    
                    # Drop any invalid timestamps
                    df = df.dropna(subset=['timestamp'])
                    
                    # Keep only essential columns
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    
                    return df
        else:
            # Monthly file not available, try daily files
            print(f"       ⚠ Monthly file not available ({response.status_code}), trying daily files...")
            logging.info(f"No monthly file available for {symbol} {interval} {year}-{month:02d}")
            
            # Try to download daily files for this month
            return download_daily_klines(symbol, interval, year, month)
            
    except Exception as e:
        logging.error(f"Error downloading data for {symbol}/{interval} {year}-{month:02d}: {e}")
        print(f"       ❌ Error downloading: {str(e)}")
        return None

def download_daily_klines(symbol, interval, year, month):
    """
    Download daily kline files for a specific month when monthly file is not available.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '4h', '1d')
        year: Year to download
        month: Month to download
        
    Returns:
        DataFrame with combined daily data or None if download fails
    """
    import calendar
    
    base_url = "https://data.binance.vision/data/spot/daily/klines"
    
    # Get number of days in the month
    _, days_in_month = calendar.monthrange(year, month)
    
    # Initialize empty combined dataframe
    combined_df = None
    
    # Track which files couldn't be processed
    unprocessed_files = []
    successful_days = 0
    
    print(f"       🔍 Attempting to download {days_in_month} daily files for {month:02d}/{year}")
    logging.info(f"Downloading daily klines for {symbol}/{interval} {year}-{month:02d}")
    
    # Try to download each day's data
    for day in range(1, days_in_month + 1):
        # Skip future dates
        current_date = datetime(year, month, day)
        if current_date > datetime.now():
            logging.info(f"Skipping future date: {current_date.strftime('%Y-%m-%d')}")
            continue
            
        # Format day with leading zero
        day_str = f"{day:02d}"
        
        # Construct the URL for daily file
        date_str = f"{year}-{month:02d}-{day_str}"
        filename = f"{symbol.upper()}-{interval}-{date_str}.zip"
        url = f"{base_url}/{symbol.upper()}/{interval}/{filename}"
        
        try:
            # Try to download the file
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                # File downloaded successfully
                logging.info(f"Downloaded daily file for {date_str}")
                
                # Process the ZIP file
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                    csv_file = zip_file.namelist()[0]
                    
                    with zip_file.open(csv_file) as f:
                        # Use the same column names as in monthly downloads
                        df = pd.read_csv(f, header=None, names=[
                            'open_time', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                            'taker_buy_quote_volume', 'ignore'
                        ])
                        
                        # Convert numeric columns
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col])
                        
                        # Convert timestamp
                        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
                        
                        # Keep only essential columns
                        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                        
                        # Add to combined dataframe
                        if combined_df is None:
                            combined_df = df
                        else:
                            combined_df = pd.concat([combined_df, df])
                            
                        successful_days += 1
            else:
                logging.info(f"Daily file not available for {date_str} (status: {response.status_code})")
                unprocessed_files.append(date_str)
                
        except Exception as e:
            logging.error(f"Error downloading daily file for {date_str}: {e}")
            unprocessed_files.append(date_str)
    
    # Log summary of daily files processing
    if unprocessed_files:
        logging.warning(f"Could not process {len(unprocessed_files)} daily files for {symbol}/{interval} {year}-{month:02d}")
        print(f"       ⚠ {len(unprocessed_files)} daily files could not be processed")
    
    if combined_df is not None and not combined_df.empty:
        # Sort by timestamp and remove duplicates
        combined_df = combined_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
        print(f"       ✅ Successfully downloaded {successful_days}/{days_in_month} daily files with {len(combined_df)} candles")
        return combined_df
    else:
        logging.warning(f"No data available from daily files for {symbol}/{interval} {year}-{month:02d}")
        print(f"       ❌ No data available from daily files")
        return None

def calculate_and_save_indicators(df, symbol, interval):
    """
    Calculate technical indicators for a DataFrame and save to database
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Trading pair symbol
        interval: Time interval
    """
    if df is None or df.empty:
        logging.warning(f"No data to calculate indicators for {symbol}/{interval}")
        return
        
    # Calculate technical indicators
    df = indicators.add_bollinger_bands(df)
    df = indicators.add_rsi(df)
    df = indicators.add_macd(df)
    df = indicators.add_ema(df)
    df = indicators.add_atr(df)
    df = indicators.add_stochastic(df)
    
    # Calculate trading signals
    df = evaluate_buy_sell_signals(df)
    
    # Save to database
    save_historical_data(df, symbol, interval)
    save_indicators(df, symbol, interval)
    
    logging.info(f"Calculated and saved indicators for {symbol}/{interval} ({len(df)} records)")

def backfill_symbol_interval(symbol, interval, years_back=3):
    """
    Backfill data for a symbol and interval
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '4h', '1d')
        years_back: Number of years to look back for data
    """
    logging.info(f"Processing {symbol} {interval}")
    print(f"\nDownloading data for {symbol} {interval}")

    # Get the current year and month
    now = datetime.now()
    current_year = now.year
    current_month = now.month
    
    # Calculate the starting year and month
    start_year = current_year - years_back
    start_month = current_month
    
    # Get last timestamp from database (if any)
    conn = get_db_connection()
    if conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT MAX(timestamp) FROM historical_data 
            WHERE symbol = %s AND interval = %s
        """, (symbol, interval))
        result = cur.fetchone()
        last_timestamp = result[0] if result and result[0] else None
        conn.close()
    else:
        last_timestamp = None

    # If we have data in database, use that to determine start date
    if last_timestamp:
        logging.info(f"Found existing data for {symbol}/{interval}, last timestamp: {last_timestamp}")
        if last_timestamp.year > start_year or (last_timestamp.year == start_year and last_timestamp.month > start_month):
            # We already have data more recent than our default start date
            # Just go back 1 month from the last timestamp to ensure overlap
            start_year = last_timestamp.year
            start_month = last_timestamp.month - 1
            if start_month <= 0:
                start_month = 12
                start_year -= 1
                
    # Process each month from start date to now
    for year in range(start_year, current_year + 1):
        # Determine which months to process for this year
        months_to_process = []
        if year == start_year:
            # Start from the specified start month
            months_to_process = list(range(start_month, 13))
        elif year == current_year:
            # End with the current month
            months_to_process = list(range(1, current_month + 1))
        else:
            # Process all months for years in between
            months_to_process = list(range(1, 13))
        
        # Download data for each month
        for month in months_to_process:
            print(f"Downloading data for {symbol}/{interval} for {year}-{month:02d}")
            df = download_monthly_klines(symbol, interval, year, month)
            
            if df is not None and not df.empty:
                # Save to database
                calculate_and_save_indicators(df, symbol, interval)
                logging.info(f"Saved {len(df)} records for {symbol}/{interval} {year}-{month:02d}")
            else:
                logging.warning(f"No data available for {symbol}/{interval} for {year}-{month:02d}")
            
            # Respect rate limits
            time.sleep(1)

def main():
    """
    Main entry point for the Windows-compatible data download script
    """
    print("="*80)
    print("Windows-compatible Cryptocurrency Data Downloader")
    print("This script downloads data directly from Binance Data Vision")
    print("No need for Node.js or the binance-historical-data package")
    print("="*80)
    
    # Ask user for symbols to download
    print("\nWhich symbols would you like to download?")
    print(f"Default: {', '.join(DEFAULT_SYMBOLS)}")
    symbols_input = input("Enter symbols (comma-separated) or press Enter for defaults: ")
    
    if symbols_input.strip():
        symbols = [s.strip().upper() for s in symbols_input.split(',')]
    else:
        symbols = DEFAULT_SYMBOLS
    
    # Ask user for intervals
    print("\nWhich intervals would you like to download?")
    print(f"Default: {', '.join(DEFAULT_INTERVALS)}")
    print("Options: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1mo")
    intervals_input = input("Enter intervals (comma-separated) or press Enter for defaults: ")
    
    if intervals_input.strip():
        intervals = [i.strip().lower() for i in intervals_input.split(',')]
    else:
        intervals = DEFAULT_INTERVALS
    
    # Ask for years to look back
    years_input = input("\nHow many years of data would you like to download? (default: 3): ")
    years_back = int(years_input) if years_input.strip() and years_input.isdigit() else 3
    
    # Confirm with user
    print(f"\nWill download data for {len(symbols)} symbols and {len(intervals)} intervals:")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Intervals: {', '.join(intervals)}")
    print(f"Years back: {years_back}")
    
    confirm = input("\nProceed? (y/n): ")
    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # Process each symbol and interval
    start_time = datetime.now()
    total_symbols = len(symbols)
    total_intervals = len(intervals)
    current = 0
    total_combinations = total_symbols * total_intervals
    
    for symbol in symbols:
        for interval in intervals:
            current += 1
            print(f"\n[{current}/{total_combinations}] Processing {symbol}/{interval}")
            try:
                backfill_symbol_interval(symbol, interval, years_back)
            except Exception as e:
                logging.error(f"Error processing {symbol}/{interval}: {e}")
                print(f"❌ Error processing {symbol}/{interval}: {e}")
    
    # Show summary
    end_time = datetime.now()
    duration = end_time - start_time
    print("\n" + "="*80)
    print(f"Download completed in {duration}")
    print(f"Processed {total_combinations} symbol/interval combinations")
    print("="*80)

if __name__ == "__main__":
    main()