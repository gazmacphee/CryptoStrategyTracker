"""
Script to download historical cryptocurrency data using binance-historical-data npm package
"""

import os
import sys
import time
import json
import subprocess
import requests
import io
import zipfile
import calendar
from datetime import datetime, timedelta
import pandas as pd
import logging

from database import save_historical_data, get_db_connection, save_indicators
import indicators
from strategy import evaluate_buy_sell_signals
from trading_signals import save_trading_signals, create_signals_table
from binance_file_listing import get_available_kline_files

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Symbols and intervals to download
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT",
    "SOLUSDT", "DOTUSDT", "LTCUSDT", "LINKUSDT", "MATICUSDT", "AVAXUSDT"
]

INTERVALS = ['15m', '30m', '1h', '4h', '1d']

def check_binance_historical_data_installed():
    """Check if binance-historical-data is installed"""
    # Possible paths for Windows users
    windows_paths = [
        # Standard global npm path
        os.path.join(os.environ.get('APPDATA', ''), 'npm', 'binance-historical-data.cmd'),
        # NPM global bin directory
        os.path.join(os.environ.get('ProgramFiles', ''), 'nodejs', 'binance-historical-data.cmd'),
        # Alternative path
        os.path.join(os.environ.get('APPDATA', ''), 'npm', 'node_modules', 'binance-historical-data', 'bin', 'binance-historical-data')
    ]
    
    # For Windows, first try direct paths
    if os.name == 'nt':  # Windows
        for path in windows_paths:
            if os.path.exists(path):
                logging.info(f"Found binance-historical-data at: {path}")
                return path
    
    # Standard method for non-Windows or if direct path not found
    try:
        result = subprocess.run(['binance-historical-data', '--version'], 
                      capture_output=True, text=True)
        if result.returncode == 0:
            return 'binance-historical-data'  # Command works directly
    except FileNotFoundError:
        logging.error("binance-historical-data not found. Installing...")
        try:
            subprocess.run(['npm', 'install', '-g', 'binance-historical-data'], 
                         check=True)
            # Try to find the path after installation
            if os.name == 'nt':  # Windows
                for path in windows_paths:
                    if os.path.exists(path):
                        logging.info(f"Found binance-historical-data at: {path}")
                        return path
            return 'binance-historical-data'  # Assume it's in PATH after install
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install binance-historical-data: {e}")
            return False
    except Exception as e:
        logging.error(f"Error checking binance-historical-data: {e}")
        return False
    
    return False

def download_klines(symbol, interval, start_date=None):
    """Download klines using binance-historical-data"""
    # Get the actual command/path to use
    binance_cmd = check_binance_historical_data_installed()
    if not binance_cmd:
        raise RuntimeError("binance-historical-data package not available")
        
    try:
        # Use the detected path or command
        if isinstance(binance_cmd, str):
            cmd = [binance_cmd, '--symbol', symbol, '--interval', interval]
        else:
            logging.error("Invalid binance-historical-data command type")
            return None
            
        if start_date:
            cmd.extend(['--startDate', start_date.strftime('%Y-%m-%d')])
            
        logging.info(f"Executing command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logging.error(f"Error downloading data: {result.stderr}")
            return None

        # Parse the output into DataFrame
        data = json.loads(result.stdout)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamps to datetime with error handling
        try:
            # First safely convert to integers to handle any extreme values
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            
            # Use errors='coerce' to handle out-of-bounds values by converting them to NaT
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
            
            # Drop any rows with invalid timestamps
            df = df.dropna(subset=['timestamp'])
            
            if df.empty:
                logging.warning("All timestamps were invalid in the downloaded data")
                return None
        except Exception as e:
            logging.error(f"Error converting timestamps: {e}")
            return None
            
        return df

    except Exception as e:
        logging.error(f"Error downloading klines: {e}")
        return None

def process_and_save_chunk(df, symbol, interval):
    """Process and save a chunk of data"""
    if df is None or df.empty:
        return

    # Calculate indicators
    df = indicators.add_bollinger_bands(df)
    df = indicators.add_rsi(df)
    df = indicators.add_macd(df)
    df = indicators.add_ema(df)

    # Calculate signals
    df = evaluate_buy_sell_signals(df)

    # Save to database
    save_historical_data(df, symbol, interval)

def backfill_symbol_interval(symbol, interval):
    """Backfill data for a symbol and interval"""
    logging.info(f"Processing {symbol} {interval}")

    # Get last timestamp from database
    conn = get_db_connection()
    if conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT MAX(timestamp) FROM historical_data 
            WHERE symbol = %s AND interval = %s
        """, (symbol, interval))
        last_timestamp = cur.fetchone()[0]
        conn.close()
    else:
        last_timestamp = None

    # Get the current year and month
    now = datetime.now()
    current_year = now.year
    current_month = now.month
    
    # Try to use binance-historical-data (npm package) first
    try:
        # Download data in chunks
        start_date = datetime(2017, 1, 1) if not last_timestamp else last_timestamp
        current_date = start_date

        while current_date < datetime.now():
            df = download_klines(symbol, interval, current_date)
            if df is not None and not df.empty:
                process_and_save_chunk(df, symbol, interval)
                current_date = df['timestamp'].max() + timedelta(minutes=1)
                logging.info(f"Processed up to {current_date}")
            else:
                logging.warning(f"No data available via npm tool for {symbol} {interval} from {current_date}")
                break
            time.sleep(1)  # Rate limiting
    except Exception as e:
        logging.warning(f"Failed to use binance-historical-data npm package: {e}")
        logging.info(f"Falling back to direct download method for {symbol}/{interval}")
        
        # Fallback to direct downloads from Binance Data Vision
        # Start from 3 years ago
        start_year = now.year - 3
        start_month = now.month
        
        # Special handling for older data if we need it
        if last_timestamp and last_timestamp.year < start_year:
            start_year = last_timestamp.year
            start_month = last_timestamp.month
        
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
                logging.info(f"Downloading data for {symbol}/{interval} for {year}-{month:02d}")
                df = download_monthly_klines(symbol, interval, year, month)
                
                if df is not None and not df.empty:
                    # Save to database
                    save_historical_data(df, symbol, interval)
                    calculate_and_save_indicators(df, symbol, interval)
                    logging.info(f"Saved {len(df)} records for {symbol}/{interval} {year}-{month:02d}")
                else:
                    logging.warning(f"No data available for {symbol}/{interval} for {year}-{month:02d}")
                
                # Respect rate limits
                time.sleep(1)

def run_backfill(symbols=None, intervals=None, lookback_years=3):
    """
    Run backfill for specified symbols and intervals
    
    Args:
        symbols: List of symbols to process (default: uses predefined SYMBOLS)
        intervals: List of intervals to process (default: uses predefined INTERVALS)
        lookback_years: Number of years to look back for historical data
    
    Returns:
        Total number of candles downloaded
    """
    # Use default values if parameters not provided
    symbols_to_process = symbols if symbols is not None else SYMBOLS
    intervals_to_process = intervals if intervals is not None else INTERVALS
    
    logging.info(f"Running backfill for {len(symbols_to_process)} symbols and {len(intervals_to_process)} intervals")
    logging.info(f"Symbols: {', '.join(symbols_to_process)}")
    logging.info(f"Intervals: {', '.join(intervals_to_process)}")
    
    total_candles = 0
    
    for symbol in symbols_to_process:
        for interval in intervals_to_process:
            # Process the symbol/interval combination
            try:
                backfill_symbol_interval(symbol, interval)
                # Add approximate candle count based on interval and lookback years
                if interval == '1h':
                    total_candles += 24 * 365 * lookback_years
                elif interval == '4h':
                    total_candles += 6 * 365 * lookback_years
                elif interval == '1d':
                    total_candles += 365 * lookback_years
                else:
                    total_candles += 100  # Generic estimate for other intervals
            except Exception as e:
                logging.error(f"Error processing {symbol}/{interval}: {e}")
            
            time.sleep(2)  # Rate limiting between symbol/interval pairs
    
    return total_candles

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
                    
                    # Print sample data types for debugging
                    logging.debug(f"Dataframe dtypes: {df.dtypes}")
                    logging.debug(f"Sample open_time values: {df['open_time'].head(3).tolist()}")
                    
                    # Convert numeric data types
                    df['open'] = pd.to_numeric(df['open'])
                    df['high'] = pd.to_numeric(df['high'])
                    df['low'] = pd.to_numeric(df['low'])
                    df['close'] = pd.to_numeric(df['close'])
                    df['volume'] = pd.to_numeric(df['volume'])
                    
                    # Convert timestamps with more robust error handling
                    try:
                        # Print a sample of the data for debugging
                        logging.debug(f"Sample data for monthly file: {df['open_time'].head().tolist()}")
                        
                        # Handle potential string data by first trying to convert to numeric
                        if df['open_time'].dtype == object:  # if it's a string or object
                            logging.info(f"Converting string timestamps for {symbol}/{interval} {year}-{month:02d}")
                            df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
                        
                        # Try millisecond timestamps first (standard Binance format)
                        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', errors='coerce')
                        
                        # If we have NaT values, try other formats
                        if df['timestamp'].isna().any():
                            logging.info(f"Trying alternative timestamp formats for {symbol}/{interval} {year}-{month:02d}")
                            
                            # Try seconds format
                            mask = df['timestamp'].isna()
                            df.loc[mask, 'timestamp'] = pd.to_datetime(df.loc[mask, 'open_time'], unit='s', errors='coerce')
                            
                            # Try direct conversion (ISO format)
                            mask = df['timestamp'].isna()
                            if mask.any():
                                df.loc[mask, 'timestamp'] = pd.to_datetime(df.loc[mask, 'open_time'], errors='coerce')
                        
                        # For any remaining NaT values, generate timestamps based on file date
                        if df['timestamp'].isna().any():
                            logging.warning(f"Some timestamps still invalid for {symbol}/{interval} {year}-{month:02d} - creating valid timestamps")
                            start_date = datetime(year, month, 1)
                            end_date = datetime(year, month, calendar.monthrange(year, month)[1])
                            
                            # Count missing values
                            missing_count = df['timestamp'].isna().sum()
                            
                            # Create timestamp range
                            total_minutes = (end_date - start_date).total_seconds() / 60
                            interval_minutes = int(total_minutes / (missing_count + 1))
                            
                            # Generate timestamps
                            na_indices = df['timestamp'].isna()
                            synthetic_timestamps = []
                            
                            for i in range(missing_count):
                                new_ts = start_date + timedelta(minutes=i * interval_minutes)
                                synthetic_timestamps.append(new_ts)
                                
                            df.loc[na_indices, 'timestamp'] = synthetic_timestamps
                        
                        # Final check for empty dataframe (should never happen now)
                        if df.empty:
                            logging.warning(f"No valid data found for {symbol}/{interval} {year}-{month:02d}")
                            return None
                    except Exception as e:
                        logging.error(f"Error converting timestamps for {symbol}/{interval} {year}-{month:02d}: {e}")
                        logging.error(f"Timestamp sample: {df['open_time'].head(3).tolist() if not df.empty else 'Empty dataframe'}")
                        return None
                    
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
    
    # Save indicators to database
    save_indicators(df, symbol, interval)
    
    logging.info(f"Calculated and saved indicators for {symbol}/{interval} ({len(df)} records)")

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
    import pandas as pd
    
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
                        # Use the same column names as in monthly downloads with explicit dtypes
                        df = pd.read_csv(f, header=None, names=[
                            'open_time', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                            'taker_buy_quote_volume', 'ignore'
                        ], dtype={
                            'open_time': 'int64',  # Force this to be int64
                            'close_time': 'int64'  # Force this to be int64 too
                        })
                        
                        # Print sample data types for debugging
                        logging.debug(f"Daily file dtypes: {df.dtypes}")
                        logging.debug(f"Daily file sample open_time values: {df['open_time'].head(3).tolist()}")
                        
                        # Convert numeric columns
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col])
                        
                        # Convert timestamps with more robust error handling
                        try:
                            # Print a sample of the data for debugging
                            logging.debug(f"Sample data for {date_str}: {df['open_time'].head().tolist()}")
                            
                            # Handle potential string data by first trying to convert to numeric
                            if df['open_time'].dtype == object:  # if it's a string or object
                                logging.info(f"Converting string timestamps for {date_str}")
                                df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
                            
                            # Try millisecond timestamps first (standard Binance format)
                            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', errors='coerce')
                            
                            # If we have NaT values, try other formats
                            if df['timestamp'].isna().any():
                                logging.info(f"Trying alternative timestamp formats for {date_str}")
                                
                                # Try seconds format
                                mask = df['timestamp'].isna()
                                df.loc[mask, 'timestamp'] = pd.to_datetime(df.loc[mask, 'open_time'], unit='s', errors='coerce')
                                
                                # Try direct conversion (ISO format)
                                mask = df['timestamp'].isna()
                                if mask.any():
                                    df.loc[mask, 'timestamp'] = pd.to_datetime(df.loc[mask, 'open_time'], errors='coerce')
                            
                            # For any remaining NaT values, try to fill with sensible timestamps based on the file date
                            # This ensures we never lose data due to timestamp issues
                            if df['timestamp'].isna().any():
                                logging.warning(f"Some timestamps still invalid for {date_str} - using file date")
                                file_date = pd.to_datetime(date_str)
                                # Create sequence of timestamps through the day
                                count_na = df['timestamp'].isna().sum()
                                time_increment = pd.Timedelta(minutes=60) # For daily bars, hourly increments
                                
                                # Generate synthetic timestamps based on file date
                                na_indices = df['timestamp'].isna()
                                synthetic_timestamps = [file_date + (i * time_increment) for i in range(count_na)]
                                df.loc[na_indices, 'timestamp'] = synthetic_timestamps
                            
                            # Final check for empty dataframe
                            if df.empty:
                                logging.warning(f"No valid data found for daily file {date_str}")
                                unprocessed_files.append(date_str)
                                continue
                                
                        except Exception as e:
                            logging.error(f"Error converting timestamps for {date_str}: {e}")
                            logging.error(f"Timestamp sample: {df['open_time'].head(3).tolist() if not df.empty else 'Empty dataframe'}")
                            unprocessed_files.append(date_str)
                            continue
                        
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
        # Try to use the specialized module for logging unprocessed files
        try:
            from unprocessed_files import log_unprocessed_file
            for date_str in unprocessed_files:
                log_unprocessed_file(symbol, interval, year, int(date_str.split('-')[2]), 
                                    f"Daily file could not be processed")
        except ImportError:
            # Fall back to simple logging if module not available
            with open("unprocessed_files.log", "a") as f:
                for date_str in unprocessed_files:
                    f.write(f"{symbol}/{interval}/{date_str}: Daily file could not be processed\n")
        
        logging.warning(f"Could not process {len(unprocessed_files)} daily files for {symbol}/{interval} {year}-{month:02d}")
        print(f"       ⚠ {len(unprocessed_files)} daily files could not be processed (see unprocessed_files.log)")
    
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
        return
        
    try:
        # Calculate indicators
        with_bb = indicators.add_bollinger_bands(df)
        with_rsi = indicators.add_rsi(with_bb)
        with_macd = indicators.add_macd(with_rsi)
        with_ema = indicators.add_ema(with_macd)
        
        # Save historical data
        save_historical_data(df, symbol, interval)
        
        # Save indicators
        save_indicators(with_ema, symbol, interval)
        
        return with_ema
    except Exception as e:
        logging.error(f"Error calculating indicators for {symbol}/{interval}: {e}")
        return None

if __name__ == "__main__":
    run_backfill()