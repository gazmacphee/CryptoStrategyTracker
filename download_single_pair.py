"""
Script to download and process a single trading pair and interval
"""

import os
import requests
import pandas as pd
import zipfile
import io
from datetime import date, datetime
import time
import logging
from database import create_tables, save_historical_data, save_indicators
import indicators
from strategy import evaluate_buy_sell_signals

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Binance Data Vision base URL
BASE_URL = "https://data.binance.vision/data/spot"

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

def process_kline_data(data_content):
    """Process kline data from a ZIP file content"""
    if not data_content:
        return None
    
    try:
        # Extract ZIP file to memory
        with zipfile.ZipFile(io.BytesIO(data_content)) as zip_file:
            # CSV files typically have the same name as the ZIP file
            csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
            if not csv_files:
                logging.warning(f"No CSV files found in the ZIP")
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
                
                # Convert timestamp columns with error handling
                try:
                    # First safely convert to integers to handle any extreme values
                    df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
                    df['close_time'] = pd.to_numeric(df['close_time'], errors='coerce')
                    
                    # Use errors='coerce' to handle out-of-bounds values by converting them to NaT
                    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', errors='coerce')
                    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', errors='coerce')
                    df['timestamp'] = df['open_time']  # More intuitive name
                    
                    # Drop any rows with invalid timestamps
                    df = df.dropna(subset=['timestamp'])
                    
                    if df.empty:
                        logging.warning("All timestamps were invalid in the downloaded data")
                        return None
                except Exception as e:
                    logging.error(f"Error converting timestamps: {e}")
                    return None
                
                # Convert string values to float
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                
                # Select relevant columns and return
                result_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                return result_df
    
    except Exception as e:
        logging.error(f"Error processing kline data: {e}")
        return None

def download_single_month(symbol, interval, year, month):
    """Download a single month of data"""
    # Format month with leading zero
    month_str = f"{month:02d}"
    
    # Monthly data URL
    monthly_url = f"{BASE_URL}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month_str}.zip"
    
    # Try to download monthly data
    content = download_file(monthly_url)
    if content:
        df = process_kline_data(content)
        if df is not None and not df.empty:
            logging.info(f"Successfully downloaded monthly data: {len(df)} rows")
            return df
        else:
            # Log the unprocessed file
            logging.warning(f"Downloaded content for {symbol}/{interval}/{year}-{month_str} but failed to process it")
            try:
                from unprocessed_files import log_unprocessed_file
                log_unprocessed_file(symbol, interval, year, month, "Failed to process ZIP content")
            except ImportError:
                # Fallback if unprocessed_files module is not available
                with open("unprocessed_files.log", "a") as f:
                    f.write(f"{symbol}/{interval}/{year}-{month_str}: Failed to process ZIP content\n")
    else:
        logging.warning(f"No monthly data available for {year}-{month_str}")
        try:
            from unprocessed_files import log_unprocessed_file
            log_unprocessed_file(symbol, interval, year, month, "Monthly file not available")
        except ImportError:
            # Fallback if unprocessed_files module is not available
            with open("unprocessed_files.log", "a") as f:
                f.write(f"{symbol}/{interval}/{year}-{month_str}: Monthly file not available\n")
    
    return None

def download_and_process(symbol, interval, year, month):
    """
    Download, process and save data for a single month
    
    Returns:
        dictionary with process results or None if failed
    """
    # Ensure database tables exist
    create_tables()
    
    results = {
        'symbol': symbol,
        'interval': interval,
        'year': year,
        'month': month,
        'downloaded': False,
        'candles': 0,
        'indicators_saved': False,
        'error': None
    }
    
    # Download the data
    start_time = time.time()
    df = download_single_month(symbol, interval, year, month)
    download_time = time.time() - start_time
    
    if df is not None and not df.empty:
        # Save to database
        try:
            results['candles'] = len(df)
            results['downloaded'] = True
            
            # Save historical data
            start_time = time.time()
            save_historical_data(df, symbol, interval)
            save_time = time.time() - start_time
            logging.info(f"Saved {len(df)} historical data points to database in {save_time:.2f}s")
            
            # Add technical indicators
            start_time = time.time()
            df = indicators.add_bollinger_bands(df)
            df = indicators.add_rsi(df)
            df = indicators.add_macd(df)
            df = indicators.add_ema(df)
            
            # Evaluate trading signals
            df = evaluate_buy_sell_signals(df)
            indicator_time = time.time() - start_time
            
            # Save indicators to database
            start_time = time.time()
            save_indicators(df, symbol, interval)
            indicator_save_time = time.time() - start_time
            
            results['indicators_saved'] = True
            results['timings'] = {
                'download': download_time,
                'save_historical': save_time,
                'calculate_indicators': indicator_time,
                'save_indicators': indicator_save_time,
                'total': download_time + save_time + indicator_time + indicator_save_time
            }
            
            logging.info(f"Saved indicators to database in {indicator_save_time:.2f}s")
            return results
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Error saving data to database: {error_msg}")
            results['error'] = error_msg
            return results
    else:
        logging.warning("No data to save")
        results['error'] = "No data found for this period"
        return results

if __name__ == "__main__":
    # Verify the database tables exist
    create_tables()
    
    # Download a single pair's data - use 2023 data which should exist
    symbol = "BTCUSDT"
    interval = "1d"
    year = 2023
    month = 12  # December
    
    print(f"\n📥 Downloading {symbol} {interval} data for {year}-{month:02d}...")
    
    results = download_and_process(symbol, interval, year, month)
    
    if results and results.get('downloaded'):
        print(f"\n✅ Process completed successfully!")
        print(f"   Downloaded: {results['candles']} candles")
        
        if 'timings' in results:
            timing = results['timings']
            print(f"\n⏱️  Performance metrics:")
            print(f"   Download time: {timing['download']:.2f}s")
            print(f"   Save to database: {timing['save_historical']:.2f}s")
            print(f"   Calculate indicators: {timing['calculate_indicators']:.2f}s")
            print(f"   Save indicators: {timing['save_indicators']:.2f}s")
            print(f"   TOTAL TIME: {timing['total']:.2f}s")
    else:
        error = results.get('error', 'Unknown error') if results else 'Failed to process'
        print(f"\n❌ Process failed: {error}")
    
    # Check the database
    from database import get_db_connection
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM historical_data WHERE symbol = %s AND interval = %s", 
                  (symbol, interval))
    hist_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM technical_indicators WHERE symbol = %s AND interval = %s", 
                  (symbol, interval))
    ind_count = cursor.fetchone()[0]
    
    print(f"\n📊 Database statistics:")
    print(f"   {hist_count} historical data records")
    print(f"   {ind_count} indicator records")
    print(f"   for {symbol} {interval}")
    
    cursor.close()
    conn.close()