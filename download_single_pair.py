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
    
    logging.warning(f"No monthly data available for {year}-{month_str}")
    return None

def download_and_process(symbol, interval, year, month):
    """Download, process and save data for a single month"""
    # Ensure database tables exist
    create_tables()
    
    # Download the data
    df = download_single_month(symbol, interval, year, month)
    
    if df is not None and not df.empty:
        # Save to database
        try:
            save_historical_data(df, symbol, interval)
            logging.info(f"Saved {len(df)} historical data points to database")
            
            # Add technical indicators
            df = indicators.add_bollinger_bands(df)
            df = indicators.add_rsi(df)
            df = indicators.add_macd(df)
            df = indicators.add_ema(df)
            
            # Evaluate trading signals
            df = evaluate_buy_sell_signals(df)
            
            # Save indicators to database
            save_indicators(df, symbol, interval)
            logging.info(f"Saved indicators to database")
            
            return len(df)
        except Exception as e:
            logging.error(f"Error saving data to database: {e}")
            return 0
    else:
        logging.warning("No data to save")
        return 0

if __name__ == "__main__":
    # Verify the database tables exist
    create_tables()
    
    # Download a single pair's data
    symbol = "BTCUSDT"
    interval = "1d"
    year = 2024
    month = 3  # March
    
    print(f"Downloading {symbol} {interval} data for {year}-{month:02d}...")
    
    row_count = download_and_process(symbol, interval, year, month)
    
    print(f"Process completed. Added {row_count} rows to database.")
    
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
    
    print(f"Database now has {hist_count} historical data records and {ind_count} indicator records for {symbol} {interval}")
    
    cursor.close()
    conn.close()