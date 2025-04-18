"""
Script to download historical cryptocurrency data using binance-historical-data npm package
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime, timedelta
import pandas as pd
import logging

from database import save_historical_data, get_db_connection
import indicators
from strategy import evaluate_buy_sell_signals

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
    try:
        subprocess.run(['binance-historical-data', '--version'], 
                      capture_output=True, text=True)
        return True
    except FileNotFoundError:
        logging.error("binance-historical-data not found. Installing...")
        try:
            subprocess.run(['npm', 'install', '-g', 'binance-historical-data'], 
                         check=True)
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install binance-historical-data: {e}")
            return False
    except Exception as e:
        logging.error(f"Error checking binance-historical-data: {e}")
        return False

def download_klines(symbol, interval, start_date=None):
    """Download klines using binance-historical-data"""
    if not check_binance_historical_data_installed():
        raise RuntimeError("binance-historical-data package not available")
        
    try:
        cmd = ['binance-historical-data', '--symbol', symbol, '--interval', interval]
        if start_date:
            cmd.extend(['--startDate', start_date.strftime('%Y-%m-%d')])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logging.error(f"Error downloading data: {result.stderr}")
            return None

        # Parse the output into DataFrame
        data = json.loads(result.stdout)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
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
            logging.warning(f"No data available for {symbol} {interval} from {current_date}")
            break
        time.sleep(1)  # Rate limiting

def run_backfill():
    """Run backfill for all symbols and intervals"""
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            backfill_symbol_interval(symbol, interval)
            time.sleep(2)  # Rate limiting between symbol/interval pairs

if __name__ == "__main__":
    run_backfill()