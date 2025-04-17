"""
Script to backfill the database by downloading data sequentially from Binance Data Vision
to avoid timeouts and ensure complete data collection.
"""

import os
import sys
import time
from datetime import datetime, date, timedelta
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backfill.log')
    ]
)

# Import the download function
from download_single_pair import download_and_process

# Default symbols and intervals
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", 
    "XRPUSDT", "SOLUSDT", "DOTUSDT", "LTCUSDT"
]

DEFAULT_INTERVALS = [
    "1h", "4h", "1d", "1w"
]

def run_sequential_backfill(symbols=None, intervals=None, lookback_months=12):
    """
    Run backfill sequentially month by month to avoid timeouts
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    
    if intervals is None:
        intervals = DEFAULT_INTERVALS
    
    # Calculate date range
    end_date = date.today()
    start_date = date(end_date.year - (lookback_months // 12), end_date.month - (lookback_months % 12), 1)
    if start_date.month <= 0:
        start_date = date(start_date.year - 1, start_date.month + 12, 1)
    
    logging.info(f"Starting sequential backfill from {start_date} to {end_date}")
    logging.info(f"Processing {len(symbols)} symbols: {', '.join(symbols)}")
    logging.info(f"Processing {len(intervals)} intervals: {', '.join(intervals)}")
    
    # Track progress
    total_rows_added = 0
    total_combinations = len(symbols) * len(intervals) * lookback_months
    processed_combinations = 0
    
    # Process each symbol, interval, and month
    current_date = start_date
    while current_date <= end_date:
        year = current_date.year
        month = current_date.month
        
        for symbol in symbols:
            for interval in intervals:
                try:
                    logging.info(f"Processing {symbol} {interval} for {year}-{month:02d}...")
                    rows_added = download_and_process(symbol, interval, year, month)
                    total_rows_added += rows_added
                    
                    processed_combinations += 1
                    completion_pct = (processed_combinations / total_combinations) * 100
                    
                    logging.info(f"Added {rows_added} rows. Progress: {processed_combinations}/{total_combinations} ({completion_pct:.1f}%)")
                    
                    # Small delay to avoid overwhelming the server
                    time.sleep(1)
                except Exception as e:
                    logging.error(f"Error processing {symbol} {interval} for {year}-{month:02d}: {e}")
        
        # Move to next month
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        
        current_date = date(year, month, 1)
        
        # Exit if we've reached the end date
        if current_date > end_date:
            break
    
    logging.info(f"Sequential backfill completed. Added {total_rows_added} total rows to database.")
    return total_rows_added

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backfill cryptocurrency data from Binance Data Vision repository')
    parser.add_argument('--symbols', nargs='+', help='List of symbols to download', default=DEFAULT_SYMBOLS)
    parser.add_argument('--intervals', nargs='+', help='List of intervals to download', default=DEFAULT_INTERVALS)
    parser.add_argument('--months', type=int, default=12, help='Number of months to look back')
    parser.add_argument('--reset', action='store_true', help='Reset database before backfill')
    
    args = parser.parse_args()
    
    # Reset database if requested
    if args.reset:
        from clean_reset_database import reset_database
        reset_database()
    
    # Run backfill
    run_sequential_backfill(
        symbols=args.symbols,
        intervals=args.intervals,
        lookback_months=args.months
    )