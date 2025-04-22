"""
Script to backfill the database with historical cryptocurrency data.
This script runs automatically at application startup to ensure data is available.
"""

import logging
import time
import sys
import argparse
from datetime import datetime, timedelta
import os

import download_binance_data as dbd
import data_loader
from database import create_tables
from binance_api import get_available_symbols

# Configure logging
log_file = "binance_data_download.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def remove_lock_files():
    """Remove any stale lock files to ensure a clean start"""
    lock_files = ['.backfill_lock', 'backfill_progress.json.lock']
    
    for lock_file in lock_files:
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                print(f"Removed potentially stale lock file: {lock_file}")
                logging.info(f"Removed potentially stale lock file: {lock_file}")
            except Exception as e:
                logging.warning(f"Failed to remove lock file {lock_file}: {e}")
                print(f"Warning: Failed to remove lock file {lock_file}: {e}")

def get_popular_symbols(limit=10):
    """Get a list of popular cryptocurrency symbols"""
    # Default list of popular symbols if API call fails
    default_symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT",
        "XRPUSDT", "DOTUSDT", "UNIUSDT", "LTCUSDT", "LINKUSDT"
    ]
    
    if limit < len(default_symbols):
        return default_symbols[:limit]
    
    try:
        # Try to get symbols from Binance API
        symbols = get_available_symbols(quote_asset="USDT", limit=limit)
        if symbols and len(symbols) > 0:
            return symbols
    except Exception as e:
        logging.warning(f"Could not fetch symbols from API: {e}")
    
    logging.info("Using pre-defined list of popular symbols")
    return default_symbols

def backfill_database(full=False, background=False):
    """
    Run a backfill operation on the database
    
    Args:
        full: Whether to do a full backfill (more symbols, longer history)
        background: Whether this is running in the background (affects logging)
    """
    print("\n" + "*" * 80)
    print(f"DATABASE BACKFILL {'BACKGROUND ' if background else ''}PROCESS STARTING")
    print("This will download cryptocurrency historical data from Binance")
    print("*" * 80)
    
    if background:
        logging.info("Starting backfill process in background mode")
    
    # Remove any stale lock files first
    remove_lock_files()
    
    # Create tables if they don't exist
    create_tables()
    
    # Check if a backfill process is already running
    if data_loader.create_backfill_lock():
        try:
            # Set up popular symbols and intervals for backfill
            symbols = get_popular_symbols()
            intervals = ['1h', '4h', '1d']
            
            if full:
                # For full backfill, use more symbols and add weekly/monthly intervals
                symbols = get_popular_symbols(limit=20)
                intervals = ['1h', '4h', '8h', '12h', '1d', '3d', '1w', '1M']
            
            # Print some information about what we're doing
            print(f"\nPreparing to download data for {len(symbols)} cryptocurrency pairs:")
            print(f"  {', '.join(symbols)}")
            print(f"\nUsing {len(intervals)} time intervals:")
            print(f"  {', '.join(intervals)}")
            print("\nThis process may take several minutes. Progress will be shown below...")
            print("-" * 80)
            
            logging.info(f"Starting backfill for {len(symbols)} symbols and {len(intervals)} intervals")
            
            # Run backfill for all specified symbols and intervals
            total_candles = dbd.run_backfill(symbols=symbols, intervals=intervals, lookback_years=3)
            
            # Print summary
            print("\n" + "*" * 80)
            print(f"BACKFILL COMPLETED - Downloaded {total_candles} total candles")
            if total_candles > 0:
                print("Your database now contains historical cryptocurrency price data!")
            else:
                print("No new data was needed. Your database is already up to date!")
            print("*" * 80 + "\n")
            
        finally:
            # Always release the lock when done
            data_loader.release_backfill_lock()
            logging.info("Backfill process completed and lock released")
    else:
        print("\nA backfill process is already running. Current progress will be shown in the logs.\n")
        logging.info("A backfill process is already running. Skipping.")

def continuous_backfill(interval_minutes=15, full=False):
    """
    Run backfill continuously at specified intervals
    
    Args:
        interval_minutes: Minutes between updates
        full: Whether to do a full backfill
    """
    logging.info(f"Starting continuous backfill with {interval_minutes} minute intervals")
    
    while True:
        # Make sure we remove any stale lock files before each iteration
        remove_lock_files()
        
        # Run backfill
        try:
            backfill_database(full=full, background=True)
        except Exception as e:
            logging.error(f"Error in continuous backfill: {e}")
            # Force release any locks if there was an error
            if os.path.exists(".backfill_lock"):
                os.remove(".backfill_lock")
                logging.info("Backfill lock released after error in continuous mode")
        
        # Wait for next interval
        logging.info(f"Sleeping for {interval_minutes} minutes until next update")
        time.sleep(interval_minutes * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill cryptocurrency data")
    parser.add_argument("--full", action="store_true", help="Run a full backfill with more symbols and intervals")
    parser.add_argument("--background", action="store_true", help="Run in background mode with minimal output")
    parser.add_argument("--continuous", action="store_true", help="Run continuously at specified intervals")
    parser.add_argument("--interval", type=int, default=15, help="Minutes between updates in continuous mode")
    
    args = parser.parse_args()
    
    print("****************************************************************")
    print("Starting cryptocurrency database backfill process")
    print(f"Full backfill mode: {args.full}")
    print(f"Continuous mode: {args.continuous}")
    print(f"Background mode: {args.background}")
    if args.continuous:
        print(f"Update interval: {args.interval} minutes")
    print("This process ensures your database has historical price data")
    print("****************************************************************")
    
    # Clean up any stale lock files first
    remove_lock_files()
    
    # Create lock file immediately to prevent parallel execution
    with open(".backfill_lock", "w") as f:
        f.write(f"Process started at {datetime.now()}")
    
    try:
        if args.continuous:
            # Run continuous backfill
            continuous_backfill(interval_minutes=args.interval, full=args.full)
        else:
            # Run a single backfill
            backfill_database(full=args.full, background=args.background)
    except Exception as e:
        logging.error(f"Error in backfill process: {e}")
        # Make sure to release lock on error
        if os.path.exists(".backfill_lock"):
            os.remove(".backfill_lock")
            print("Backfill lock released due to error")