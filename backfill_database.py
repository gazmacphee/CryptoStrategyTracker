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
from database import create_tables, get_db_connection
from binance_api import get_available_symbols

# Import ML modules for automatic pattern detection
try:
    import advanced_ml
    ML_AVAILABLE = True
    logging.info("Advanced ML module available for pattern detection")
except ImportError:
    ML_AVAILABLE = False
    logging.warning("Advanced ML module not available - pattern detection disabled")
    
# Import ML database operations
try:
    import db_ml_operations
    ML_DB_AVAILABLE = True
except ImportError:
    logging.warning("db_ml_operations module not available - ML data will not be saved to database")
    ML_DB_AVAILABLE = False

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

def check_for_ml_analysis():
    """
    Check if we have sufficient data in the database to run machine learning analysis.
    If sufficient data is found, runs pattern detection and saves results.
    
    Returns:
        Number of patterns detected and saved
    """
    if not ML_AVAILABLE or not ML_DB_AVAILABLE:
        logging.warning("ML or DB modules not available - skipping pattern detection")
        return 0
    
    # Define the minimum data requirements for pattern analysis
    MIN_CANDLES = 5000  # Need at least this many candles for a symbol/interval
    MIN_SYMBOLS = 3     # Need at least this many symbols with sufficient data
    
    conn = get_db_connection()
    if not conn:
        logging.error("Could not connect to database to check for ML analysis")
        return 0
    
    # Check the most popular symbols for sufficient data
    symbols = get_popular_symbols(limit=8)  # Start with the most popular ones
    intervals = ['1h', '4h', '1d']          # Focus on the most commonly used intervals
    
    symbols_with_sufficient_data = []
    
    try:
        cur = conn.cursor()
        
        # Check each symbol/interval combination for sufficient data
        for symbol in symbols:
            for interval in intervals:
                cur.execute("""
                    SELECT COUNT(*) FROM historical_data 
                    WHERE symbol = %s AND interval = %s
                """, (symbol, interval))
                
                count = cur.fetchone()[0]
                
                if count >= MIN_CANDLES:
                    symbols_with_sufficient_data.append((symbol, interval))
                    logging.info(f"Found sufficient data for ML analysis: {symbol}/{interval} ({count} candles)")
                else:
                    logging.info(f"Insufficient data for ML analysis: {symbol}/{interval} ({count} candles)")
        
        conn.close()
        
        # If we have enough symbols with sufficient data, run pattern detection
        if len(symbols_with_sufficient_data) >= MIN_SYMBOLS:
            logging.info(f"Running ML pattern detection on {len(symbols_with_sufficient_data)} symbol/interval pairs")
            
            # First, train all pattern models
            try:
                training_results = advanced_ml.train_all_pattern_models()
                logging.info(f"Pattern model training completed: {training_results['successful']}/{training_results['total']} models trained")
            except Exception as e:
                logging.error(f"Error training pattern models: {e}")
                return 0
            
            # Now analyze current patterns
            try:
                patterns = advanced_ml.analyze_all_market_patterns()
                logging.info(f"Pattern analysis complete - found patterns: {not patterns.empty}")
            except Exception as e:
                logging.error(f"Error analyzing market patterns: {e}")
                return 0
            
            # Save high-confidence recommendations
            try:
                saved_count = advanced_ml.save_current_recommendations()
                logging.info(f"Saved {saved_count} pattern-based trading signals")
                return saved_count
            except Exception as e:
                logging.error(f"Error saving pattern recommendations: {e}")
                return 0
        else:
            logging.info(f"Not enough symbols with sufficient data for ML analysis. Found {len(symbols_with_sufficient_data)}, need {MIN_SYMBOLS}")
            return 0
    
    except Exception as e:
        logging.error(f"Error checking for ML analysis readiness: {e}")
        if conn:
            conn.close()
        return 0

def get_popular_symbols(limit=10):
    """Get a list of popular cryptocurrency symbols"""
    # Default list of popular symbols if API call fails
    default_symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT",
        "XRPUSDT", "DOTUSDT", "UNIUSDT", "LTCUSDT", "LINKUSDT",
        "SOLUSDT", "MATICUSDT", "AVAXUSDT", "SHIBUSDT", "TRXUSDT",
        "ATOMUSDT", "NEARUSDT", "ALGOUSDT", "FTMUSDT", "SANDUSDT",
        "MANAUSDT", "AXSUSDT", "VETUSDT", "ICPUSDT", "FILUSDT",
        "HBARUSDT", "EGLDUSDT", "THETAUSDT", "EOSUSDT", "AAVEUSDT"
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
            
            # Only use intervals 30m or larger
            intervals = ['30m', '1h', '4h', '1d']
            
            if full:
                # For full backfill, use more symbols and all available intervals ≥30m
                symbols = get_popular_symbols(limit=30)
                intervals = ['30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
            
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
                
                # Check if we have enough data for ML analysis
                if ML_AVAILABLE and ML_DB_AVAILABLE:
                    print("Checking if sufficient data is available for ML pattern detection...")
                    patterns_found = check_for_ml_analysis()
                    
                    if patterns_found > 0:
                        print(f"✅ ML pattern analysis complete - {patterns_found} trading patterns detected and saved!")
                        logging.info(f"ML pattern analysis complete - {patterns_found} patterns saved")
                    else:
                        print("ℹ️ Not enough data yet for ML pattern detection.")
                        logging.info("Not enough data yet for ML pattern detection")
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
            
            # Run ML pattern detection separately (not tied to backfill success)
            # This ensures ML runs even on incremental updates that don't add significant data
            if ML_AVAILABLE and ML_DB_AVAILABLE:
                logging.info("Running ML pattern detection as part of continuous backfill...")
                try:
                    patterns_found = check_for_ml_analysis()
                    if patterns_found > 0:
                        logging.info(f"Continuous ML analysis complete - {patterns_found} patterns saved")
                    else:
                        logging.info("Continuous ML analysis - no new patterns detected")
                except Exception as ml_error:
                    logging.error(f"Error running ML analysis in continuous mode: {ml_error}")
            
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