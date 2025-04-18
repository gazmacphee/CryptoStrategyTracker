#!/usr/bin/env python3
"""
Script to trigger improved backfill process at application startup.
This script will:
1. Check which data is already populated in the database
2. Determine available data files on Binance for each symbol and interval
3. Download missing data in chronological order (oldest first)
4. Calculate indicators and signals with proper lookback data
"""

import os
import sys
import time
import logging
import json
import threading
from datetime import datetime, date
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('improved_backfill.log')
    ]
)

# Import project modules
from database import create_tables
from download_binance_data import SYMBOLS, INTERVALS, download_monthly_klines, calculate_and_save_indicators
from binance_file_listing import get_date_range_for_symbol_interval
from data_loader import create_backfill_lock, release_backfill_lock

# Lock file
LOCK_FILE = ".backfill_lock"
PROGRESS_FILE = "backfill_progress.json"

def initialize_progress_file():
    """Initialize progress tracking file"""
    progress_data = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_symbols": len(SYMBOLS),
        "symbols_completed": 0,
        "symbols_progress": {symbol: 0.0 for symbol in SYMBOLS},
        "overall_progress": 0.0,
        "is_running": True,
        "errors": []
    }
    
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress_data, f, indent=2)
    
    return progress_data

def update_progress(symbol, interval_progress, error=None):
    """Update progress tracking file"""
    try:
        # Read existing progress
        with open(PROGRESS_FILE, 'r') as f:
            progress_data = json.load(f)
        
        # Update timestamp
        progress_data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Update symbol progress
        progress_data["symbols_progress"][symbol] = interval_progress
        
        # Calculate overall progress
        completed_symbols = sum(1 for _, progress in progress_data["symbols_progress"].items() if progress >= 100.0)
        progress_data["symbols_completed"] = completed_symbols
        
        total_progress = sum(progress_data["symbols_progress"].values()) / len(progress_data["symbols_progress"])
        progress_data["overall_progress"] = round(total_progress, 1)
        
        # Add error if provided
        if error:
            progress_data["errors"].append(error)
        
        # Write updated progress
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
    except Exception as e:
        logging.error(f"Error updating progress: {e}")

def process_symbol_interval(symbol, interval, max_months=36):
    """
    Process a single symbol and interval pair
    
    Args:
        symbol: Trading pair symbol
        interval: Time interval
        max_months: Maximum number of months to process
    
    Returns:
        Number of candles downloaded
    """
    try:
        logging.info(f"Processing {symbol}/{interval}")
        print(f"\n🔄 Processing {symbol}/{interval}")
        
        # Get the date range for available data
        min_date, max_date = get_date_range_for_symbol_interval(symbol, interval)
        
        if not min_date or not max_date:
            logging.warning(f"No data available on Binance for {symbol}/{interval}")
            print(f"  ⚠️ No data available on Binance for {symbol}/{interval}")
            return 0
        
        # Convert to date objects
        min_date = min_date.date()
        max_date = max_date.date()
        
        # Using a fixed reference date to avoid issues with future dates
        reference_date = date(2024, 12, 31)
        
        if max_date > reference_date:
            logging.info(f"Limiting max date to reference date {reference_date} (was {max_date})")
            max_date = reference_date
        
        print(f"  ℹ️ Available data for {symbol}/{interval} from {min_date} to {max_date}")
        
        # Calculate all months between min_date and max_date
        all_months = []
        current_year = min_date.year
        current_month = min_date.month
        
        while (current_year < max_date.year or
               (current_year == max_date.year and current_month <= max_date.month)):
            all_months.append((current_year, current_month))
            
            # Move to next month
            if current_month == 12:
                current_year += 1
                current_month = 1
            else:
                current_month += 1
        
        # Limit to most recent months if there are too many
        if len(all_months) > max_months:
            starting_idx = len(all_months) - max_months
            all_months = all_months[starting_idx:]
            print(f"  ℹ️ Limiting to {max_months} most recent months (from {all_months[0][0]}-{all_months[0][1]:02d})")
        
        print(f"  🔄 Processing {len(all_months)} months for {symbol}/{interval}")
        
        # Process each month
        total_candles = 0
        for idx, (year, month) in enumerate(all_months):
            # Update progress percentage
            progress_pct = (idx / len(all_months)) * 100
            update_progress(symbol, progress_pct)
            
            print(f"  📅 Month {idx+1}/{len(all_months)} - {year}-{month:02d} ({progress_pct:.1f}%)")
            
            # Download data for this month
            monthly_df = download_monthly_klines(symbol, interval, year, month)
            
            if monthly_df is not None and not monthly_df.empty:
                # Calculate indicators with proper lookback
                calculate_and_save_indicators(monthly_df, symbol, interval)
                
                total_candles += len(monthly_df)
                logging.info(f"Processed {len(monthly_df)} candles for {symbol}/{interval} {year}-{month:02d}")
            else:
                logging.warning(f"No data available for {symbol}/{interval} {year}-{month:02d}")
        
        # Mark symbol as completed
        update_progress(symbol, 100.0)
        
        print(f"✅ Completed {symbol}/{interval} - Processed {total_candles} candles in total")
        logging.info(f"Completed {symbol}/{interval} - Processed {total_candles} candles")
        
        return total_candles
    
    except Exception as e:
        error_msg = f"Error processing {symbol}/{interval}: {e}"
        logging.error(error_msg)
        print(f"❌ Error processing {symbol}/{interval}: {e}")
        
        # Log traceback for debugging
        logging.error(traceback.format_exc())
        
        # Update progress with error
        update_progress(symbol, -1.0, error_msg)
        
        return 0

def run_improved_backfill(symbols=None, intervals=None, continuous=False):
    """
    Run improved backfill process for all symbols and intervals
    
    Args:
        symbols: List of symbols to process (default: SYMBOLS)
        intervals: List of intervals to process (default: INTERVALS)
        continuous: Whether to run in continuous mode
    """
    if symbols is None:
        symbols = SYMBOLS
    
    if intervals is None:
        intervals = INTERVALS
    
    # Initialize database tables
    create_tables()
    
    # Create lock file
    create_backfill_lock()
    
    # Initialize progress tracking
    progress_data = initialize_progress_file()
    
    # Print header
    print("\n" + "=" * 80)
    print(f"IMPROVED BACKFILL PROCESS STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processing {len(symbols)} symbols × {len(intervals)} intervals = {len(symbols) * len(intervals)} tasks")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Intervals: {', '.join(intervals)}")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    
    try:
        total_candles = 0
        
        # Process each symbol
        for sym_idx, symbol in enumerate(symbols):
            sym_start_time = time.time()
            sym_candles = 0
            
            # Process each interval
            for int_idx, interval in enumerate(intervals):
                task_num = sym_idx * len(intervals) + int_idx + 1
                total_tasks = len(symbols) * len(intervals)
                
                # Print task header
                print("\n" + "-" * 60)
                print(f"Task {task_num}/{total_tasks} ({(task_num/total_tasks)*100:.1f}%)")
                print(f"Symbol: {symbol}, Interval: {interval}")
                print("-" * 60)
                
                # Process this symbol/interval pair
                candles = process_symbol_interval(symbol, interval)
                total_candles += candles
                sym_candles += candles
            
            # Print symbol summary
            sym_time = time.time() - sym_start_time
            print(f"\n📊 {symbol} complete - Processed {sym_candles} candles in {sym_time:.1f} seconds")
        
        # Print completion summary
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"BACKFILL PROCESS COMPLETED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total candles processed: {total_candles}")
        print(f"Total time: {total_time:.1f} seconds ({(total_time/60):.1f} minutes)")
        print("=" * 80 + "\n")
        
        # Update progress to mark completion
        with open(PROGRESS_FILE, 'r') as f:
            progress_data = json.load(f)
        
        progress_data["is_running"] = not continuous
        
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        if continuous:
            # Sleep for some time and then repeat
            sleep_minutes = 15
            print(f"Continuous mode enabled. Sleeping for {sleep_minutes} minutes until next update.")
            logging.info(f"Sleeping for {sleep_minutes} minutes until next update")
            time.sleep(sleep_minutes * 60)
            
            # Recursive call for continuous mode
            run_improved_backfill(symbols, intervals, continuous)
    
    except KeyboardInterrupt:
        print("\n⚠️ Backfill process interrupted by user.")
        logging.warning("Backfill process interrupted by user.")
    
    except Exception as e:
        print(f"\n❌ Critical error in backfill process: {e}")
        logging.error(f"Critical error in backfill process: {e}")
        logging.error(traceback.format_exc())
    
    finally:
        # Always release lock when done
        release_backfill_lock()

def start_background_backfill(symbols=None, intervals=None, continuous=False):
    """Start backfill in a background thread"""
    thread = threading.Thread(
        target=run_improved_backfill,
        args=(symbols, intervals, continuous)
    )
    thread.daemon = True
    thread.start()
    
    return thread

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run improved backfill process")
    parser.add_argument("--continuous", action="store_true", help="Run in continuous mode")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to process")
    parser.add_argument("--intervals", nargs="+", help="Specific intervals to process")
    parser.add_argument("--background", action="store_true", help="Run in background")
    
    args = parser.parse_args()
    
    if os.path.exists(LOCK_FILE):
        logging.warning("A backfill process is already running. Use --force to override.")
        print("⚠️ A backfill process is already running. Delete the lock file to restart.")
        sys.exit(1)
    
    if args.background:
        start_background_backfill(args.symbols, args.intervals, args.continuous)
        print("🔄 Backfill process started in background thread.")
        logging.info("Backfill process started in background thread.")
    else:
        run_improved_backfill(args.symbols, args.intervals, args.continuous)