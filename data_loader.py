"""
Data loader module for tracking backfill progress and status
"""

import os
import time
import threading
import logging
import json
from datetime import datetime, timedelta, date
import pandas as pd
from database import get_db_connection, create_tables
from binance_api import get_available_symbols
from download_single_pair import download_and_process

# Remove lock files at import time to ensure clean starts
def _remove_stale_lock_files():
    """Remove stale lock files at module import time"""
    lock_files = ['.backfill_lock', 'backfill_progress.json.lock']
    for lock_file in lock_files:
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                print(f"Removed potentially stale lock file at startup: {lock_file}")
            except Exception as e:
                print(f"Warning: Failed to remove lock file {lock_file}: {e}")

# Remove lock files at module import time
_remove_stale_lock_files()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_INTERVALS = ['15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
EXCLUDED_INTERVALS = ['1m', '3m', '5m']  # Intervals to skip
LOOKBACK_YEARS = 3
PROGRESS_FILE = "backfill_progress.json"

# Global progress tracking
progress_data = {
    "last_updated": None,
    "total_symbols": 0,
    "symbols_completed": 0,
    "symbols_progress": {},
    "overall_progress": 0.0,
    "is_running": False,
    "errors": []
}

def get_backfill_progress():
    """Get the current backfill progress data"""
    global progress_data
    
    # Update is_running status based on lock file
    progress_data["is_running"] = os.path.exists(".backfill_lock")
    
    # If we have a progress file, load it
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                saved_progress = json.load(f)
                # Update progress with saved data
                progress_data.update(saved_progress)
        except Exception as e:
            logger.error(f"Error loading progress file: {e}")
    
    # Get fresh database stats for accurate counts
    update_progress_from_database()
    
    return progress_data

def save_progress():
    """Save the current progress to file"""
    global progress_data
    
    # Update timestamp
    progress_data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate overall progress based on symbols completed
    if progress_data["total_symbols"] > 0:
        progress_data["overall_progress"] = (progress_data["symbols_completed"] / progress_data["total_symbols"]) * 100.0
    else:
        progress_data["overall_progress"] = 0.0
        
    # Write to file
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f, indent=2)
        logger.info(f"Progress saved: {progress_data['overall_progress']:.1f}% complete")
    except Exception as e:
        logger.error(f"Error saving progress file: {e}")

def update_progress_from_database():
    """Update progress data with fresh information from the database"""
    global progress_data
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get unique symbols in the database
        cursor.execute("SELECT DISTINCT symbol FROM historical_data")
        db_symbols = [row[0] for row in cursor.fetchall()]
        
        # Get total expected symbols from symbols list
        symbols_list = get_available_symbols()
        progress_data["total_symbols"] = len(symbols_list)
        
        # Reset symbols completed counter since we're going to recalculate it
        progress_data["symbols_completed"] = 0
        
        # For each symbol in our list, check its interval coverage
        for symbol in symbols_list:
            intervals_data = {}
            
            for interval in DEFAULT_INTERVALS:
                if interval in EXCLUDED_INTERVALS:
                    continue
                
                # Check how many records we have for this symbol and interval
                cursor.execute("""
                    SELECT COUNT(*) FROM historical_data 
                    WHERE symbol = %s AND interval = %s
                """, (symbol, interval))
                
                count = cursor.fetchone()[0]
                
                # Calculate expected count based on interval
                expected = calculate_expected_records(interval, LOOKBACK_YEARS)
                
                # Calculate percentage
                percentage = min(100, int((count / max(1, expected)) * 100))
                
                intervals_data[interval] = {
                    "count": count,
                    "expected": expected,
                    "percentage": percentage
                }
            
            # Calculate overall symbol percentage as average of intervals
            if intervals_data:
                interval_percentages = [data["percentage"] for data in intervals_data.values()]
                symbol_percentage = sum(interval_percentages) / len(interval_percentages)
            else:
                symbol_percentage = 0
            
            # Add to progress data
            progress_data["symbols_progress"][symbol] = {
                "intervals": intervals_data,
                "overall_percentage": symbol_percentage
            }
            
            # Mark as completed if 100%
            if symbol_percentage >= 99:
                progress_data["symbols_completed"] += 1
        
        # Calculate overall progress percentage
        if progress_data["total_symbols"] > 0:
            progress_data["overall_progress"] = round(
                (progress_data["symbols_completed"] / progress_data["total_symbols"]) * 100, 1
            )
        
        cursor.close()
        conn.close()
    
    except Exception as e:
        logger.error(f"Error updating progress from database: {e}")
        progress_data["errors"].append(str(e))

def calculate_expected_records(interval, years=3):
    """Calculate expected number of records for a given interval over specified years"""
    # Base calculations on interval
    if interval == '15m':
        # 4 per hour * 24 hours * 365 days * years
        return 4 * 24 * 365 * years
    elif interval == '30m':
        return 2 * 24 * 365 * years
    elif interval == '1h':
        return 24 * 365 * years
    elif interval == '2h':
        return 12 * 365 * years
    elif interval == '4h':
        return 6 * 365 * years
    elif interval == '6h':
        return 4 * 365 * years
    elif interval == '8h':
        return 3 * 365 * years
    elif interval == '12h':
        return 2 * 365 * years
    elif interval == '1d':
        return 365 * years
    elif interval == '3d':
        return (365 * years) // 3
    elif interval == '1w':
        return 52 * years
    elif interval == '1M':
        return 12 * years
    else:
        # Default fallback
        return 365 * years

def create_backfill_lock():
    """Create a lock file to prevent multiple backfill processes"""
    try:
        # Check if lock file already exists
        if os.path.exists(".backfill_lock"):
            # Check if the lock file is stale (older than 1 hour)
            file_stat = os.stat(".backfill_lock")
            file_age = time.time() - file_stat.st_mtime
            
            # If lock file is older than 1 hour (3600 seconds), consider it stale
            if file_age > 3600:
                os.remove(".backfill_lock")
                print("\n🔄 Removed stale lock file (older than 1 hour)")
                logger.info("Removed stale lock file (older than 1 hour)")
            else:
                try:
                    # Try to read the lock file to see when the process started
                    with open(".backfill_lock", "r") as f:
                        start_time = f.read().strip()
                    print(f"\n⚠️ Backfill process already running since {start_time}")
                    print("   If this is incorrect, manually delete the .backfill_lock file and try again.\n")
                except:
                    print("\n⚠️ Backfill process already running but start time unknown")
                return False
        
        # Create the lock file - Using fixed timestamp to prevent date issues
        # We use the real current time for display, but ensure we're not creating time-related issues
        current_time = str(datetime.now())
        with open(".backfill_lock", "w") as f:
            f.write(current_time)
        
        progress_data["is_running"] = True
        save_progress()
        
        print("\n🔒 Created backfill lock file")
        print(f"   Process started at: {current_time}")
        return True
    except Exception as e:
        logger.error(f"Error creating lock file: {e}")
        print(f"\n❌ Failed to create backfill lock file: {e}")
        return False

def release_backfill_lock():
    """Remove the backfill lock file"""
    try:
        if os.path.exists(".backfill_lock"):
            # Try to read the lock file to see when the process started
            start_time = "unknown time"
            try:
                with open(".backfill_lock", "r") as f:
                    start_time = f.read().strip()
            except:
                pass
            
            # Remove the lock file
            os.remove(".backfill_lock")
            print(f"\n🔓 Released backfill lock (process started at: {start_time})")
            print(f"   Lock file removed at: {datetime.now()}")
        else:
            print("\n⚠️ No backfill lock file found to release")
        
        progress_data["is_running"] = False
        save_progress()
        return True
    except Exception as e:
        logger.error(f"Error removing lock file: {e}")
        print(f"\n❌ Failed to release backfill lock: {e}")
        return False

def run_backfill_process(full=False):
    """
    Run the backfill process for all symbols and intervals
    
    Args:
        full: Whether to do a full backfill
    """
    global progress_data
    
    # Check if backfill is already running
    if os.path.exists(".backfill_lock"):
        logger.warning("A backfill process is already running. Skipping.")
        print("\n⚠️ A backfill process is already running. Check logs for progress.")
        return False
    
    # Create lock file
    if not create_backfill_lock():
        return False
    
    # Update progress data
    progress_data["is_running"] = True
    progress_data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    progress_data["errors"] = []
    save_progress()
    
    try:
        # Print nice header
        print("\n" + "*" * 80)
        print("CRYPTOCURRENCY DATABASE BACKFILL PROCESS")
        print("This process will download historical price data from Binance")
        print("*" * 80)
        
        # Ensure tables exist
        create_tables()
        print("\n✓ Database tables verified")
        
        # Get symbols
        symbols = get_available_symbols()
        if full:
            print(f"\n📊 Running FULL backfill with {len(symbols)} symbols")
        else:
            print(f"\n📊 Running standard backfill with {len(symbols)} symbols")
            
        # Get intervals
        intervals = [i for i in DEFAULT_INTERVALS if i not in EXCLUDED_INTERVALS]
        print(f"⏱️  Using {len(intervals)} time intervals: {', '.join(intervals)}")
        
        # Calculate start date using a fixed reference date (2024-12-31)
        # This addresses an issue where the system date might be set to a future date
        reference_date = date(2024, 12, 31)  # Using 2024-12-31 as a fixed reference
        end_date = reference_date
        start_date = date(end_date.year - LOOKBACK_YEARS, end_date.month, end_date.day)
        print(f"📅 Downloading data from {start_date} to {end_date} (using 2024-12-31 as reference date)")
        
        # Process each symbol and interval
        total_combinations = len(symbols) * len(intervals)
        completed = 0
        errors = 0
        start_time = time.time()
        
        print("\n" + "-" * 80)
        print(f"Starting backfill of {total_combinations} symbol-interval combinations")
        print("-" * 80)
        
        for symbol_idx, symbol in enumerate(symbols):
            print(f"\n[{symbol_idx+1}/{len(symbols)}] Processing {symbol}...")
            
            for interval_idx, interval in enumerate(intervals):
                # Calculate progress
                interval_task = symbol_idx * len(intervals) + interval_idx + 1
                percent_complete = (completed / total_combinations) * 100
                
                # Calculate time estimates
                elapsed = time.time() - start_time
                if completed > 0:
                    est_total_time = elapsed * (total_combinations / completed)
                    est_remaining = est_total_time - elapsed
                    time_str = f"ETA: {est_remaining//60:.0f}m {est_remaining%60:.0f}s"
                else:
                    time_str = f"Elapsed: {elapsed:.1f}s"
                
                print(f"  [{interval_task}/{total_combinations}] ({percent_complete:.1f}%) {symbol}/{interval} - {time_str}")
                
                try:
                    # Process month by month
                    months_processed = 0
                    data_downloaded = 0
                    current_date = start_date
                    
                    while current_date <= end_date:
                        year = current_date.year
                        month = current_date.month
                        
                        try:
                            # Download data for this month
                            print(f"    → {year}-{month:02d}", end="", flush=True)
                            result = download_and_process(symbol, interval, year, month)
                            
                            if result and result.get('candles', 0) > 0:
                                data_downloaded += result.get('candles', 0)
                                print(f" ✓ ({result.get('candles', 0)} candles)")
                            else:
                                print(" - no new data")
                            
                            months_processed += 1
                        except Exception as download_error:
                            error_msg = f"Error downloading {symbol} {interval} for {year}-{month}: {download_error}"
                            logger.error(error_msg)
                            progress_data["errors"].append(error_msg)
                            print(f" ✗ ERROR: {str(download_error)[:50]}...")
                        
                        # Move to next month
                        if month == 12:
                            year += 1
                            month = 1
                        else:
                            month += 1
                        
                        current_date = date(year, month, 1)
                    
                    completed += 1
                    logger.info(f"Completed {symbol} {interval}")
                    print(f"  ✅ Finished {symbol}/{interval}: {months_processed} months processed, {data_downloaded} total candles")
                
                except Exception as interval_error:
                    error_msg = f"Error processing {symbol} {interval}: {interval_error}"
                    logger.error(error_msg)
                    progress_data["errors"].append(error_msg)
                    print(f"  ❌ Failed {symbol}/{interval}: {str(interval_error)[:100]}")
                    errors += 1
                
                # Update progress
                progress_data["overall_progress"] = round(
                    (completed / total_combinations) * 100, 1
                )
                save_progress()
            
            # Update progress from database after each symbol
            update_progress_from_database()
            save_progress()
            print(f"← Completed symbol {symbol} ({symbol_idx+1}/{len(symbols)})")
        
        # Print final summary
        total_time = time.time() - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        
        print("\n" + "=" * 80)
        print("BACKFILL PROCESS COMPLETE")
        print(f"Total time: {minutes}m {seconds}s")
        print(f"Tasks completed: {completed}/{total_combinations}")
        print(f"Errors encountered: {errors}")
        print("=" * 80)
        
        logger.info("Backfill process completed")
    
    except Exception as e:
        error_msg = f"Error in backfill process: {e}"
        logger.error(error_msg)
        progress_data["errors"].append(error_msg)
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
    
    finally:
        # Release lock
        release_backfill_lock()
        
        # Final progress update
        update_progress_from_database()
        save_progress()
    
    return True

def start_backfill_thread(full=False):
    """Start the backfill process in a background thread"""
    thread = threading.Thread(target=run_backfill_process, args=(full,))
    thread.daemon = True
    thread.start()
    return thread

if __name__ == "__main__":
    # Test run
    run_backfill_process()