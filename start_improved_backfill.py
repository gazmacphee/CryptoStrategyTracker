"""
Improved Backfill Module

This module provides an improved background process for backfilling
cryptocurrency data with better error handling, progress tracking,
and optimized performance.
"""

import os
import sys
import time
import logging
import threading
import pandas as pd
from datetime import datetime, timedelta

from download_binance_data import run_backfill, SYMBOLS, INTERVALS
from data_loader import get_backfill_progress, save_progress, update_progress_from_database
from database import get_db_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('improved_backfill.log')
    ]
)

def get_priority_symbols():
    """Return a list of symbols in priority order for backfilling"""
    # Major assets first, then others
    priority_symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", 
        "DOGEUSDT", "XRPUSDT", "SOLUSDT", "DOTUSDT"
    ]
    
    # Add any other symbols that aren't in the priority list
    for symbol in SYMBOLS:
        if symbol not in priority_symbols:
            priority_symbols.append(symbol)
            
    return priority_symbols

def get_priority_intervals():
    """Return a list of intervals in priority order for backfilling"""
    # Start with the most commonly used intervals, ensuring we only use intervals ≥30m
    return ["1h", "1d", "4h", "30m", "2h", "6h", "8h", "12h", "3d", "1w", "1M"]

def start_background_backfill(full=False, continuous=False, interval_minutes=15):
    """
    Start an improved background backfill process
    
    Args:
        full: Whether to perform a full backfill (more symbols, longer history)
        continuous: Whether to run the backfill continuously at regular intervals
        interval_minutes: How often to run the backfill in continuous mode (minutes)
    """
    if continuous:
        def continuous_backfill_thread():
            while True:
                try:
                    _run_background_backfill(full=full)
                    logging.info(f"Completed backfill cycle, waiting {interval_minutes} minutes before next run")
                    # Sleep for the specified interval
                    time.sleep(interval_minutes * 60)
                except Exception as e:
                    logging.error(f"Error in continuous backfill: {e}")
                    # Even on error, keep the thread alive
                    time.sleep(60)  # Wait a minute before retrying
                    
        thread = threading.Thread(
            target=continuous_backfill_thread,
            daemon=True
        )
    else:
        thread = threading.Thread(
            target=_run_background_backfill,
            args=(full,),
            daemon=True
        )
    
    thread.start()
    
    logging.info(f"Started improved background backfill process (full={full}, continuous={continuous}, interval={interval_minutes}min)")
    return True

def _run_background_backfill(full=False):
    """
    Run the improved backfill process in the background
    
    Args:
        full: Whether to perform a full backfill (more symbols, longer history)
    """
    try:
        # Get priority-ordered lists
        priority_symbols = get_priority_symbols()
        priority_intervals = get_priority_intervals()
        
        # Limit the initial backfill to save time
        symbols_to_process = priority_symbols if full else priority_symbols[:4]
        intervals_to_process = priority_intervals if full else priority_intervals[:3]
        
        lookback_years = 5 if full else 3
        
        logging.info(f"Starting improved backfill with {len(symbols_to_process)} symbols and {len(intervals_to_process)} intervals")
        logging.info(f"Symbols: {', '.join(symbols_to_process)}")
        logging.info(f"Intervals: {', '.join(intervals_to_process)}")
        
        # Initialize progress data
        progress_data = get_backfill_progress()
        total_combinations = len(symbols_to_process) * len(intervals_to_process)
        completed = 0
        
        # Process each symbol and interval
        for symbol in symbols_to_process:
            for interval in intervals_to_process:
                try:
                    logging.info(f"Processing {symbol}/{interval} ({completed+1}/{total_combinations})")
                    
                    # Run the backfill for this combination
                    run_backfill(symbols=[symbol], intervals=[interval], lookback_years=lookback_years)
                    
                    # Update progress
                    completed += 1
                    
                    # Update progress data
                    progress_data['symbols_completed'] = completed
                    progress_data['total_symbols'] = total_combinations
                    progress_data['percent_complete'] = round(completed / total_combinations * 100, 2)
                    progress_data['last_update'] = datetime.now().isoformat()
                    save_progress()
                    
                    # Get fresh progress data from the database
                    update_progress_from_database()
                    
                except Exception as e:
                    logging.error(f"Error processing {symbol}/{interval}: {e}")
                    
                # Add a small delay between combinations to avoid rate limiting
                time.sleep(2)
                
        logging.info(f"Improved backfill completed successfully. Processed {completed}/{total_combinations} combinations.")
    
    except Exception as e:
        logging.error(f"Error in improved backfill process: {e}")

if __name__ == "__main__":
    # When run directly, start a one-time backfill
    full_backfill = "--full" in sys.argv
    _run_background_backfill(full=full_backfill)