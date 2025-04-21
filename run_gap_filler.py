#!/usr/bin/env python3
"""
Script to run the gap filler process after the regular backfill has completed.
This ensures that any gaps in data are identified and filled.
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime, timedelta

# Import local modules
from gap_filler import run_gap_analysis, check_technical_indicators_completeness
from data_loader import create_backfill_lock, release_backfill_lock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gap_filler.log')
    ]
)

def check_backfill_complete():
    """
    Check if the backfill process is complete by monitoring the lock file.
    
    Returns:
        True if backfill is complete, False otherwise
    """
    # If lock file doesn't exist, backfill is complete
    if not os.path.exists(".backfill_lock"):
        return True
        
    # If lock file exists but is old (>30 minutes), consider backfill stalled and proceed
    try:
        lock_age = time.time() - os.path.getmtime(".backfill_lock")
        if lock_age > 1800:  # 30 minutes in seconds
            logging.warning("Backfill lock file is stale (>30 minutes old). Proceeding with gap filling.")
            return True
    except Exception as e:
        logging.error(f"Error checking lock file age: {e}")
        
    return False

def run_gap_filler(wait_for_backfill=True, lookback_days=90, max_gaps=10):
    """
    Run the gap filler process.
    
    Args:
        wait_for_backfill: Whether to wait for backfill to complete before starting
        lookback_days: Number of days to look back for gaps
        max_gaps: Maximum number of gaps to fill per symbol/interval
    """
    if wait_for_backfill:
        logging.info("Waiting for backfill process to complete...")
        
        # Wait up to 30 minutes for backfill to complete
        start_time = time.time()
        while time.time() - start_time < 1800:  # 30 minutes
            if check_backfill_complete():
                logging.info("Backfill process is complete. Starting gap analysis.")
                break
                
            # Check every 30 seconds
            time.sleep(30)
            
        if not check_backfill_complete():
            logging.warning("Backfill did not complete within timeout. Starting gap analysis anyway.")
    
    # Create our own lock file to prevent other processes from interfering
    create_backfill_lock()
    
    try:
        logging.info("=" * 80)
        logging.info(f"STARTING GAP FILLER PROCESS - {datetime.now()}")
        logging.info(f"Lookback period: {lookback_days} days, Max gaps per pair: {max_gaps}")
        logging.info("=" * 80)
        
        # First run the indicator completeness check
        logging.info("Checking technical indicators completeness...")
        indicator_results = check_technical_indicators_completeness(lookback_days=lookback_days)
        
        logging.info(f"Found {indicator_results['total_missing']} missing indicators")
        
        # Then run the gap analysis
        logging.info("Running data gap analysis...")
        gap_results = run_gap_analysis(lookback_days=lookback_days, max_gaps=max_gaps)
        
        logging.info(f"Found {gap_results['total_gaps']} data gaps")
        logging.info(f"Successfully filled {gap_results['filled_gaps']} gaps")
        logging.info(f"Failed to fill {gap_results['failed_gaps']} gaps")
        
        # Final indicator check to ensure all indicators are calculated
        logging.info("Final check of technical indicators...")
        final_results = check_technical_indicators_completeness(lookback_days=lookback_days)
        
        logging.info(f"Remaining missing indicators: {final_results['total_missing']}")
        
        logging.info("=" * 80)
        logging.info(f"GAP FILLER PROCESS COMPLETE - {datetime.now()}")
        logging.info("=" * 80)
        
        return {
            'indicator_results': indicator_results,
            'gap_results': gap_results,
            'final_results': final_results
        }
        
    except Exception as e:
        logging.error(f"Error in gap filler process: {e}")
        return None
    finally:
        # Always release our lock when done
        release_backfill_lock()

def start_gap_filler_thread(wait_for_backfill=True, lookback_days=90, max_gaps=10):
    """Start the gap filler process in a background thread"""
    thread = threading.Thread(
        target=run_gap_filler, 
        args=(wait_for_backfill, lookback_days, max_gaps)
    )
    thread.daemon = True
    thread.start()
    return thread

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run gap filler process for cryptocurrency data")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for backfill to complete")
    parser.add_argument("--lookback", type=int, default=90, help="Lookback days (default: 90)")
    parser.add_argument("--max-gaps", type=int, default=10, help="Maximum gaps to fill per pair (default: 10)")
    
    args = parser.parse_args()
    
    print(f"=== Cryptocurrency Data Gap Filler ===")
    print(f"Waiting for backfill: {not args.no_wait}")
    print(f"Lookback period: {args.lookback} days")
    print(f"Maximum gaps to fill: {args.max_gaps}")
    print("")
    
    results = run_gap_filler(
        wait_for_backfill=not args.no_wait,
        lookback_days=args.lookback,
        max_gaps=args.max_gaps
    )
    
    if results:
        print("\nResults Summary:")
        print(f"Missing indicators initially: {results['indicator_results']['total_missing']}")
        print(f"Data gaps found: {results['gap_results']['total_gaps']}")
        print(f"Data gaps filled: {results['gap_results']['filled_gaps']}")
        print(f"Remaining missing indicators: {results['final_results']['total_missing']}")