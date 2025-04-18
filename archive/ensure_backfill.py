#!/usr/bin/env python3
"""
Utility script to ensure database backfill process is running.
This can be run directly or imported by other scripts.
"""

import os
import subprocess
import time
import logging
from datetime import datetime
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_backfill_running(force=False):
    """
    Check if backfill process is running, and start it if not (or if force is True).
    
    Args:
        force: If True, kill any existing process and start a new one
    
    Returns:
        True if backfill was started, False if already running and not forced
    """
    lock_file = ".backfill_lock"
    
    # Check if lock file exists (indicating a backfill is already running)
    if os.path.exists(lock_file) and not force:
        logging.info("Backfill process appears to be already running (lock file exists)")
        return False
    
    # If forcing, remove existing lock file
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
            logging.info("Removed existing backfill lock file")
        except Exception as e:
            logging.error(f"Error removing lock file: {e}")
    
    # Kill any existing backfill processes
    try:
        subprocess.run(["pkill", "-f", "backfill_database.py"], 
                       stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        time.sleep(1)  # Give processes time to clean up
    except Exception as e:
        logging.error(f"Error killing existing processes: {e}")
    
    # Start new backfill process
    try:
        logging.info("Starting new background backfill process...")
        # Use continuous mode for regular updates
        subprocess.Popen(["python", "backfill_database.py", "--background", "--continuous", "--interval=15"], 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE)
        logging.info("Backfill process started successfully")
        return True
    except Exception as e:
        logging.error(f"Error starting backfill process: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensure cryptocurrency database backfill is running")
    parser.add_argument("--force", action="store_true", help="Force restart of backfill process even if already running")
    
    args = parser.parse_args()
    
    if ensure_backfill_running(force=args.force):
        print("Backfill process started successfully")
    else:
        print("Backfill process already running (use --force to restart)")