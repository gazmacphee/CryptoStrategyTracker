#!/usr/bin/env python3
"""
Script to reset all backfill state and start a fresh backfill process.
"""

import os
import sys
import time
import logging
import json
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Lock file used by backfill process
LOCK_FILE = ".backfill_lock"
PROGRESS_FILE = "backfill_progress.json"

def main():
    # 1. Force kill any running backfill processes
    logging.info("Killing any running backfill processes...")
    try:
        subprocess.run(["pkill", "-f", "backfill_database.py"], 
                      stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        # Give it a moment to shutdown
        time.sleep(1)
    except Exception as e:
        logging.warning(f"Error killing processes: {e}")
    
    # 2. Remove lock file
    if os.path.exists(LOCK_FILE):
        logging.info(f"Removing lock file: {LOCK_FILE}")
        os.remove(LOCK_FILE)
    else:
        logging.info("No lock file found.")
    
    # 3. Reset progress file
    progress_data = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_symbols": 0,
        "symbols_completed": 0,
        "symbols_progress": {},
        "overall_progress": 0.0,
        "is_running": False,
        "errors": []
    }
    
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress_data, f, indent=2)
    
    logging.info(f"Reset progress file: {PROGRESS_FILE}")
    
    # 4. Start a new backfill process
    logging.info("Starting fresh backfill process...")
    try:
        # Run directly to see output
        os.system("python backfill_database.py")
    except Exception as e:
        logging.error(f"Error starting backfill: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)