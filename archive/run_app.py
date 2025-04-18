#!/usr/bin/env python3
"""
Main entry point for the Cryptocurrency Trading Platform.
This script:
1. Resets the database (drops all tables)
2. Creates fresh tables
3. Starts the backfill process to download cryptocurrency data
4. Launches the Streamlit application
"""

import os
import sys
import time
import logging
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app_startup.log')
    ]
)

def start_backfill_process():
    """Run the reset_and_start.py script"""
    try:
        logging.info("Running database reset and backfill process...")
        
        # Run the reset_and_start.py script
        result = subprocess.run(
            ["python", "reset_and_start.py"],
            capture_output=True,
            text=True
        )
        
        # Log the output
        logging.info(f"Reset and start script stdout: {result.stdout}")
        if result.stderr:
            logging.warning(f"Reset and start script stderr: {result.stderr}")
        
        if result.returncode != 0:
            logging.error(f"Reset and start script failed with return code: {result.returncode}")
            return False
        
        logging.info("Reset and start script completed successfully")
        return True
    except Exception as e:
        logging.error(f"Error running reset and start script: {e}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print(f"CRYPTOCURRENCY TRADING PLATFORM STARTUP - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Run the reset and start process
    success = start_backfill_process()
    
    if success:
        print("\nApplication startup complete")
    else:
        print("\nApplication startup failed")
        sys.exit(1)