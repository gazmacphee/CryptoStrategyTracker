#!/usr/bin/env python3
"""
Tool to check if the database has been populated with historical data
and to run ML analysis when data is available.
"""

import sys
import logging
import time
import argparse
import psycopg2
import os
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_database_for_data(symbol="BTCUSDT", interval="30m", min_records=100):
    """
    Check if the database has enough historical data for ML analysis.
    
    Args:
        symbol: The symbol to check for
        interval: The interval to check for
        min_records: Minimum number of records required
        
    Returns:
        Boolean indicating if enough data is available
    """
    try:
        # Get database URL from environment
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            logging.error("DATABASE_URL environment variable not set")
            return False
            
        # Connect to database
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Execute query to check for data
        query = """
        SELECT COUNT(*) 
        FROM historical_data 
        WHERE symbol = %s AND interval = %s
        """
        cursor.execute(query, (symbol, interval))
        
        # Get result
        result = cursor.fetchone()
        count = result[0] if result else 0
        
        # Close connection
        cursor.close()
        conn.close()
        
        # Check if we have enough data
        if count >= min_records:
            logging.info(f"Found {count} records for {symbol}/{interval} in database")
            return True
        else:
            logging.warning(f"Only {count} records found for {symbol}/{interval} in database. Waiting for more...")
            return False
            
    except Exception as e:
        logging.error(f"Error checking database: {e}")
        return False

def run_ml_analysis(symbol="BTCUSDT", interval="30m"):
    """Run ML analysis with the specified symbol and interval"""
    try:
        # Build the command
        cmd = [
            "python", "run_ml_fixed.py",
            "--analyze",
            "--symbol", symbol,
            "--interval", interval,
            "--verbose"
        ]
        
        # Run the command
        logging.info(f"Running ML analysis for {symbol}/{interval}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        
        # Log output
        logging.info("ML analysis completed with output:")
        for line in proc.stdout.splitlines():
            logging.info(f"  {line}")
            
        return proc.returncode == 0
        
    except Exception as e:
        logging.error(f"Error running ML analysis: {e}")
        return False

def wait_for_data_and_run_ml(symbol="BTCUSDT", interval="30m", max_wait_minutes=10, check_interval_seconds=30):
    """
    Wait for data to be populated in the database and run ML analysis when ready.
    
    Args:
        symbol: The symbol to analyze
        interval: The interval to analyze
        max_wait_minutes: Maximum minutes to wait for data
        check_interval_seconds: How often to check for data (seconds)
    """
    logging.info(f"Waiting for {symbol}/{interval} data to be populated (max {max_wait_minutes} minutes)...")
    
    # Calculate max attempts
    max_attempts = int((max_wait_minutes * 60) / check_interval_seconds)
    
    # Wait and check loop
    for attempt in range(max_attempts):
        # Check if we have data
        if check_database_for_data(symbol, interval):
            # Run ML analysis
            success = run_ml_analysis(symbol, interval)
            if success:
                logging.info(f"ML analysis completed successfully for {symbol}/{interval}")
            else:
                logging.error(f"ML analysis failed for {symbol}/{interval}")
            return success
            
        # Wait before next check
        logging.info(f"Waiting for data... (attempt {attempt+1}/{max_attempts})")
        time.sleep(check_interval_seconds)
    
    # If we get here, we've exceeded max wait time
    logging.error(f"Exceeded maximum wait time of {max_wait_minutes} minutes. No data available.")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for data and run ML analysis")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol to check")
    parser.add_argument("--interval", default="30m", help="Interval to check")
    parser.add_argument("--max-wait", type=int, default=10, help="Maximum minutes to wait")
    parser.add_argument("--check-interval", type=int, default=30, help="Seconds between checks")
    args = parser.parse_args()
    
    # Run the main function
    success = wait_for_data_and_run_ml(
        symbol=args.symbol,
        interval=args.interval,
        max_wait_minutes=args.max_wait,
        check_interval_seconds=args.check_interval
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)