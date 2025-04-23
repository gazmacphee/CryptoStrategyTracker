#!/usr/bin/env python3
"""
Script to train all pattern recognition models once sufficient data is available.
This script can be run manually or scheduled to run daily.
"""

import os
import time
import logging
import psycopg2
import argparse
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pattern_model_training.log')
    ]
)
logger = logging.getLogger()

def check_symbols_with_sufficient_data(min_records=1000):
    """
    Check which symbol/interval combinations have sufficient data for training.
    
    Args:
        min_records: Minimum number of records required for training
    
    Returns:
        List of (symbol, interval) tuples that have sufficient data
    """
    try:
        # Get database URL from environment
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            logger.error("DATABASE_URL environment variable not set")
            return []
            
        # Connect to database
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Execute query to check for symbols with sufficient data
        query = """
        SELECT symbol, interval, COUNT(*) as record_count
        FROM historical_data
        GROUP BY symbol, interval
        HAVING COUNT(*) >= %s
        ORDER BY symbol, interval
        """
        cursor.execute(query, (min_records,))
        
        # Get results
        results = cursor.fetchall()
        
        # Close connection
        cursor.close()
        conn.close()
        
        # Convert to list of tuples
        symbol_intervals = [(row[0], row[1]) for row in results]
        
        if symbol_intervals:
            logger.info(f"Found {len(symbol_intervals)} symbol/interval combinations with sufficient data")
            for symbol, interval in symbol_intervals:
                logger.info(f"  - {symbol}/{interval}")
        else:
            logger.warning(f"No symbol/interval combinations have sufficient data (min {min_records} records)")
            
        return symbol_intervals
            
    except Exception as e:
        logger.error(f"Error checking for symbols with sufficient data: {e}")
        return []

def train_pattern_model(symbol, interval, days=90):
    """
    Train a pattern model for a specific symbol and interval.
    
    Args:
        symbol: Trading pair symbol
        interval: Time interval
        days: Number of days of data to use for training
        
    Returns:
        Boolean indicating if training was successful
    """
    try:
        # Build the command
        cmd = [
            "python", "run_ml_fixed.py",
            "--train",
            "--symbol", symbol,
            "--interval", interval,
            "--days", str(days),
            "--verbose"
        ]
        
        # Run the command
        logger.info(f"Training pattern model for {symbol}/{interval}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        
        # Log output
        for line in proc.stdout.splitlines():
            logger.info(f"  {line}")
            
        if proc.returncode == 0:
            logger.info(f"Successfully trained pattern model for {symbol}/{interval}")
            return True
        else:
            logger.error(f"Failed to train pattern model for {symbol}/{interval}")
            for line in proc.stderr.splitlines():
                logger.error(f"  {line}")
            return False
            
    except Exception as e:
        logger.error(f"Error training pattern model for {symbol}/{interval}: {e}")
        return False

def train_all_available_models(min_records=1000, days=90, max_symbols=None):
    """
    Train models for all symbol/interval combinations with sufficient data.
    
    Args:
        min_records: Minimum number of records required for training
        days: Number of days of data to use for training
        max_symbols: Maximum number of symbols to train (None for all)
        
    Returns:
        Number of successfully trained models
    """
    # Check which symbols have sufficient data
    symbol_intervals = check_symbols_with_sufficient_data(min_records)
    
    # Limit number of symbols if requested
    if max_symbols is not None and max_symbols > 0:
        symbol_intervals = symbol_intervals[:max_symbols]
        
    if not symbol_intervals:
        logger.warning("No symbols with sufficient data available for training")
        return 0
        
    # Train models for each symbol/interval
    successful = 0
    start_time = datetime.now()
    
    logger.info(f"Starting pattern model training for {len(symbol_intervals)} symbol/intervals at {start_time}")
    
    for i, (symbol, interval) in enumerate(symbol_intervals):
        logger.info(f"Training model {i+1}/{len(symbol_intervals)}: {symbol}/{interval}")
        success = train_pattern_model(symbol, interval, days)
        if success:
            successful += 1
            
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60.0
    
    logger.info(f"Pattern model training completed: {successful}/{len(symbol_intervals)} models trained")
    logger.info(f"Total training time: {duration:.2f} minutes")
    
    return successful

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pattern recognition models")
    parser.add_argument("--min-records", type=int, default=1000, help="Minimum records required for training")
    parser.add_argument("--days", type=int, default=90, help="Days of data to use for training")
    parser.add_argument("--max-symbols", type=int, default=None, help="Maximum number of symbols to train")
    parser.add_argument("--wait", action="store_true", help="Wait for data if none is available")
    parser.add_argument("--wait-interval", type=int, default=5, help="Minutes to wait between checks")
    parser.add_argument("--max-wait", type=int, default=60, help="Maximum minutes to wait for data")
    args = parser.parse_args()
    
    logger.info("Starting pattern model training script")
    
    if args.wait:
        # Wait for data to be available
        wait_attempts = 0
        max_attempts = args.max_wait // args.wait_interval
        
        while wait_attempts < max_attempts:
            # Check if any symbols have sufficient data
            symbols = check_symbols_with_sufficient_data(args.min_records)
            if symbols:
                break
                
            # Wait before checking again
            wait_attempts += 1
            logger.info(f"Waiting for data... (attempt {wait_attempts}/{max_attempts})")
            time.sleep(args.wait_interval * 60)
    
    # Train all available models
    successful = train_all_available_models(args.min_records, args.days, args.max_symbols)
    
    logger.info(f"Pattern model training script completed with {successful} models trained")