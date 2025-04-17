"""
Script to backfill the database with historical cryptocurrency data.
This script is called from the Dockerfile during container setup.
"""

import logging
import time
from datetime import datetime, timedelta

import download_binance_data as dbd
import data_loader
from database import create_tables
from binance_api import get_available_symbols

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_popular_symbols(limit=10):
    """Get a list of popular cryptocurrency symbols"""
    # Default list of popular symbols if API call fails
    default_symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT",
        "XRPUSDT", "DOTUSDT", "UNIUSDT", "LTCUSDT", "LINKUSDT"
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

def backfill_database(full=False):
    """
    Run a backfill operation on the database
    
    Args:
        full: Whether to do a full backfill (more symbols, longer history)
    """
    # Create tables if they don't exist
    create_tables()
    
    # Check if a backfill process is already running
    if data_loader.create_backfill_lock():
        try:
            # Set up popular symbols and intervals for backfill
            symbols = get_popular_symbols()
            intervals = ['1h', '4h', '1d']
            
            if full:
                # For full backfill, use more symbols and add weekly/monthly intervals
                symbols = get_popular_symbols(limit=20)
                intervals = ['1h', '4h', '8h', '12h', '1d', '3d', '1w', '1M']
            
            # Run backfill for all specified symbols and intervals
            dbd.run_backfill(symbols=symbols, intervals=intervals, lookback_years=3)
            
        finally:
            # Always release the lock when done
            data_loader.release_backfill_lock()
    else:
        logging.info("A backfill process is already running. Skipping.")

if __name__ == "__main__":
    backfill_database(full=False)