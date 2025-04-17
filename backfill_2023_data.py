"""
Script to backfill data from 2023 specifically, which we know exists in the Binance Data Vision repository
"""

import logging
from datetime import date
from download_binance_data import run_backfill

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def run_2023_data_backfill():
    """Run a backfill for 2023 data, which we know exists in the Binance Data Vision repository"""
    # Specify the symbols (popular cryptocurrencies)
    symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", 
        "ADAUSDT", "DOGEUSDT", "XRPUSDT"
    ]
    
    # Specify the intervals (timeframes)
    intervals = [
        "1h", "4h", "1d"
    ]
    
    # We'll only look back 1 year to focus on 2023 data
    lookback_years = 1
    
    # Log what we're doing
    print(f"Starting 2023 data backfill for {len(symbols)} symbols with {len(intervals)} intervals")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Intervals: {', '.join(intervals)}")
    
    # Run the backfill with explicit date range for 2023
    total_candles = run_backfill(
        symbols=symbols,
        intervals=intervals,
        lookback_years=lookback_years
    )
    
    print(f"Backfill completed. Total candles downloaded: {total_candles}")
    return total_candles

if __name__ == "__main__":
    # Reset the database first
    from clean_reset_database import reset_database
    
    print("Resetting database...")
    reset_database()
    
    # Run the 2023 data backfill
    run_2023_data_backfill()