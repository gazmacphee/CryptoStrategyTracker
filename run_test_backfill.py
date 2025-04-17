"""
Script to run a test backfill with a limited set of symbols and intervals
"""

from download_binance_data import run_backfill
from clean_reset_database import reset_database

# Reset the database
print("Resetting database...")
reset_database()

# Run backfill for a subset of symbols and intervals
# Just testing with BTC and ETH, with 1h and 1d intervals for 1 year
symbols = ["BTCUSDT", "ETHUSDT"]
intervals = ["1h", "1d"]

print(f"Running test backfill for {', '.join(symbols)} with intervals {', '.join(intervals)}...")
run_backfill(symbols=symbols, intervals=intervals, lookback_years=1)

print("Test backfill completed!")