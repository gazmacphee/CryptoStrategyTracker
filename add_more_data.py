"""
Script to add more data to the database from different months
"""

from download_single_pair import download_and_process
from database import create_tables

def add_more_data():
    """Download several months of data for key cryptocurrencies"""
    # Make sure tables exist
    create_tables()
    
    # Bitcoin data for multiple months
    for month in [10, 11]: # October, November 2023
        download_and_process("BTCUSDT", "1d", 2023, month)
    
    # Hourly data for Bitcoin
    for month in [11, 12]: # November, December 2023
        download_and_process("BTCUSDT", "4h", 2023, month)
        
    # Ethereum data
    for month in [11, 12]: # November, December 2023 
        download_and_process("ETHUSDT", "1d", 2023, month)
        
    # Binance Coin data
    for month in [11, 12]: # November, December 2023
        download_and_process("BNBUSDT", "1d", 2023, month)
    
    print("Added additional data to the database")

if __name__ == "__main__":
    add_more_data()
    
    # Check database counts
    import db_check