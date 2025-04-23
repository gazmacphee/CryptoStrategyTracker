"""
Database extensions module that adds missing functions needed for ML processing.
This module imports from database.py and adds additional functionality.
"""

import database
from database import get_db_connection
import pandas as pd

def get_symbols_from_database(limit=None):
    """
    Get a list of all unique symbols in the database
    
    Args:
        limit: Optional limit on the number of symbols to return
    
    Returns:
        List of symbol strings
    """
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM historical_data")
            symbols = [row[0] for row in cursor.fetchall()]
            
            # Sort symbols and apply limit if provided
            symbols.sort()
            if limit and limit < len(symbols):
                symbols = symbols[:limit]
                
            return symbols
        except Exception as e:
            print(f"Error getting symbols from database: {e}")
            return []
        finally:
            conn.close()
    return []


def get_available_symbols(quote_asset="USDT", limit=30):
    """
    Get available trading pairs, prioritizing ones in our database
    Similar to the function in binance_api.py but directly from database

    Args:
        quote_asset: Quote currency (e.g., "USDT")
        limit: Maximum number of symbols to return

    Returns:
        List of symbol strings
    """
    # Default popular symbols to return if DB access fails
    popular_symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT",
        "SOLUSDT", "DOTUSDT", "LTCUSDT", "LINKUSDT", "MATICUSDT", "AVAXUSDT",
        "ATOMUSDT", "UNIUSDT", "FILUSDT", "AAVEUSDT", "NEARUSDT", "ALGOUSDT",
        "ICPUSDT", "XTZUSDT", "AXSUSDT", "FTMUSDT", "SANDUSDT", "MANAUSDT",
        "HBARUSDT", "THETAUSDT", "EGLDUSDT", "FLOWUSDT", "APUSDT", "CAKEUSDT"
    ]
    
    # First try to get symbols from our database
    db_symbols = get_symbols_from_database()
    
    if db_symbols:
        # Filter for the specified quote asset
        filtered_symbols = [s for s in db_symbols if s.endswith(quote_asset)]
        
        # Prioritize popular symbols that we have data for
        available_popular = [s for s in popular_symbols if s in filtered_symbols]
        remaining = [s for s in filtered_symbols if s not in popular_symbols]
        
        # Combine and limit
        result = available_popular + remaining
        if limit and limit < len(result):
            return result[:limit]
        return result
    
    # Fallback to default popular symbols
    return popular_symbols[:limit]

# Add the functions to the database module namespace
database.get_symbols_from_database = get_symbols_from_database
database.get_available_symbols = get_available_symbols