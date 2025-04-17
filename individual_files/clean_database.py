#!/usr/bin/env python3
"""
One-time complete database cleanup to eliminate all duplicates and normalize timestamps
"""
import psycopg2
import os
from datetime import datetime, timedelta
import time

def get_db_connection():
    """Create a database connection"""
    try:
        # Try connecting using DATABASE_URL directly
        DATABASE_URL = os.environ.get("DATABASE_URL")
        if DATABASE_URL:
            conn = psycopg2.connect(DATABASE_URL)
            return conn
        else:
            # Use individual parameters if no DATABASE_URL
            DB_HOST = os.environ.get("PGHOST", "localhost")
            DB_PORT = os.environ.get("PGPORT", "5432")
            DB_NAME = os.environ.get("PGDATABASE", "crypto")
            DB_USER = os.environ.get("PGUSER", "postgres")
            DB_PASS = os.environ.get("PGPASSWORD", "postgres")
            
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASS
            )
            return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def normalize_timestamp(interval, timestamp):
    """
    Normalize a timestamp to the correct interval boundary
    """
    if interval in ['1h', '2h', '4h', '6h', '8h', '12h']:
        # For hour-based intervals, normalize to exact hours
        normalized = timestamp.replace(minute=0, second=0, microsecond=0)
    elif interval in ['1d', '3d', '1w']:
        # For day-based intervals, normalize to midnight
        normalized = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    elif interval == '1M':
        # For month interval, normalize to first day of month
        normalized = timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        # For minute-based intervals, ensure timestamps are at exact minute boundaries
        normalized = timestamp.replace(second=0, microsecond=0)
    
    return normalized

def clean_database():
    """
    Execute a complete database cleanup by rebuilding the tables
    """
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database")
        return
    
    try:
        cursor = conn.cursor()
        
        # Create a temporary table to store cleaned data
        cursor.execute("""
        CREATE TEMP TABLE temp_historical_data (
            symbol VARCHAR(20) NOT NULL,
            interval VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            close NUMERIC NOT NULL,
            volume NUMERIC NOT NULL,
            PRIMARY KEY(symbol, interval, timestamp)
        )
        """)
        
        # Get all distinct symbol/interval combinations
        cursor.execute("SELECT DISTINCT symbol, interval FROM historical_data")
        symbol_intervals = cursor.fetchall()
        total_cleaned = 0
        
        for symbol, interval in symbol_intervals:
            print(f"Cleaning {symbol} on {interval} timeframe...")
            
            # Get all raw data for this symbol/interval
            cursor.execute("""
            SELECT timestamp, open, high, low, close, volume
            FROM historical_data
            WHERE symbol = %s AND interval = %s
            ORDER BY timestamp
            """, (symbol, interval))
            
            entries = cursor.fetchall()
            
            # Track normalized timestamps to avoid duplicates
            normalized_entries = {}
            
            for timestamp, open_price, high, low, close, volume in entries:
                # Normalize the timestamp
                norm_ts = normalize_timestamp(interval, timestamp)
                
                # If this timestamp already exists, take the latest one
                normalized_entries[norm_ts] = (open_price, high, low, close, volume)
            
            # Insert cleaned data into temp table
            for norm_ts, values in normalized_entries.items():
                open_price, high, low, close, volume = values
                
                cursor.execute("""
                INSERT INTO temp_historical_data (symbol, interval, timestamp, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, interval, timestamp) DO NOTHING
                """, (symbol, interval, norm_ts, open_price, high, low, close, volume))
                
            # Count how many entries we have now
            cursor.execute("""
            SELECT COUNT(*) FROM temp_historical_data WHERE symbol = %s AND interval = %s
            """, (symbol, interval))
            
            new_count = cursor.fetchone()[0]
            cursor.execute("""
            SELECT COUNT(*) FROM historical_data WHERE symbol = %s AND interval = %s
            """, (symbol, interval))
            
            old_count = cursor.fetchone()[0]
            print(f"  Reduced {old_count} entries to {new_count} (removed {old_count - new_count} duplicates)")
            total_cleaned += (old_count - new_count)
        
        # Now replace the original table data with cleaned data
        print("\nReplacing historical data with cleaned data...")
        cursor.execute("DELETE FROM historical_data")
        
        cursor.execute("""
        INSERT INTO historical_data (symbol, interval, timestamp, open, high, low, close, volume)
        SELECT symbol, interval, timestamp, open, high, low, close, volume FROM temp_historical_data
        """)
        
        # Also clean up the technical indicators table based on the cleaned historical data
        print("\nCleaning technical indicators table...")
        cursor.execute("""
        DELETE FROM technical_indicators ti
        WHERE NOT EXISTS (
            SELECT 1 FROM historical_data hd
            WHERE ti.symbol = hd.symbol
            AND ti.interval = hd.interval
            AND ti.timestamp = hd.timestamp
        )
        """)
        
        conn.commit()
        print(f"\nDatabase cleanup completed. Removed {total_cleaned} duplicate entries.")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    print("Starting complete database cleanup...")
    start_time = time.time()
    clean_database()
    end_time = time.time()
    print(f"Cleanup completed in {end_time - start_time:.2f} seconds")