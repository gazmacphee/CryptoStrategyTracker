#!/usr/bin/env python3
"""
Cleanup script to remove duplicate historical data entries
"""
import psycopg2
import os
from datetime import datetime, timedelta

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

def cleanup_historical_data():
    """
    Clean up historical data by removing duplicate entries and normalizing timestamps
    """
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database")
        return
    
    try:
        cursor = conn.cursor()
        
        # First, get all the symbol/interval combinations
        cursor.execute("SELECT DISTINCT symbol, interval FROM historical_data")
        symbol_intervals = cursor.fetchall()
        
        total_removed = 0
        
        for symbol, interval in symbol_intervals:
            print(f"Processing {symbol} on {interval} timeframe...")
            
            # Get all entries for this symbol/interval
            cursor.execute(
                "SELECT id, timestamp FROM historical_data WHERE symbol = %s AND interval = %s ORDER BY timestamp",
                (symbol, interval)
            )
            entries = cursor.fetchall()
            
            if not entries:
                continue
            
            prev_normalized_ts = None
            to_keep = []
            to_remove = []
            
            for entry_id, timestamp in entries:
                # Normalize the timestamp based on interval
                norm_ts = normalize_timestamp(interval, timestamp)
                
                # If this is a new normalized timestamp, keep it
                if prev_normalized_ts is None or norm_ts != prev_normalized_ts:
                    to_keep.append(entry_id)
                    prev_normalized_ts = norm_ts
                else:
                    # This is a duplicate, mark for removal
                    to_remove.append(entry_id)
            
            # Remove the duplicates
            if to_remove:
                placeholders = ','.join(['%s'] * len(to_remove))
                cursor.execute(
                    f"DELETE FROM historical_data WHERE id IN ({placeholders})",
                    to_remove
                )
                
                removed_count = len(to_remove)
                total_removed += removed_count
                print(f"Removed {removed_count} duplicate entries for {symbol} on {interval}")
            else:
                print(f"No duplicates found for {symbol} on {interval}")
            
            # Normalize the timestamps for all remaining entries
            cursor.execute(
                "SELECT id, timestamp FROM historical_data WHERE symbol = %s AND interval = %s",
                (symbol, interval)
            )
            remaining_entries = cursor.fetchall()
            
            for entry_id, timestamp in remaining_entries:
                norm_ts = normalize_timestamp(interval, timestamp)
                cursor.execute(
                    "UPDATE historical_data SET timestamp = %s WHERE id = %s",
                    (norm_ts, entry_id)
                )
            
            # Commit after each symbol/interval to avoid long transactions
            conn.commit()
        
        print(f"Total entries removed: {total_removed}")
        
        # Also update the indicators table to match normalized timestamps
        print("Updating technical indicators table timestamps...")
        cursor.execute("""
        UPDATE technical_indicators ti
        SET timestamp = hd.timestamp
        FROM historical_data hd
        WHERE ti.symbol = hd.symbol 
        AND ti.interval = hd.interval 
        AND DATE_TRUNC('minute', ti.timestamp) = DATE_TRUNC('minute', hd.timestamp)
        """)
        
        conn.commit()
        print("Cleanup completed successfully")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    print("Starting database cleanup process...")
    cleanup_historical_data()