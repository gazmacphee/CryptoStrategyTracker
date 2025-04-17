#!/usr/bin/env python3
"""
Quick cleanup script that uses SQL to directly remove duplicate records
"""
import psycopg2
import os
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

def quick_clean_duplicates():
    """
    Perform a quick cleanup of duplicates using SQL DELETE
    """
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database")
        return
    
    try:
        cursor = conn.cursor()
        
        # Create a temporary table to identify duplicates
        print("Creating temporary table to identify duplicates...")
        cursor.execute("""
        CREATE TEMP TABLE duplicate_ids AS
        SELECT id
        FROM (
            SELECT id,
                   ROW_NUMBER() OVER (PARTITION BY symbol, interval, date_trunc('hour', timestamp) 
                                     ORDER BY id DESC) as row_num
            FROM historical_data
        ) ranked
        WHERE row_num > 1;
        """)
        
        # Count duplicates
        cursor.execute("SELECT COUNT(*) FROM duplicate_ids")
        dup_count = cursor.fetchone()[0]
        print(f"Found {dup_count} duplicate records to remove")
        
        if dup_count > 0:
            # Delete duplicate records
            print("Deleting duplicate records...")
            cursor.execute("""
            DELETE FROM historical_data
            WHERE id IN (SELECT id FROM duplicate_ids)
            """)
            
            conn.commit()
            print(f"Successfully removed {dup_count} duplicate records")
        
        # Now normalize timestamps for remaining records
        print("Normalizing timestamps for all historical data...")
        
        # For hour-based intervals
        cursor.execute("""
        UPDATE historical_data
        SET timestamp = date_trunc('hour', timestamp)
        WHERE interval IN ('1h', '2h', '4h', '6h', '8h', '12h')
        """)
        
        # For day-based intervals
        cursor.execute("""
        UPDATE historical_data
        SET timestamp = date_trunc('day', timestamp)
        WHERE interval IN ('1d', '3d', '1w')
        """)
        
        # For month interval
        cursor.execute("""
        UPDATE historical_data
        SET timestamp = date_trunc('month', timestamp)
        WHERE interval = '1M'
        """)
        
        # For minute-based intervals
        cursor.execute("""
        UPDATE historical_data
        SET timestamp = date_trunc('minute', timestamp)
        WHERE interval IN ('3m', '5m', '15m', '30m')
        """)
        
        conn.commit()
        print("Timestamps normalized")
        
        # Remove 1m interval data as requested
        print("Removing 1m interval data as requested...")
        cursor.execute("DELETE FROM historical_data WHERE interval = '1m'")
        
        # Count deleted rows
        deleted_count = cursor.rowcount
        print(f"Removed {deleted_count} records with 1m interval")
        
        # Also remove associated indicator data
        cursor.execute("DELETE FROM technical_indicators WHERE interval = '1m'")
        indicators_deleted = cursor.rowcount
        print(f"Removed {indicators_deleted} indicator records with 1m interval")
        
        conn.commit()
        print("Cleanup completed successfully")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    print("Starting quick database cleanup...")
    start_time = time.time()
    quick_clean_duplicates()
    end_time = time.time()
    print(f"Cleanup completed in {end_time - start_time:.2f} seconds")