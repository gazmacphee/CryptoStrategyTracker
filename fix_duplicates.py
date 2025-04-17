#!/usr/bin/env python3
"""
Cleanup script to remove duplicate historical data entries by directly removing and recreating them
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

def fix_duplicates():
    """
    More aggressive approach to fix duplicates by keeping only one row per hour
    """
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database")
        return
    
    try:
        cursor = conn.cursor()
        
        # Target specific symbol/interval combinations with duplicates
        cursor.execute("""
        SELECT symbol, interval, date_trunc('hour', timestamp) as hour, COUNT(*) 
        FROM historical_data 
        GROUP BY symbol, interval, hour
        HAVING COUNT(*) > 1
        ORDER BY COUNT(*) DESC
        """)
        
        duplicates = cursor.fetchall()
        print(f"Found {len(duplicates)} hours with duplicate entries")
        
        for symbol, interval, hour, count in duplicates:
            print(f"Fixing {symbol} {interval} at {hour} - {count} duplicates")
            
            # Get the latest entry for this hour (assuming it's the most accurate)
            cursor.execute("""
            SELECT id, open, high, low, close, volume, timestamp
            FROM historical_data
            WHERE symbol = %s AND interval = %s AND date_trunc('hour', timestamp) = %s
            ORDER BY id DESC
            LIMIT 1
            """, (symbol, interval, hour))
            
            latest = cursor.fetchone()
            if latest:
                id, open_price, high, low, close, volume, timestamp = latest
                
                # Delete all entries for this hour
                cursor.execute("""
                DELETE FROM historical_data
                WHERE symbol = %s AND interval = %s AND date_trunc('hour', timestamp) = %s
                """, (symbol, interval, hour))
                
                # Normalize the timestamp to exact hour
                normalized_timestamp = hour
                
                # Reinsert the single entry with normalized timestamp
                cursor.execute("""
                INSERT INTO historical_data (symbol, interval, timestamp, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (symbol, interval, normalized_timestamp, open_price, high, low, close, volume))
                
                print(f"  Replaced with single entry at {normalized_timestamp}")
        
        conn.commit()
        print("Duplicate fix completed successfully")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    print("Starting duplicate fix process...")
    fix_duplicates()