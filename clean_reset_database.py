"""
Script to completely reset the database, removing all data and recreating tables
"""
import os
import psycopg2
from database import get_db_connection, create_tables

def reset_database():
    """Completely reset the database by dropping all tables and recreating them"""
    print("Connecting to database...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("Dropping all tables...")
    tables = [
        "historical_data",
        "technical_indicators",
        "trades",
        "portfolio",
        "benchmarks",
        "sentiment_data"
    ]
    
    for table in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        print(f"Dropped table: {table}")
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print("Recreating database tables...")
    create_tables()
    
    # Also remove any backfill lock file if it exists
    lock_file = ".backfill_lock"
    if os.path.exists(lock_file):
        os.remove(lock_file)
        print(f"Removed lock file: {lock_file}")
    
    print("Database reset completed successfully.")

if __name__ == "__main__":
    reset_database()