#!/usr/bin/env python
"""
Database reset script for the cryptocurrency trading analysis platform.
This script drops and recreates all tables in the database.
"""

import os
import sys
import psycopg2
from database import create_tables

def reset_database():
    """Reset the database by dropping and recreating all tables"""
    try:
        # Connect to database
        conn = psycopg2.connect(
            host=os.environ.get('PGHOST'),
            port=os.environ.get('PGPORT'),
            user=os.environ.get('PGUSER'),
            password=os.environ.get('PGPASSWORD'),
            database=os.environ.get('PGDATABASE'),
            sslmode='require'
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Drop all tables
        print("Dropping all tables...")
        tables = [
            'historical_data',
            'indicators',
            'trades',
            'portfolio',
            'benchmarks',
            'sentiment_data'
        ]
        
        for table in tables:
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                print(f"Dropped table: {table}")
            except Exception as e:
                print(f"Error dropping table {table}: {e}")
        
        # Close connection
        cursor.close()
        conn.close()
        
        # Recreate tables
        print("Recreating tables...")
        create_tables()
        
        print("Database reset completed successfully.")
        return True
        
    except Exception as e:
        print(f"Error resetting database: {e}")
        return False

if __name__ == "__main__":
    if not os.environ.get('DATABASE_URL'):
        print("Error: DATABASE_URL environment variable not set.")
        sys.exit(1)
    
    reset_database()