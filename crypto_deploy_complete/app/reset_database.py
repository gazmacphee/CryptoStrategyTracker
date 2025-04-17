#!/usr/bin/env python3
"""
Script to reset (clear) the database for fresh deployment
"""
import psycopg2
import os
from database import get_db_connection, create_tables

def reset_database():
    """
    Reset the database by dropping all tables and recreating them
    """
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database")
        return
    
    try:
        cursor = conn.cursor()
        
        # Disable foreign key checks temporarily if database supports it
        try:
            cursor.execute("SET session_replication_role = 'replica';")
        except:
            pass
        
        # Get all tables in the database
        cursor.execute("""
        SELECT tablename FROM pg_tables 
        WHERE schemaname = 'public'
        """)
        
        tables = cursor.fetchall()
        
        # Drop each table
        for table in tables:
            table_name = table[0]
            print(f"Dropping table: {table_name}")
            cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
        
        # Re-enable foreign key checks
        try:
            cursor.execute("SET session_replication_role = 'origin';")
        except:
            pass
        
        conn.commit()
        print("All tables dropped successfully")
        
        # Recreate tables
        create_tables()
        print("Database reset completed successfully")
        
    except Exception as e:
        print(f"Error resetting database: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    print("Starting database reset...")
    reset_database()
    print("Database reset complete")