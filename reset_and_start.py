#!/usr/bin/env python3
"""
Script to reset the database, create fresh tables, and start the backfill process.
This script will:
1. Drop all existing tables from the database
2. Create fresh tables
3. Start the backfill process for all symbols and intervals
4. Verify the data is being populated
"""

import os
import sys
import time
import logging
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('reset_and_start.log')
    ]
)

def reset_database():
    """Reset the database by dropping and recreating all tables"""
    try:
        logging.info("Resetting database - dropping all tables...")
        from database import get_db_connection
        
        # Get database connection
        conn = get_db_connection()
        if conn is None:
            logging.error("Failed to get database connection")
            return False
            
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        # Drop each table
        for table in tables:
            logging.info(f"Dropping table: {table}")
            cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        
        # Commit changes
        conn.commit()
        cursor.close()
        conn.close()
        
        logging.info("All tables dropped successfully")
        return True
    except Exception as e:
        logging.error(f"Error resetting database: {e}")
        return False

def create_fresh_tables():
    """Create fresh database tables"""
    try:
        logging.info("Creating fresh database tables...")
        from database import create_tables
        
        # Create tables
        create_tables()
        
        logging.info("Fresh tables created successfully")
        return True
    except Exception as e:
        logging.error(f"Error creating fresh tables: {e}")
        return False

def start_backfill():
    """Start the backfill process for all symbols and intervals"""
    try:
        logging.info("Starting backfill process...")
        
        # Remove any existing lock file
        if os.path.exists(".backfill_lock"):
            os.remove(".backfill_lock")
            logging.info("Removed existing backfill lock file")
        
        # Import the improved backfill module
        from start_improved_backfill import start_background_backfill
        
        # Start the backfill process in background
        backfill_thread = start_background_backfill(
            symbols=None,  # Use default symbols
            intervals=None,  # Use default intervals
            continuous=False  # Run once, not continuously
        )
        
        logging.info("Backfill process started successfully")
        return backfill_thread
    except Exception as e:
        logging.error(f"Error starting backfill: {e}")
        return None

def check_database_population():
    """Check if the database is being populated"""
    try:
        from database import get_db_connection
        
        # Function to count records in a table
        def count_records(table_name):
            conn = get_db_connection()
            if conn is None:
                logging.error(f"Failed to get database connection when counting {table_name}")
                return 0
                
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            result = cursor.fetchone()
            count = result[0] if result else 0
            cursor.close()
            conn.close()
            return count
        
        # Check historical_data table
        initial_count = count_records("historical_data")
        logging.info(f"Initial historical data count: {initial_count}")
        
        # Wait for 30 seconds and check again
        time.sleep(30)
        
        # Check again to see if data is being populated
        new_count = count_records("historical_data")
        logging.info(f"New historical data count after 30 seconds: {new_count}")
        
        # Check other tables
        indicators_count = count_records("indicators")
        logging.info(f"Indicators count: {indicators_count}")
        
        # Check if data is being populated
        if new_count > initial_count:
            logging.info("Database is being populated successfully")
            return True
        else:
            logging.warning("Database population may not be working correctly")
            return False
    except Exception as e:
        logging.error(f"Error checking database population: {e}")
        return False

def main():
    """Main function to reset database and start backfill"""
    print("=" * 80)
    print(f"DATABASE RESET AND BACKFILL PROCESS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Step 1: Reset database
    print("\n[1/3] Resetting database...")
    if not reset_database():
        print("❌ Failed to reset database")
        return False
    print("✅ Database reset complete")
    
    # Step 2: Create fresh tables
    print("\n[2/3] Creating fresh tables...")
    if not create_fresh_tables():
        print("❌ Failed to create fresh tables")
        return False
    print("✅ Fresh tables created")
    
    # Step 3: Start backfill process in background
    print("\n[3/3] Starting backfill process in background...")
    backfill_thread = start_backfill()
    if not backfill_thread:
        print("❌ Failed to start backfill process")
        return False
    print("✅ Backfill process started in background")
    
    print("\n" + "=" * 80)
    print("Database setup complete. Starting Streamlit application...")
    print("The backfill process will continue running in the background.")
    print("=" * 80)
    
    return True

def start_streamlit():
    """Start the Streamlit application and wait for it to be ready"""
    try:
        print("\nStarting Streamlit application...")
        streamlit_process = subprocess.Popen(
            ["streamlit", "run", "app.py", "--server.port=5000", "--server.address=0.0.0.0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Wait for Streamlit to start
        start_time = time.time()
        timeout = 60  # Wait up to 60 seconds for Streamlit to start
        
        while time.time() - start_time < timeout:
            output = streamlit_process.stdout.readline()
            if output:
                print(output.strip())
                if "You can now view your Streamlit app in your browser" in output:
                    print("✅ Streamlit application started successfully")
                    break
            
            # Check if process is still running
            if streamlit_process.poll() is not None:
                print("❌ Streamlit process exited unexpectedly")
                return False
                
            time.sleep(0.1)
        
        # Keep the main process running
        while True:
            time.sleep(10)
            
        return True
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        # Start the Streamlit application and keep running
        start_streamlit()
    else:
        print("\nFailed to complete the database setup process")
        sys.exit(1)