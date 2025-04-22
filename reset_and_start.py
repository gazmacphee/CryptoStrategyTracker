"""
Reset and Start Script (Simplified)

This script serves as the main entry point for the application.
It performs the following tasks:
1. Resets the database
2. Creates fresh tables
3. Starts the original Streamlit application without dependency injection
"""

import os
import sys
import logging
import subprocess
import time
import psycopg2
from datetime import datetime

# Try to load environment variables from .env file for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("dotenv package not found. Using system environment variables only.")
except Exception as e:
    print(f"Failed to load environment variables from .env file: {e}")
    print("Continuing with system environment variables only.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("reset_and_start.log")
    ]
)
logger = logging.getLogger(__name__)


def get_db_connection():
    """Create a database connection with retry capability"""
    max_retries = 3
    retry_delay = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Check if DATABASE_URL is set
            if "DATABASE_URL" in os.environ:
                logger.info(f"Connecting to database using DATABASE_URL: {os.environ['DATABASE_URL'].split('@')[1]}")
                conn = psycopg2.connect(os.environ["DATABASE_URL"])
                conn.autocommit = True
                logger.info("Successfully connected using DATABASE_URL")
                return conn
            else:
                logger.error("DATABASE_URL environment variable is not set")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            retry_count += 1
            if retry_count < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Max retries reached. Could not connect to database.")
                sys.exit(1)


def reset_database():
    """Reset the database by dropping and recreating tables"""
    logger.info("Resetting database...")
    
    conn = get_db_connection()
    if conn is None:
        logger.error("Failed to get database connection")
        return
        
    cursor = conn.cursor()
    
    # Drop all tables
    tables = [
        "historical_data",
        "technical_indicators",
        "trades",
        "portfolio",
        "benchmarks",
        "sentiment_data"
    ]
    
    for table in tables:
        try:
            logger.info(f"Dropping table: {table}")
            cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        except Exception as e:
            logger.error(f"Error dropping table {table}: {e}")
    
    if conn is not None:
        conn.commit()
        cursor.close()
    logger.info("All tables dropped successfully")


def create_tables():
    """Create fresh database tables using the original database.py module"""
    logger.info("Creating fresh database tables...")
    
    try:
        # Use the original database module
        from database import create_tables as create_db_tables
        create_db_tables()
        logger.info("Fresh tables created successfully")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        sys.exit(1)


def start_backfill():
    """Start a simple backfill process directly"""
    logger.info("Starting backfill process...")
    
    # Clear any existing lock file
    if os.path.exists(".backfill_lock"):
        os.remove(".backfill_lock")
        logger.info("Removed existing backfill lock file")
    
    # Run the original backfill script directly
    try:
        # Run in background
        subprocess.Popen(["python", "backfill_database.py"],
                        stdout=open("backfill.log", "w"),
                        stderr=subprocess.STDOUT)
        logger.info("Backfill process started successfully")
    except Exception as e:
        logger.error(f"Error starting backfill: {e}")


def start_streamlit():
    """Start the Streamlit application using the original app.py"""
    logger.info("Starting Streamlit application...")
    
    try:
        # Use the original app.py instead of src/app.py
        subprocess.run(["streamlit", "run", "app.py", "--server.port", "5000"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error starting Streamlit application: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Streamlit application stopped by user")
        sys.exit(0)


def main():
    """Main function to run the application setup and startup"""
    print("=" * 80)
    logger.info("Starting application setup...")
    
    print("[1/3] Resetting database...")
    reset_database()
    print("✅ Database reset complete")
    
    print("[2/3] Creating fresh tables...")
    create_tables()
    print("✅ Fresh tables created")
    
    print("[3/3] Starting backfill process in background...")
    start_backfill()
    print("✅ Backfill process started in background")
    
    print("=" * 80)
    print("Database setup complete. Starting Streamlit application...")
    print("The backfill process will continue running in the background.")
    print("=" * 80)
    
    # Start Streamlit
    start_streamlit()


if __name__ == "__main__":
    main()