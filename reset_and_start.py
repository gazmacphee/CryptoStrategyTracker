"""
Reset and Start Script

This script serves as the main entry point for the application.
It performs the following tasks:
1. Resets the database if needed
2. Creates fresh tables
3. Starts a background process to download and process historical data
4. Launches the Streamlit application
"""

import os
import sys
import logging
import subprocess
import time
import signal
import psycopg2
from datetime import datetime
import threading

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
    
    conn.commit()
    cursor.close()
    logger.info("All tables dropped successfully")


def create_tables():
    """Create fresh database tables"""
    logger.info("Creating fresh database tables...")
    
    # Initialize container first
    from src.config.initialize import initialize_container
    initialize_container()
    
    # Then use the database module
    from src.data.database import create_tables as create_db_tables
    create_db_tables()
    
    logger.info("Fresh tables created successfully")


def start_backfill():
    """Start the backfill process in the background"""
    logger.info("Starting backfill process...")
    
    # Clear any existing lock file
    if os.path.exists(".backfill_lock"):
        os.remove(".backfill_lock")
        logger.info("Removed existing backfill lock file")
    
    # Test Binance API availability
    try:
        import requests
        response = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
        if response.status_code == 200:
            logger.info("Binance API is available. The application will use real-time data.")
            
            # Check if API keys are set
            if "BINANCE_API_KEY" in os.environ and "BINANCE_API_SECRET" in os.environ:
                logger.info("Binance API keys found. Using authenticated API access.")
            else:
                logger.info("Binance API keys not found. Using public API access.")
        else:
            logger.warning("Binance API is not available. The application will use backfilled historical data.")
    except Exception as e:
        logger.warning(f"Error testing Binance API: {e}")
        logger.warning("Binance API may not be available. The application will use backfilled historical data.")
    
    # Import and initialize the container correctly
    from src.config.initialize import initialize_container
    
    # Get the container with all services initialized
    container = initialize_container()
    
    # Get backfill service from the container
    backfill_service = container.get("backfill_service")
    
    # Start backfill
    result = backfill_service.start_backfill()
    
    if result['status'] == 'started':
        logger.info("Backfill process started successfully")
    else:
        logger.error(f"Error starting backfill: {result}")


def start_streamlit():
    """Start the Streamlit application"""
    logger.info("Starting Streamlit application...")
    
    try:
        # Use the new src/app.py module
        subprocess.run(["streamlit", "run", "src/app.py", "--server.port", "5000"], check=True)
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
    
    # Process command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Reset and start the application')
    parser.add_argument('--no-reset', action='store_true', help='Skip database reset')
    args = parser.parse_args()
    
    if not args.no_reset:
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