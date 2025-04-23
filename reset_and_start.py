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
    # First try using python-dotenv if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("Loaded environment variables from .env file using python-dotenv")
    except ImportError:
        print("dotenv package not found. Trying manual .env file loading.")
        
        # Manual loading as fallback
        import pathlib
        env_path = pathlib.Path('.env')
        if env_path.exists():
            print(f"Found .env file at {env_path.absolute()}")
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value and value[0] == value[-1] and value[0] in ["'", "\""]:
                        value = value[1:-1]
                    
                    os.environ[key] = value
            print("Loaded environment variables from .env file manually")
        else:
            print("No .env file found. Using system environment variables only.")
    
    # Try to construct DATABASE_URL from individual parameters if it's not set
    if "DATABASE_URL" not in os.environ and all(k in os.environ for k in ['PGHOST', 'PGPORT', 'PGUSER', 'PGPASSWORD', 'PGDATABASE']):
        host = os.environ.get('PGHOST')
        port = os.environ.get('PGPORT')
        user = os.environ.get('PGUSER')
        password = os.environ.get('PGPASSWORD')
        database = os.environ.get('PGDATABASE')
        
        database_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        os.environ['DATABASE_URL'] = database_url
        print(f"Constructed DATABASE_URL from individual parameters: postgresql://{user}:******@{host}:{port}/{database}")
        
except Exception as e:
    print(f"Failed to load environment variables: {e}")
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
        "sentiment_data",
        "news_data",
        "ml_predictions",
        "ml_model_performance",
        "detected_patterns",
        "trading_signals"
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
    """Create fresh database tables using the ensure_tables module"""
    logger.info("Creating fresh database tables...")
    
    try:
        # Use the comprehensive table creation script
        import ensure_tables
        all_tables_created = ensure_tables.ensure_all_tables()
        
        if all_tables_created:
            logger.info("Fresh tables created successfully - all tables verified")
        else:
            logger.warning("Some tables may be missing. Check table_creation.log for details.")
            
            # Fallback to traditional table creation
            logger.info("Attempting traditional table creation as fallback...")
            from database import create_tables as create_db_tables
            create_db_tables()
            
            # Create economic indicator tables
            try:
                from economic_indicators import create_economic_indicator_tables
                create_economic_indicator_tables()
                logger.info("Economic indicator tables created using fallback method")
            except Exception as e:
                logger.warning(f"Warning: Could not create economic indicator tables: {e}")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        sys.exit(1)


def remove_lock_files():
    """Remove all lock files to ensure a clean start"""
    lock_files = ['.backfill_lock', 'backfill_progress.json.lock']
    
    for lock_file in lock_files:
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                logger.info(f"Removed lock file at startup: {lock_file}")
                print(f"✓ Removed existing lock file: {lock_file}")
            except Exception as e:
                logger.error(f"Failed to remove lock file {lock_file}: {e}")
                print(f"⚠️ Warning: Failed to remove lock file {lock_file}: {e}")

def start_backfill():
    """Start a continuous full backfill process directly"""
    logger.info("Starting backfill process...")
    
    # Clear any existing lock files
    remove_lock_files()
    
    # Run the original backfill script directly with full and continuous options
    try:
        # Run in background with continuous mode, full backfill, and 15-minute interval
        subprocess.Popen(["python", "backfill_database.py", "--full", "--continuous", "--interval", "15"],
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


def initialize_economic_data():
    """Initialize economic indicators data if possible"""
    logger.info("Initializing economic indicators data...")
    
    try:
        from economic_indicators import update_economic_indicators
        update_result = update_economic_indicators()
        if update_result:
            logger.info("Economic indicators data initialized successfully")
            print("✅ Economic indicators initialized")
        else:
            logger.warning("Could not fully initialize economic indicators data")
            print("⚠️ Economic indicators initialization incomplete (some data may be missing)")
    except Exception as e:
        logger.warning(f"Could not initialize economic indicators: {e}")
        print("⚠️ Economic indicators initialization skipped")

def populate_additional_tables():
    """Populate news, sentiment, ML, and other secondary tables"""
    logger.info("Populating additional data tables...")
    
    try:
        # Apply ML fixes to prevent recursion errors
        try:
            # First, check if we have the ML fix module
            if os.path.exists("direct_ml_fix.py"):
                logger.info("ML fix module found, applying fixes...")
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                import direct_ml_fix
                if direct_ml_fix.fix_ml_modules():
                    logger.info("Successfully applied ML module recursion fixes")
                else:
                    logger.warning("Failed to apply ML module fixes - ML operations may fail")
            else:
                logger.warning("ML fix module not found - ML functions may experience recursion errors")
        except Exception as ml_fix_err:
            logger.error(f"Error applying ML fixes: {ml_fix_err}")
            
        # Start the data population script in the background
        subprocess.Popen(["python", "populate_all_tables.py"],
                       stdout=open("population.log", "w"),
                       stderr=subprocess.STDOUT)
        logger.info("Additional data tables population started in background")
        return True
    except Exception as e:
        logger.error(f"Error starting additional tables population: {e}")
        return False

def main():
    """Main function to run the application setup and startup"""
    print("=" * 80)
    logger.info("Starting application setup...")
    
    # Remove any existing lock files before doing anything else
    print("Cleaning up lock files...")
    remove_lock_files()
    print("✅ Lock files removed")
    
    print("[1/5] Resetting database...")
    reset_database()
    print("✅ Database reset complete")
    
    print("[2/5] Creating fresh tables...")
    create_tables()
    print("✅ Fresh tables created")
    
    print("[3/5] Initializing economic data...")
    initialize_economic_data()
    
    print("[4/5] Starting backfill process in background...")
    start_backfill()
    print("✅ Backfill process started in background")
    
    print("[5/5] Populating ML, news and sentiment data in background...")
    populate_additional_tables()
    print("✅ Additional data population started in background")
    
    print("=" * 80)
    print("Database setup complete. Starting Streamlit application...")
    print("The backfill process will continue running in the background.")
    print("=" * 80)
    
    # Start Streamlit
    start_streamlit()


if __name__ == "__main__":
    main()