"""
Ensure All Tables Script

This script ensures that all required database tables are created during startup.
It creates tables from all modules that define table structures.
"""

import os
import sys
import logging
import importlib
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("table_creation.log")
    ]
)
logger = logging.getLogger(__name__)

def create_core_tables():
    """Create the core database tables"""
    try:
        from database import create_tables, get_db_connection

        # Check database connection first
        conn = get_db_connection()
        if conn is None:
            logger.error("Failed to connect to database. Cannot create tables.")
            return False

        # Close the connection
        conn.close()

        # Create the tables
        create_tables()
        logger.info("Core database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating core tables: {e}")
        return False

def create_economic_tables():
    """Create economic indicator tables"""
    try:
        from economic_indicators import create_economic_indicator_tables

        create_economic_indicator_tables()
        logger.info("Economic indicator tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating economic indicator tables: {e}")
        return False

def create_ml_tables():
    """Create machine learning related tables"""
    try:
        # The ML tables should be created by the database.py module
        # This function is a placeholder for any additional ML tables
        logger.info("ML tables should already be created by database.py")
        return True
    except Exception as e:
        logger.error(f"Error creating ML tables: {e}")
        return False

def create_news_sentiment_tables():
    """Create news and sentiment related tables"""
    try:
        # These tables should be created by the database.py module
        # This function is a placeholder for any additional news/sentiment tables
        logger.info("News and sentiment tables should already be created by database.py")
        return True
    except Exception as e:
        logger.error(f"Error creating news and sentiment tables: {e}")
        return False

def verify_tables_exist():
    """Verify that all required tables exist in the database"""
    try:
        from database import get_db_connection

        conn = get_db_connection()
        if conn is None:
            logger.error("Failed to connect to database. Cannot verify tables.")
            return False

        cursor = conn.cursor()

        # Query to list all tables in the database
        cursor.execute("""
            SELECT tablename 
            FROM pg_catalog.pg_tables 
            WHERE schemaname='public'
        """)

        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()

        # List of expected tables
        expected_tables = [
            'historical_data',
            'technical_indicators',
            'trades',
            'portfolio',
            'benchmarks',
            'sentiment_data',
            'news_data',
            'ml_predictions',
            'ml_model_performance',
            'detected_patterns',
            'trading_signals',
            'economic_indicators',
            'global_liquidity'
        ]

        # Check if all expected tables exist
        missing_tables = [table for table in expected_tables if table not in tables]

        if missing_tables:
            logger.warning(f"Missing tables: {', '.join(missing_tables)}")
            return False
        else:
            logger.info(f"All {len(expected_tables)} expected tables exist in the database")
            return True
    except Exception as e:
        logger.error(f"Error verifying tables: {e}")
        return False

def ensure_all_tables():
    """Ensure all tables are created"""
    logger.info("Ensuring all database tables are created...")

    # Step 1: Create core tables
    core_tables_created = create_core_tables()
    if not core_tables_created:
        logger.warning("Failed to create core tables. Some functionality may be limited.")

    # Step 2: Create economic tables
    economic_tables_created = create_economic_tables()
    if not economic_tables_created:
        logger.warning("Failed to create economic tables. Economic indicators may not work.")

    # Step 3: Create ML tables (likely created by core tables)
    ml_tables_created = create_ml_tables()

    # Step 4: Create news and sentiment tables (likely created by core tables)
    news_sentiment_tables_created = create_news_sentiment_tables()

    # Step 5: Verify that all tables exist
    all_tables_exist = verify_tables_exist()

    if all_tables_exist:
        logger.info("✅ All required database tables exist and are ready to use")
        return True
    else:
        logger.warning("⚠️ Some required tables may be missing. Functionality may be limited.")
        return False

if __name__ == "__main__":
    success = ensure_all_tables()
    if success:
        print("All database tables successfully created and verified.")
        sys.exit(0)
    else:
        print("Warning: Not all database tables could be created or verified.")
        sys.exit(1)