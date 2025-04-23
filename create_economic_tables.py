"""
Special script to ensure economic indicator tables are created correctly in Windows.
This script avoids unicode characters that cause problems in Windows terminals.
"""

import os
import logging
import database
import sys

# Configure logging without unicode characters
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_economic_tables_without_unicode():
    """Create economic indicator tables while avoiding unicode characters in output"""
    try:
        logger.info("Ensuring economic indicator tables exist...")
        conn = database.get_db_connection()
        
        with conn.cursor() as cursor:
            # Table for US Dollar Index (DXY)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS economic_indicators (
                id SERIAL PRIMARY KEY,
                indicator_name VARCHAR(50) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                value DECIMAL(22,8) NOT NULL,
                source VARCHAR(50),
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(indicator_name, timestamp)
            );
            """)
            
            # Table for US Dollar Index (DXY)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS dollar_index (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                close DECIMAL(18,8) NOT NULL,
                open DECIMAL(18,8),
                high DECIMAL(18,8),
                low DECIMAL(18,8),
                volume DECIMAL(18,8),
                UNIQUE(timestamp)
            );
            """)
            
            # Table for global liquidity indicators
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS global_liquidity (
                id SERIAL PRIMARY KEY,
                indicator_name VARCHAR(50) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                value DECIMAL(22,8) NOT NULL,
                UNIQUE(indicator_name, timestamp)
            );
            """)
        
        conn.commit()
        conn.close()
        
        logger.info("Economic indicator tables created successfully")
        print("SUCCESS: Economic indicator tables created.")
        return True
    except Exception as e:
        logger.error(f"Error creating economic indicator tables: {e}")
        print(f"ERROR: Failed to create economic indicator tables - {str(e)}")
        return False

if __name__ == "__main__":
    # Set environment variable for UTF-8 encoding to avoid Windows issues
    if 'win' in sys.platform.lower():
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    success = create_economic_tables_without_unicode()
    sys.exit(0 if success else 1)