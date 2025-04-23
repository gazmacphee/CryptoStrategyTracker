#!/usr/bin/env python3
"""
Script to fix the detected_patterns table in the database.
This script adds the missing detection_timestamp column and fixes the strength column issue.
"""

import os
import sys
import logging
import psycopg2
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

def get_database_url():
    """Get database URL from environment variables"""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)
    
    logger.info(f"Connecting to database using DATABASE_URL: {db_url.split('@')[-1].split('/')[0]}")
    return db_url

def check_table_schema():
    """Check if the detected_patterns table exists and what columns it has"""
    try:
        # Connect to the database
        conn = psycopg2.connect(get_database_url())
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'detected_patterns'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            logger.warning("The detected_patterns table does not exist")
            cursor.close()
            conn.close()
            return False, []
        
        # Get column information
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'detected_patterns';
        """)
        columns = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        
        logger.info(f"Table detected_patterns exists with columns: {', '.join(columns)}")
        return True, columns
        
    except Exception as e:
        logger.error(f"Error checking table schema: {e}")
        return False, []

def fix_detected_patterns_table():
    """Fix the detected_patterns table by adding missing columns"""
    try:
        # First check the current schema
        table_exists, columns = check_table_schema()
        
        if not table_exists:
            # Create the entire table
            logger.info("Creating detected_patterns table from scratch")
            conn = psycopg2.connect(get_database_url())
            cursor = conn.cursor()
            
            # Create the table with all required columns
            cursor.execute("""
                CREATE TABLE detected_patterns (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    interval VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    detection_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                    pattern_type VARCHAR(50) NOT NULL,
                    strength FLOAT NOT NULL,
                    price_at_detection FLOAT NOT NULL,
                    target_price FLOAT,
                    stop_loss FLOAT,
                    features JSON,
                    description TEXT,
                    status VARCHAR(20) DEFAULT 'active'
                );
                
                CREATE INDEX detected_patterns_symbol_interval_idx ON detected_patterns (symbol, interval);
                CREATE INDEX detected_patterns_timestamp_idx ON detected_patterns (timestamp);
                CREATE INDEX detected_patterns_detection_timestamp_idx ON detected_patterns (detection_timestamp);
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("Successfully created detected_patterns table with all required columns")
            return True
            
        # Check for specific missing columns
        missing_columns = []
        
        if 'detection_timestamp' not in columns:
            missing_columns.append("detection_timestamp TIMESTAMP NOT NULL DEFAULT NOW()")
        
        if 'strength' not in columns:
            missing_columns.append("strength FLOAT NOT NULL DEFAULT 0.5")
        
        if 'target_price' not in columns:
            missing_columns.append("target_price FLOAT")
        
        if 'stop_loss' not in columns:
            missing_columns.append("stop_loss FLOAT")
        
        if 'description' not in columns:
            missing_columns.append("description TEXT")
        
        if 'status' not in columns:
            missing_columns.append("status VARCHAR(20) DEFAULT 'active'")
        
        if 'features' not in columns:
            missing_columns.append("features JSON")
        
        # If we have missing columns, add them
        if missing_columns:
            conn = psycopg2.connect(get_database_url())
            cursor = conn.cursor()
            
            for column_def in missing_columns:
                column_name = column_def.split()[0]
                try:
                    logger.info(f"Adding column {column_name} to detected_patterns table")
                    cursor.execute(f"ALTER TABLE detected_patterns ADD COLUMN {column_def};")
                    conn.commit()
                except Exception as e:
                    logger.error(f"Error adding column {column_name}: {e}")
                    conn.rollback()
            
            # Create indexes for performance
            try:
                cursor.execute("CREATE INDEX IF NOT EXISTS detected_patterns_symbol_interval_idx ON detected_patterns (symbol, interval);")
                cursor.execute("CREATE INDEX IF NOT EXISTS detected_patterns_timestamp_idx ON detected_patterns (timestamp);")
                
                if 'detection_timestamp' in missing_columns:
                    cursor.execute("CREATE INDEX IF NOT EXISTS detected_patterns_detection_timestamp_idx ON detected_patterns (detection_timestamp);")
                
                conn.commit()
                logger.info("Created necessary indexes on detected_patterns table")
            except Exception as e:
                logger.warning(f"Error creating indexes: {e}")
                conn.rollback()
            
            cursor.close()
            conn.close()
            
            logger.info(f"Added {len(missing_columns)} missing columns to detected_patterns table")
            return True
        else:
            logger.info("No missing columns to add to detected_patterns table")
            return True
            
    except Exception as e:
        logger.error(f"Error fixing detected_patterns table: {e}")
        return False

def fix_strength_column():
    """Fix the strength column if it's causing issues"""
    try:
        # First check if the table and column exist
        table_exists, columns = check_table_schema()
        
        if not table_exists or 'strength' not in columns:
            logger.warning("Cannot fix strength column - table or column doesn't exist")
            return False
        
        # Connect to the database
        conn = psycopg2.connect(get_database_url())
        cursor = conn.cursor()
        
        # Get current column type
        cursor.execute("""
            SELECT data_type 
            FROM information_schema.columns 
            WHERE table_name = 'detected_patterns' AND column_name = 'strength';
        """)
        data_type = cursor.fetchone()[0]
        
        if data_type.lower() != 'double precision' and data_type.lower() != 'real':
            # We need to fix the column type
            logger.info(f"Changing strength column type from {data_type} to FLOAT")
            
            # Create a backup of the existing data
            try:
                cursor.execute("CREATE TABLE detected_patterns_backup AS SELECT * FROM detected_patterns;")
                conn.commit()
                logger.info("Created backup of detected_patterns table")
            except Exception as e:
                logger.error(f"Error creating backup: {e}")
                conn.rollback()
                cursor.close()
                conn.close()
                return False
            
            # Alter the column type
            try:
                cursor.execute("ALTER TABLE detected_patterns ALTER COLUMN strength TYPE FLOAT USING strength::float;")
                conn.commit()
                logger.info("Successfully changed strength column type to FLOAT")
            except Exception as e:
                logger.error(f"Error changing strength column type: {e}")
                conn.rollback()
                
                # Try to recover from backup if the migration failed
                try:
                    cursor.execute("DROP TABLE detected_patterns;")
                    cursor.execute("ALTER TABLE detected_patterns_backup RENAME TO detected_patterns;")
                    conn.commit()
                    logger.info("Restored from backup after failed migration")
                except Exception as e2:
                    logger.error(f"Error restoring from backup: {e2}")
                
                cursor.close()
                conn.close()
                return False
        else:
            logger.info(f"Strength column is already the correct type ({data_type})")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error fixing strength column: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting detected_patterns table fix")
    
    # Fix the table structure
    table_fixed = fix_detected_patterns_table()
    if table_fixed:
        logger.info("Successfully fixed detected_patterns table structure")
    else:
        logger.error("Failed to fix detected_patterns table structure")
        sys.exit(1)
    
    # Fix the strength column
    strength_fixed = fix_strength_column()
    if strength_fixed:
        logger.info("Successfully fixed strength column")
    else:
        logger.warning("Could not fix strength column - may need manual intervention")
    
    logger.info("Database fix completed")