#!/usr/bin/env python3
"""
Script to fix the pattern strength issue in advanced_ml.py.
This script updates the code that handles pattern detection and strength calculation.
"""

import sys
import logging
import os
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

def fix_analyze_patterns_method():
    """Fix the code that handles pattern analysis to properly use strength column"""
    try:
        # Check if run_ml_fixed.py exists
        if not os.path.exists('run_ml_fixed.py'):
            logger.error("run_ml_fixed.py file not found")
            return False
            
        # First, read the file
        with open('run_ml_fixed.py', 'r') as f:
            content = f.read()
            
        # Find the section that's handling the strength error
        pattern_strength_error = re.search(r"(df_patterns\['strength'\]|df\['strength'\])", content)
        
        if not pattern_strength_error:
            logger.warning("Could not find pattern strength reference in run_ml_fixed.py")
            # Let's try with advanced_ml.py
            if not os.path.exists('advanced_ml.py'):
                logger.error("advanced_ml.py file not found")
                return False
                
            with open('advanced_ml.py', 'r') as f:
                content = f.read()
                
            pattern_strength_error = re.search(r"(df_patterns\['strength'\]|df\['strength'\])", content)
            if not pattern_strength_error:
                logger.warning("Could not find pattern strength reference in advanced_ml.py either")
                return False
        
        # Now check for the analyze_patterns_and_save.py file
        if os.path.exists('analyze_patterns_and_save.py'):
            logger.info("Checking analyze_patterns_and_save.py for strength reference")
            
            with open('analyze_patterns_and_save.py', 'r') as f:
                analyze_content = f.read()
                
            if "'strength'" in analyze_content:
                # If there are column checks, update them to include pattern_strength
                updated_analyze = analyze_content.replace(
                    "if 'strength' not in", 
                    "if 'strength' not in df.columns and 'pattern_strength' not in df.columns:"
                )
                
                # Add code to handle pattern_strength as a fallback
                updated_analyze = updated_analyze.replace(
                    "results['pattern_count'] = pattern_count", 
                    "results['pattern_count'] = pattern_count\n" +
                    "            # Handle different column names\n" +
                    "            if 'pattern_strength' in df.columns and 'strength' not in df.columns:\n" +
                    "                df['strength'] = df['pattern_strength']"
                )
                
                with open('analyze_patterns_and_save.py', 'w') as f:
                    f.write(updated_analyze)
                    
                logger.info("Updated analyze_patterns_and_save.py to handle both strength and pattern_strength columns")
                
        # Create a fix to handle both column names
        # First, let's create a helper script that fixes database patterns
        with open('db_pattern_strength_fix.py', 'w') as f:
            f.write("""#!/usr/bin/env python3
\"\"\"
Helper script to fix pattern strength issues between database and code.
\"\"\"

import os
import sys
import logging
import psycopg2
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

def get_database_url():
    \"\"\"Get database URL from environment variables\"\"\"
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)
    
    logger.info(f"Connecting to database using DATABASE_URL: {db_url.split('@')[-1].split('/')[0]}")
    return db_url

def read_patterns_from_db():
    \"\"\"Read patterns from database and handle column name differences\"\"\"
    try:
        conn = psycopg2.connect(get_database_url())
        
        # Read patterns from database
        query = \"\"\"
        SELECT 
            id, symbol, interval, timestamp, detection_timestamp, 
            pattern_type, 
            COALESCE(strength, pattern_strength) as strength, 
            COALESCE(pattern_strength, strength) as pattern_strength,
            features, description, status
        FROM detected_patterns
        ORDER BY detection_timestamp DESC
        LIMIT 100;
        \"\"\"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        logger.info(f"Read {len(df)} patterns from database")
        
        # Make sure both column names exist
        if 'strength' not in df.columns:
            df['strength'] = df['pattern_strength']
        
        if 'pattern_strength' not in df.columns:
            df['pattern_strength'] = df['strength']
        
        return df
        
    except Exception as e:
        logger.error(f"Error reading patterns from database: {e}")
        return pd.DataFrame()

def fix_pattern_columns():
    \"\"\"Fix database pattern columns to ensure both strength and pattern_strength exist\"\"\"
    try:
        conn = psycopg2.connect(get_database_url())
        cursor = conn.cursor()
        
        # Check if strength column exists
        cursor.execute(\"\"\"
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'detected_patterns' AND column_name = 'strength'
            );
        \"\"\")
        strength_exists = cursor.fetchone()[0]
        
        # Check if pattern_strength column exists
        cursor.execute(\"\"\"
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'detected_patterns' AND column_name = 'pattern_strength'
            );
        \"\"\")
        pattern_strength_exists = cursor.fetchone()[0]
        
        # If one exists and the other doesn't, add the missing one
        if strength_exists and not pattern_strength_exists:
            logger.info("Adding pattern_strength column based on strength")
            cursor.execute(\"\"\"
                ALTER TABLE detected_patterns 
                ADD COLUMN pattern_strength FLOAT;
                
                UPDATE detected_patterns 
                SET pattern_strength = strength;
            \"\"\")
            conn.commit()
            
        elif pattern_strength_exists and not strength_exists:
            logger.info("Adding strength column based on pattern_strength")
            cursor.execute(\"\"\"
                ALTER TABLE detected_patterns 
                ADD COLUMN strength FLOAT;
                
                UPDATE detected_patterns 
                SET strength = pattern_strength;
            \"\"\")
            conn.commit()
            
        # If both exist, make sure they're in sync
        elif strength_exists and pattern_strength_exists:
            logger.info("Syncing strength and pattern_strength columns")
            cursor.execute(\"\"\"
                UPDATE detected_patterns 
                SET strength = pattern_strength 
                WHERE strength IS NULL;
                
                UPDATE detected_patterns 
                SET pattern_strength = strength 
                WHERE pattern_strength IS NULL;
            \"\"\")
            conn.commit()
        
        cursor.close()
        conn.close()
        
        logger.info("Successfully fixed pattern strength columns in database")
        return True
        
    except Exception as e:
        logger.error(f"Error fixing pattern columns: {e}")
        return False

if __name__ == "__main__":
    # Fix the database columns
    fix_pattern_columns()
    
    # Read patterns to verify
    patterns = read_patterns_from_db()
    if not patterns.empty:
        logger.info("Successfully read patterns with fixed columns")
        logger.info(f"Sample pattern: {patterns.iloc[0].to_dict()}")
""")
        
        logger.info("Created db_pattern_strength_fix.py to fix database column issues")
        
        # Now create a wrapper for run_ml_fixed.py to handle both column names
        with open('run_ml_wrapper.py', 'w') as f:
            f.write("""#!/usr/bin/env python3
\"\"\"
Wrapper for run_ml_fixed.py that fixes strength column issues.
\"\"\"

import sys
import logging
import os
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

def fix_db_and_run():
    \"\"\"Fix database pattern strength issues and then run ML script\"\"\"
    try:
        # First run the database fix
        logger.info("Running database pattern strength fix")
        result = subprocess.run(['python', 'db_pattern_strength_fix.py'], check=True)
        
        if result.returncode != 0:
            logger.error("Database fix failed")
            sys.exit(1)
            
        # Now run the original ML script with all arguments
        cmd = ['python', 'run_ml_fixed.py'] + sys.argv[1:]
        logger.info(f"Running ML script with args: {' '.join(cmd)}")
        
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
        
    except Exception as e:
        logger.error(f"Error running ML wrapper: {e}")
        sys.exit(1)

if __name__ == "__main__":
    fix_db_and_run()
""")
        
        logger.info("Created run_ml_wrapper.py to handle both column names")
        return True
        
    except Exception as e:
        logger.error(f"Error fixing analyze patterns method: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting pattern strength issue fix")
    
    # Fix the analyze patterns method
    if fix_analyze_patterns_method():
        logger.info("Successfully created fixes for pattern strength issue")
        logger.info("\nTo use the fixed version, run: python run_ml_wrapper.py --analyze --days 1065")
    else:
        logger.error("Failed to fix pattern strength issue")
        sys.exit(1)