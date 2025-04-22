"""
Database Connection Test Script

This script tests the database connection using environment variables
from the .env file. It's useful for diagnosing connection issues
before starting the full application.
"""

import os
import sys
import psycopg2
import pathlib

def manual_load_env():
    """Manually load environment variables from .env file"""
    print("Loading environment variables from .env file...")
    env_path = pathlib.Path('.env')
    
    if not env_path.exists():
        print("ERROR: .env file not found in current directory.")
        return False
    
    print(f"Found .env file at {env_path.absolute()}")
    
    # Manually read and set environment variables
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
            if key in ['PGHOST', 'PGPORT', 'PGUSER', 'PGDATABASE', 'DATABASE_URL']:
                if key == 'DATABASE_URL':
                    masked_value = value.split('@')[1] if '@' in value else value
                    print(f"Set {key}={masked_value}")
                elif key == 'PGPASSWORD':
                    print(f"Set {key}=******")
                else:
                    print(f"Set {key}={value}")
    
    return True

def construct_database_url():
    """Construct DATABASE_URL from individual parameters if needed"""
    if "DATABASE_URL" not in os.environ and all(k in os.environ for k in ['PGHOST', 'PGPORT', 'PGUSER', 'PGPASSWORD', 'PGDATABASE']):
        host = os.environ.get('PGHOST')
        port = os.environ.get('PGPORT')
        user = os.environ.get('PGUSER')
        password = os.environ.get('PGPASSWORD')
        database = os.environ.get('PGDATABASE')
        
        database_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        os.environ['DATABASE_URL'] = database_url
        print(f"Constructed DATABASE_URL from individual parameters: postgresql://{user}:******@{host}:{port}/{database}")
        return True
    return False

def test_database_connection():
    """Test the database connection using environment variables"""
    print("\nTesting database connection...")
    
    if "DATABASE_URL" not in os.environ:
        print("ERROR: DATABASE_URL environment variable is not set")
        if all(k in os.environ for k in ['PGHOST', 'PGPORT', 'PGUSER', 'PGPASSWORD', 'PGDATABASE']):
            print("Individual PostgreSQL parameters are set but DATABASE_URL is missing.")
            print("Attempting to construct DATABASE_URL from individual parameters...")
            construct_database_url()
        else:
            print("Required PostgreSQL parameters are missing in the environment.")
            print("Needed parameters: PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE")
            missing = [k for k in ['PGHOST', 'PGPORT', 'PGUSER', 'PGPASSWORD', 'PGDATABASE'] if k not in os.environ]
            print(f"Missing parameters: {', '.join(missing)}")
            return False
    
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print("ERROR: DATABASE_URL is empty")
        return False
    
    print(f"Using DATABASE_URL: {database_url.split('@')[1] if '@' in database_url else database_url}")
    
    try:
        print("Connecting to database...")
        conn = psycopg2.connect(database_url)
        conn.autocommit = True
        
        # Test the connection by executing a simple query
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"Successfully connected to database!")
        print(f"PostgreSQL version: {version}")
        
        # Close the connection
        cursor.close()
        conn.close()
        print("Connection test completed successfully!")
        return True
    except Exception as e:
        print(f"ERROR: Failed to connect to database: {e}")
        
        # Provide more specific error analysis
        if "could not connect to server" in str(e):
            print("\nPossible causes:")
            print("1. The PostgreSQL server is not running")
            print("2. Firewall is blocking the connection")
            print("3. The hostname or IP address is incorrect")
            print(f"4. The port ({os.environ.get('PGPORT', 'unknown')}) is incorrect or blocked")
        elif "password authentication failed" in str(e):
            print("\nAuthentication failed. Check your username and password.")
        elif "database" in str(e) and "does not exist" in str(e):
            print(f"\nThe database '{os.environ.get('PGDATABASE', 'unknown')}' does not exist.")
            print("You need to create the database first:")
            print(f"createdb {os.environ.get('PGDATABASE', 'crypto')}")
        
        return False

def remove_lock_files():
    """Remove any existing lock files to ensure clean startup"""
    lock_files = ['.backfill_lock', 'backfill_progress.json.lock']
    
    print("\nChecking for and removing any lock files...")
    for lock_file in lock_files:
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                print(f"✅ Removed existing lock file: {lock_file}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to remove lock file {lock_file}: {e}")
        else:
            print(f"✓ No {lock_file} file found")

def main():
    print("=" * 80)
    print("Database Connection Test Tool")
    print("=" * 80)
    
    # Remove any lock files to ensure a clean start
    remove_lock_files()
    
    # Load environment variables
    if not manual_load_env():
        print("Failed to load environment variables. Please check your .env file.")
        sys.exit(1)
    
    # Test database connection
    if test_database_connection():
        print("\nSUCCESS: Database connection test passed!")
        print("You can now run the application using:")
        print("  python start_local.py")
        sys.exit(0)
    else:
        print("\nFAILED: Database connection test failed.")
        print("Please fix the database connection issues before running the application.")
        sys.exit(1)

if __name__ == "__main__":
    main()