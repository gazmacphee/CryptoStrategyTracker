"""
Local Development Starter Script

This script ensures environment variables are properly loaded from the .env file
before starting the application on a local machine.
"""

import os
import sys
import subprocess
import pathlib

def remove_lock_files():
    """Remove any existing lock files to ensure clean startup"""
    lock_files = ['.backfill_lock', 'backfill_progress.json.lock']
    
    for lock_file in lock_files:
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                print(f"✅ Removed existing lock file: {lock_file}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to remove lock file {lock_file}: {e}")

def main():
    # Load environment variables from .env file
    print("Loading environment variables from .env file...")
    
    try:
        # Using direct file parsing as a fallback method
        env_path = pathlib.Path('.env')
        if env_path.exists():
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
                    print(f"Set environment variable: {key}")
            
            # Check for DATABASE_URL specifically
            if "DATABASE_URL" in os.environ:
                print(f"Database URL found: {os.environ['DATABASE_URL'].split('@')[1] if '@' in os.environ['DATABASE_URL'] else os.environ['DATABASE_URL']}")
            else:
                # Try to construct DATABASE_URL from individual parameters
                host = os.environ.get('PGHOST', 'localhost')
                port = os.environ.get('PGPORT', '5432')
                user = os.environ.get('PGUSER', 'postgres')
                password = os.environ.get('PGPASSWORD', '')
                database = os.environ.get('PGDATABASE', 'crypto')
                
                # Construct DATABASE_URL
                database_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
                os.environ['DATABASE_URL'] = database_url
                print(f"Constructed DATABASE_URL from individual parameters: postgresql://{user}:******@{host}:{port}/{database}")
        else:
            print("No .env file found in the current directory.")
            
            # Check if we already have the variables set in the environment
            if "DATABASE_URL" in os.environ:
                print(f"Using DATABASE_URL from environment: {os.environ['DATABASE_URL'].split('@')[1] if '@' in os.environ['DATABASE_URL'] else os.environ['DATABASE_URL']}")
            else:
                print("DATABASE_URL not found in environment.")
                print("Please create a .env file with your database configuration.")
                sys.exit(1)
    except Exception as e:
        print(f"Error loading environment variables: {e}")
        print("Continuing with any environment variables that might be already set...")
    
    # Final check for DATABASE_URL
    if "DATABASE_URL" not in os.environ:
        print("ERROR: DATABASE_URL is not set in your .env file or environment.")
        print("Please make sure you have a valid DATABASE_URL in your .env file.")
        print("Example format: DATABASE_URL=postgresql://username:password@localhost:5432/dbname")
        sys.exit(1)
    
    # Remove any existing lock files before starting
    print("\nChecking for and removing any existing lock files...")
    remove_lock_files()
    
    # Start the application
    print("\nStarting the application...")
    print(f"Using DATABASE_URL: {os.environ['DATABASE_URL'].split('@')[1] if '@' in os.environ['DATABASE_URL'] else os.environ['DATABASE_URL']}")
    subprocess.run([sys.executable, "reset_and_start.py"])

if __name__ == "__main__":
    main()