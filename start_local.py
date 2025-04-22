"""
Local Development Starter Script

This script ensures environment variables are properly loaded from the .env file
before starting the application on a local machine.
"""

import os
import sys
import subprocess
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    print("Loading environment variables from .env file...")
    load_dotenv(verbose=True)
    
    # Verify that DATABASE_URL is set
    if "DATABASE_URL" not in os.environ:
        print("ERROR: DATABASE_URL is not set in your .env file.")
        print("Please make sure you have a valid DATABASE_URL in your .env file.")
        print("Example format: DATABASE_URL=postgresql://username:password@localhost:5432/dbname")
        sys.exit(1)
    
    print(f"Using database: {os.environ['DATABASE_URL'].split('@')[1] if '@' in os.environ['DATABASE_URL'] else os.environ['DATABASE_URL']}")
    
    # Start the application
    print("Starting the application...")
    subprocess.run([sys.executable, "reset_and_start.py"])

if __name__ == "__main__":
    main()