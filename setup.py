#!/usr/bin/env python3
"""
Setup script for Cryptocurrency Trading Analysis Platform.
This script helps configure the environment without using Docker.
"""

import os
import sys
import subprocess
import shutil
import platform


def check_python_version():
    """Check if Python version is 3.11 or higher."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 11):
        print("Error: Python 3.11 or higher is required.")
        sys.exit(1)
    print("✓ Python version is compatible.")


def check_dependencies():
    """Check if dependencies are installed."""
    try:
        import pkg_resources
        
        with open('dependencies.txt', 'r') as f:
            required = [line.strip() for line in f.readlines() if line.strip()]
        
        for package in required:
            name = package.split('>=')[0].strip()
            try:
                pkg_resources.get_distribution(name)
                print(f"✓ {name} is installed.")
            except pkg_resources.DistributionNotFound:
                print(f"✗ {name} is not installed.")
                return False
        return True
    except Exception as e:
        print(f"Error checking dependencies: {e}")
        return False


def install_dependencies():
    """Install dependencies from dependencies.txt."""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "dependencies.txt"])
        print("✓ Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies.")
        return False


def create_database_schema():
    """Create database schema if it doesn't exist."""
    print("Setting up database schema...")
    try:
        # Import after installing dependencies
        from database import create_tables
        create_tables()
        print("✓ Database schema created successfully.")
        return True
    except Exception as e:
        print(f"✗ Failed to create database schema: {e}")
        print("Make sure your DATABASE_URL environment variable is set correctly.")
        return False


def setup_env_variables():
    """Set up environment variables."""
    if os.path.exists('.env'):
        print("✓ .env file already exists.")
        return True
    
    if os.path.exists('.env.example'):
        shutil.copy('.env.example', '.env')
        print("✓ Created .env file from .env.example.")
        print("  Please edit .env with your database connection and API keys.")
        return True
    else:
        print("✗ .env.example file not found.")
        print("  Please create a .env file with your database connection and API keys.")
        return False


def main():
    """Main function."""
    print("Cryptocurrency Trading Analysis Platform Setup")
    print("=============================================")
    
    check_python_version()
    
    if not check_dependencies():
        print("Installing dependencies...")
        if not install_dependencies():
            print("Please install the dependencies manually:")
            print("pip install -r dependencies.txt")
            sys.exit(1)
    
    setup_env_variables()
    
    # Only try to create database schema if .env file exists
    if os.path.exists('.env'):
        create_database_schema()
    
    print("\nSetup complete!")
    print("\nTo run the application:")
    print("streamlit run app.py")


if __name__ == "__main__":
    main()