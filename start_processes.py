#!/usr/bin/env python
"""
Start script for process management.

This script starts the process manager which will monitor and control all
data gathering and analysis processes for the crypto trading platform.

Usage:
    python start_processes.py
"""

import os
import sys
import subprocess
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("process_start.log")
    ]
)

def main():
    """Start the process manager and initialize all processes."""
    # Check if process manager exists
    if not os.path.exists("process_manager.py"):
        logging.error("process_manager.py not found")
        print("Error: process_manager.py not found")
        return 1
    
    try:
        # Initialize all processes by using the 'start' command
        print("Initializing all processes...")
        subprocess.run(
            ["python", "process_manager.py", "start"],
            check=True
        )
        
        print("\nAll processes have been initialized.")
        print("Process manager is now running as a background workflow.")
        print("Use 'python manage_processes.py status' to check process status.")
        return 0
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Error starting processes: {e}")
        print(f"Error starting processes: {e}")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())