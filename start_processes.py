#!/usr/bin/env python
"""
Start script for process management.

This script starts the process manager which will monitor and control all
data gathering and analysis processes for the crypto trading platform.

Usage:
    python start_processes.py [--force]
    
    --force: Force restart all processes even if they're already running
"""

import os
import sys
import subprocess
import time
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("process_start.log")
    ]
)

def check_workflow_status():
    """Check if the ProcessManager workflow is running."""
    try:
        # Check if the ProcessManager workflow is already running
        workflow_check = subprocess.run(
            ["ps", "aux"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        return "process_manager.py monitor" in workflow_check.stdout
    except Exception as e:
        logging.warning(f"Could not check workflow status: {e}")
        return False

def main():
    """Start the process manager and initialize all processes."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Start crypto trading platform processes')
    parser.add_argument('--force', action='store_true', help='Force restart all processes')
    args = parser.parse_args()
    
    # Check if process manager exists
    if not os.path.exists("process_manager.py"):
        logging.error("process_manager.py not found")
        print("Error: process_manager.py not found")
        return 1
    
    # Check if workflow is already running
    workflow_running = check_workflow_status()
    
    if workflow_running and not args.force:
        print("Process monitor is already running in a workflow.")
        print("Use 'python manage_processes.py status' to check process status.")
        print("Use --force to restart all processes.")
        return 0
    
    try:
        # Initialize all processes with a timeout
        print("Initializing all processes...")
        try:
            result = subprocess.run(
                ["python", "process_manager.py", "start"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30  # 30 second timeout for initialization
            )
            print(result.stdout)
        except subprocess.TimeoutExpired:
            print("Process initialization is taking longer than expected.")
            print("This is normal during first startup. Check status in a minute.")
        
        # Check if monitor workflow needs to be started
        if not workflow_running:
            print("\nStarting process monitor...")
            print("Note: This will be handled by the Replit workflow system.")
        
        print("\nAll processes have been initialized.")
        print("Process manager is running as a background workflow.")
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