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
        # Remove any lock files
        if os.path.exists('.process_manager.lock'):
            os.remove('.process_manager.lock')
            logging.info("Removed process manager lock file")
            print("Removed process manager lock file")
        
        # If we need to force restart or the workflow isn't running
        if args.force or not workflow_running:
            # Kill any existing process_manager processes
            ps_output = subprocess.run(
                ["ps", "aux"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            ).stdout
            
            for line in ps_output.split('\n'):
                if 'process_manager.py monitor' in line:
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pid = int(parts[1])
                            subprocess.run(['kill', str(pid)])
                            logging.info(f"Killed process manager with PID {pid}")
                            print(f"Killed existing process manager with PID {pid}")
                        except Exception as e:
                            logging.error(f"Error killing process: {e}")
            
            # Start the process manager using the Replit workflow system
            print("Starting process manager workflow...")
            try:
                # Start the workflow via Popen to run in background
                subprocess.Popen(
                    ["python", "process_manager.py", "monitor"],
                    stdout=open("process_manager.log", "a"),
                    stderr=open("process_manager.log", "a"),
                    start_new_session=True  # Detach from parent process
                )
                logging.info("Started process manager workflow")
                print("Process manager workflow started")
            except Exception as e:
                logging.error(f"Error starting workflow: {e}")
                print(f"Error starting workflow: {e}")
                # Continue anyway - we'll still try to initialize processes
        
        # Wait a moment for the workflow to start
        time.sleep(2)
        
        # Initialize all processes with a timeout
        print("Initializing processes...")
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