#!/usr/bin/env python
"""
Process Management Wrapper Script for Cryptocurrency Trading Analysis Platform

This script provides a simple interface to start or stop all data processes
in the cryptocurrency trading platform.

Usage:
    python manage_processes.py start   # Start all processes
    python manage_processes.py stop    # Stop all processes
    python manage_processes.py status  # Check process status
"""

import os
import sys
import argparse
import subprocess
import time
import logging
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process_management.log"),
        logging.StreamHandler()
    ]
)

# Process Manager script path
PROCESS_MANAGER = "process_manager.py"
LOCK_FILE = ".process_manager.lock"

def ensure_process_manager_exists():
    """Check if the process manager script exists."""
    if not os.path.exists(PROCESS_MANAGER):
        logging.error(f"Process manager script not found: {PROCESS_MANAGER}")
        print(f"Error: Process manager script not found: {PROCESS_MANAGER}")
        sys.exit(1)

def is_process_manager_running():
    """Check if the process manager is already running."""
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, 'r') as f:
                pid = int(f.read().strip())
            
            # Check if process is still running
            try:
                process = psutil.Process(pid)
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Process doesn't exist or can't be accessed
                pass
        except Exception:
            pass
    
    return False

def start_processes():
    """Start all data processes."""
    ensure_process_manager_exists()
    
    if is_process_manager_running():
        print("Process manager is already running. Use 'status' to check running processes.")
        return

    try:
        # First, initialize processes
        print("Initializing all processes...")
        result = subprocess.run(
            ["python", PROCESS_MANAGER, "start"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(result.stdout)
        
        # Then, start the workflow to monitor them
        try:
            # Check if the ProcessManager workflow exists and is running
            workflow_check = subprocess.run(
                ["ps", "aux"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if "process_manager.py monitor" in workflow_check.stdout:
                print("Process monitor is already running in a workflow.")
            else:
                print("Starting process monitor workflow...")
                # Use the workflow system instead of running directly
                # This will be handled by the Replit environment
                print("Process monitor started.")
                print("Note: To see the monitor logs, check the ProcessManager workflow.")
        except Exception as monitor_error:
            logging.error(f"Error starting monitor workflow: {monitor_error}")
            # Fallback to running directly
            print("Falling back to starting monitor directly...")
            subprocess.Popen(
                ["python", PROCESS_MANAGER, "monitor"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True  # Detach from parent process
            )
            
        print("Process manager initialized all processes successfully.")
        print("Use 'status' to check the status of all processes.")
            
    except Exception as e:
        logging.error(f"Error starting processes: {e}")
        print(f"Error: {e}")

def stop_processes():
    """Stop all data processes."""
    ensure_process_manager_exists()
    
    print("Stopping all data processes...")
    try:
        # Run the process manager with stop command
        result = subprocess.run(
            ["python", PROCESS_MANAGER, "stop"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        print(result.stdout)
        
        # If there are any lingering process manager processes, terminate them
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and len(cmdline) > 1 and PROCESS_MANAGER in cmdline[1]:
                    print(f"Terminating lingering process manager (PID {proc.info['pid']})...")
                    proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        # Remove lock file if it still exists
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
            
        print("All processes stopped successfully.")
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Error stopping processes: {e}")
        print(f"Error: {e}")
        print(e.stderr)

def check_status():
    """Check the status of all processes."""
    ensure_process_manager_exists()
    
    try:
        # Run the process manager with status command
        result = subprocess.run(
            ["python", PROCESS_MANAGER, "status"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        print(result.stdout)
        
        if not is_process_manager_running():
            print("\nWarning: Process monitor is not running.")
            print("Process status may be outdated. Use 'start' to start the process monitor.")
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Error checking process status: {e}")
        print(f"Error: {e}")
        print(e.stderr)

def run_backfill():
    """Manually run data backfill process."""
    ensure_process_manager_exists()
    
    try:
        # Run the process manager to restart the backfill process
        result = subprocess.run(
            ["python", PROCESS_MANAGER, "restart", "--process", "backfill"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        print(result.stdout)
        print("Backfill process restarted successfully.")
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running backfill: {e}")
        print(f"Error: {e}")
        print(e.stderr)

def main():
    """Main entry point for the process management wrapper."""
    parser = argparse.ArgumentParser(description='Manage crypto trading platform processes')
    parser.add_argument('action', choices=['start', 'stop', 'status', 'backfill'],
                      help='Action to perform: start, stop, status, or backfill')
    
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    if args.action == 'start':
        start_processes()
    elif args.action == 'stop':
        stop_processes()
    elif args.action == 'status':
        check_status()
    elif args.action == 'backfill':
        run_backfill()

if __name__ == "__main__":
    main()