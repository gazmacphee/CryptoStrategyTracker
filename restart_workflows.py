#!/usr/bin/env python
"""
Script to restart Replit workflows.

This script helps restart the ProcessManager workflow and initialize processes
in the Replit environment. This is useful when processes need to be restarted
or when workflows have been terminated.

Usage:
    python restart_workflows.py
"""

import os
import subprocess
import sys
import argparse
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("workflow_restart.log")
    ]
)

def restart_process_manager():
    """Restart the ProcessManager workflow."""
    try:
        # Remove any lock files
        if os.path.exists('.process_manager.lock'):
            os.remove('.process_manager.lock')
            logging.info("Removed process manager lock file")
            print("Removed process manager lock file")
        
        # Check if the .process_info.json file exists and is corrupted
        if os.path.exists('.process_info.json'):
            try:
                with open('.process_info.json', 'r') as f:
                    import json
                    json.load(f)  # Try to parse
            except Exception:
                # File is corrupted, create a clean one
                with open('.process_info.json', 'w') as f:
                    f.write('{"processes": {}}')
                logging.info("Reset corrupted process info file")
                print("Reset corrupted process info file")
        
        # Restart the workflow using the Replit workflow mechanism
        print("Restarting ProcessManager workflow...")
        
        # We can't directly access the workflow APIs, so we'll use bash
        # This effectively stops and starts the workflow
        try:
            # First kill any running instances
            ps_output = subprocess.run(
                ["ps", "aux"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            ).stdout
            
            for line in ps_output.split('\n'):
                if 'process_manager.py monitor' in line:
                    # Extract PID and kill process
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pid = int(parts[1])
                            subprocess.run(['kill', str(pid)])
                            logging.info(f"Killed process manager with PID {pid}")
                            print(f"Killed process manager with PID {pid}")
                        except Exception as e:
                            logging.error(f"Error killing process: {e}")
        except Exception as e:
            logging.error(f"Error stopping existing processes: {e}")
        
        # Start a new workflow
        try:
            subprocess.run(
                ["python", "process_manager.py", "monitor"],
                stdout=open("process_manager.log", "a"),
                stderr=open("process_manager.log", "a"),
                start_new_session=True  # Detach from parent process
            )
            logging.info("Started new ProcessManager workflow")
            print("Started new ProcessManager workflow")
        except Exception as e:
            logging.error(f"Error starting workflow: {e}")
            print(f"Error starting workflow: {e}")
        
        # Give the monitor a moment to start
        time.sleep(2)
        
        # Initialize processes
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
        
        logging.info("Process manager workflow restarted successfully")
        print("\nProcess manager workflow restarted successfully")
        print("Use 'python manage_processes.py status' to check process status")
        
        return 0
        
    except subprocess.TimeoutExpired:
        logging.warning("Process initialization is taking longer than expected")
        print("Process initialization is taking longer than expected.")
        print("This is normal during first startup. Check status in a minute.")
        return 0
    except Exception as e:
        logging.error(f"Error restarting process manager: {e}")
        print(f"Error restarting process manager: {e}")
        return 1

def main():
    """Main entry point for workflow restart script."""
    parser = argparse.ArgumentParser(description='Restart Replit workflows')
    parser.add_argument('--force', action='store_true', help='Force restart even if already running')
    
    args = parser.parse_args()
    
    return restart_process_manager()

if __name__ == "__main__":
    sys.exit(main())