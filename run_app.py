#!/usr/bin/env python3
import os
import subprocess
import threading
import time
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Get the port from environment variable or set default to 5001
port = os.getenv("APP_PORT", "5001")

# First, generate the Streamlit config
print("Generating Streamlit configuration...")
subprocess.run(["python", "generate_streamlit_config.py"])

# Start the backfill process in the background
def start_backfill_process():
    print("Starting initial data backfill process...")
    try:
        # Always ensure the database is backfilled, even if the environment variable isn't set
        backfill_on_start = os.getenv("BACKFILL_ON_START", "true").lower()
        
        # Force backfill to be true for reliable operation
        if backfill_on_start == "true" or True:  # Always run backfill regardless of env setting
            # Start backfill in the background with continuous mode for regular updates
            subprocess.Popen(["python", "backfill_database.py", "--background", "--continuous", "--interval=15"], 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
            print("Backfill process started successfully in the background with continuous updates")
        else:
            print("Backfill is disabled in configuration, but this is not recommended")
    except Exception as e:
        print(f"Error starting backfill process: {e}")

# Start backfill in a separate thread so it doesn't block the main app
backfill_thread = threading.Thread(target=start_backfill_process)
backfill_thread.daemon = True
backfill_thread.start()

# Give the backfill process a moment to initialize
time.sleep(1)

# Run Streamlit with the port from .env
print(f"Starting Streamlit on port {port}...")
os.execvp("streamlit", ["streamlit", "run", "app.py", f"--server.port={port}", "--server.address=0.0.0.0"])