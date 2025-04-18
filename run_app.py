#!/usr/bin/env python3
import os
import subprocess
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Get the port from environment variable or set default to 5001
port = os.getenv("APP_PORT", "5001")

# First, generate the Streamlit config
print("Generating Streamlit configuration...")
subprocess.run(["python", "generate_streamlit_config.py"])

# Run Streamlit with the port from .env
print(f"Starting Streamlit on port {port}...")
os.execvp("streamlit", ["streamlit", "run", "app.py", f"--server.port={port}", "--server.address=0.0.0.0"])