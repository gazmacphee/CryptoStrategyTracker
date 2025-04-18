import os
import toml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get port from environment variable, default to 5001 if not set
port = os.getenv("APP_PORT", "5001")

# Create the Streamlit config
config = {
    "server": {
        "headless": True,
        "address": "0.0.0.0",
        "port": int(port)  # Convert port to integer
    }
}

# Ensure the .streamlit directory exists
Path(".streamlit").mkdir(exist_ok=True)

# Write the config to .streamlit/config.toml
with open(".streamlit/config.toml", "w") as f:
    toml.dump(config, f)

print(f"Successfully generated Streamlit config with port {port}")