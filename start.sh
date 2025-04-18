#!/bin/bash

# Start script for Cryptocurrency Trading Platform
# This script will:
# 1. Check if the database exists and create it if needed
# 2. Reset the database (drop all tables)
# 3. Create fresh tables
# 4. Start the backfill process to download cryptocurrency data
# 5. Launch the Streamlit application

echo "==============================================================="
echo "  CRYPTOCURRENCY TRADING PLATFORM STARTUP"
echo "  $(date)"
echo "==============================================================="

# Run the Python setup script
echo "Running database reset and backfill script..."
python reset_and_start.py

# Exit with success
exit 0