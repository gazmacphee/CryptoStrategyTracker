#!/bin/bash

# Start Cryptocurrency Trading Analysis Platform locally

# Check if Python is installed
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not installed. Aborting."; exit 1; }

# Load environment variables if .env exists
if [ -f .env ]; then
  echo "Loading environment variables from .env file..."
  export $(grep -v '^#' .env | xargs)
fi

# Check if PostgreSQL is running
if [ -z "$PGHOST" ]; then
  export PGHOST=localhost
fi

if [ -z "$PGPORT" ]; then
  export PGPORT=5432
fi

echo "Checking PostgreSQL connection..."
# Use Python to check DB connection instead of pg_isready which may not be available
python3 -c "
import psycopg2
import os
import sys
try:
    conn = psycopg2.connect(
        host=os.environ.get('PGHOST'),
        port=os.environ.get('PGPORT'),
        user=os.environ.get('PGUSER'),
        password=os.environ.get('PGPASSWORD'),
        database=os.environ.get('PGDATABASE'),
        sslmode='require'
    )
    conn.close()
    print('Database connection successful')
except Exception as e:
    print(f'Unable to connect to PostgreSQL at {os.environ.get(\"PGHOST\")}:{os.environ.get(\"PGPORT\")}')
    print(f'Error: {e}')
    sys.exit(1)
"

# Reset database if requested
if [ "$RESET_DB" = "true" ]; then
  echo "Resetting database..."
  python3 reset_database.py
fi

# Start background data backfill process if requested
if [ "$BACKFILL_ON_START" = "true" ]; then
  echo "Starting background data backfill process..."
  python3 backfill_database.py --continuous --interval=15 &
  BACKFILL_PID=$!
  echo "Backfill process started with PID: $BACKFILL_PID"
fi

# Generate Streamlit config using port from .env
echo "Generating Streamlit configuration..."
python3 generate_streamlit_config.py

# Get port from .env file with fallback to 5001
APP_PORT=${APP_PORT:-5001}
echo "Using application port: $APP_PORT"

# Start Streamlit app
echo "Starting Streamlit application..."
streamlit run app.py --server.port $APP_PORT

# Cleanup background process on exit
cleanup() {
  if [ ! -z "$BACKFILL_PID" ]; then
    echo "Stopping backfill process..."
    kill $BACKFILL_PID
  fi
  exit 0
}

trap cleanup SIGINT SIGTERM

wait