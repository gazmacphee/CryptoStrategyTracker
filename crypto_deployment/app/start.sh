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
pg_isready -h $PGHOST -p $PGPORT || { 
  echo "Unable to connect to PostgreSQL at $PGHOST:$PGPORT." 
  echo "Please ensure PostgreSQL is running and properly configured in .env file."
  exit 1
}

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

# Start Streamlit app
echo "Starting Streamlit application..."
streamlit run app.py --server.port 5000

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