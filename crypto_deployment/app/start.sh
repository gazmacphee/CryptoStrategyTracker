#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Clear any existing backfill lock
if [ -f .backfill_lock ]; then
    rm .backfill_lock
    echo "Removed existing backfill lock"
fi

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL database to be ready..."
for i in {1..30}; do
    if pg_isready -h $PGHOST -p $PGPORT -U $PGUSER; then
        echo "PostgreSQL is ready!"
        break
    fi
    echo "Waiting for PostgreSQL to be ready... $i/30"
    sleep 2
done

# Setup database tables if needed
echo "Setting up database tables..."
python -c "from database import create_tables; create_tables()"

# Start the application
echo "Starting Streamlit application..."
streamlit run app.py --server.port 5000 --server.address 0.0.0.0 --server.headless true