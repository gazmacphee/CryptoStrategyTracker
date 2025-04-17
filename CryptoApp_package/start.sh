#!/bin/bash

# Start script for local development environment (not Docker)

# Check if database exists and can be accessed
echo "Checking database connection..."
python -c "from database import get_db_connection; conn = get_db_connection(); print('Database connection successful!' if conn else 'Database connection failed!'); conn.close() if conn else None"

if [ $? -ne 0 ]; then
    echo "Database connection failed. Please check your environment variables."
    exit 1
fi

# Create database tables if they don't exist
echo "Initializing database tables..."
python -c "from database import create_tables; create_tables()"

# Start the application
echo "Starting Cryptocurrency Trading Analysis Platform..."
streamlit run app.py --server.port 5000