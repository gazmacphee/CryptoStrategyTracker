import os
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
from psycopg2 import sql

# Database configuration - get from environment variables with defaults
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "crypto")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")

def get_db_connection():
    """Create a database connection"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def create_tables():
    """Create tables if they don't exist"""
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database, skipping table creation")
        return
    
    try:
        cur = conn.cursor()
        
        # Create historical price data table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS historical_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            interval VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            close NUMERIC NOT NULL,
            volume NUMERIC NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, interval, timestamp)
        );
        """)
        
        # Create index for faster queries
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_historical_data_symbol_interval_timestamp 
        ON historical_data(symbol, interval, timestamp);
        """)
        
        conn.commit()
        print("Database tables created successfully")
    except psycopg2.Error as e:
        print(f"Error creating tables: {e}")
    finally:
        if conn:
            conn.close()

def save_historical_data(df, symbol, interval):
    """Save historical data to database"""
    if df.empty:
        return False
    
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        # Insert data row by row - not optimal but safer for small datasets
        insert_query = """
        INSERT INTO historical_data 
        (symbol, interval, timestamp, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, interval, timestamp) 
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            created_at = CURRENT_TIMESTAMP
        """
        
        data_tuples = []
        for _, row in df.iterrows():
            data_tuples.append((
                symbol,
                interval,
                row['timestamp'],
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume'])
            ))
        
        cur.executemany(insert_query, data_tuples)
        conn.commit()
        
        return True
    except psycopg2.Error as e:
        print(f"Error saving historical data: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_historical_data(symbol, interval, start_time, end_time):
    """Get historical data from database"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT timestamp, open, high, low, close, volume
        FROM historical_data
        WHERE symbol = %s
        AND interval = %s
        AND timestamp BETWEEN %s AND %s
        ORDER BY timestamp ASC
        """
        
        # Convert timestamps to datetime objects if they're not already
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        df = pd.read_sql_query(
            query, 
            conn, 
            params=(symbol, interval, start_time, end_time)
        )
        
        return df
    except psycopg2.Error as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()
