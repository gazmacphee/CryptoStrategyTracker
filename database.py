import os
from dotenv import load_dotenv
import psycopg2
import pandas as pd

# Load environment variables
load_dotenv()
from datetime import datetime, timedelta
from psycopg2 import sql

# Helper function to replace pandas read_sql_query to avoid SQLAlchemy dependency issues
def execute_sql_to_df(query, conn, params=None):
    """
    Execute SQL query and return results as a pandas DataFrame without using SQLAlchemy
    
    Args:
        query: SQL query string
        conn: psycopg2 connection object
        params: Parameters for the query (optional)
        
    Returns:
        pandas DataFrame with query results
    """
    try:
        cur = conn.cursor()
        cur.execute(query, params)
        
        # Fetch results and column names
        results = cur.fetchall()
        column_names = [desc[0] for desc in cur.description]
        cur.close()
        
        # Create pandas DataFrame from the results
        return pd.DataFrame(results, columns=column_names)
    except Exception as e:
        print(f"Error executing SQL: {e}")
        return pd.DataFrame()

# Database configuration - get from environment variables with defaults
DATABASE_URL = os.getenv("DATABASE_URL", "")
print(f"Original DATABASE_URL from environment: {DATABASE_URL}")

# Fix for Docker environment remnants
if DATABASE_URL == "db:5432/crypto" or DATABASE_URL == "postgresql://postgres:postgres@db:5432/crypto":
    print("Detected Docker database URL, checking for local PostgreSQL database")
    
    # Try local PostgreSQL database first (with user's credentials)
    local_user = "postgres"
    local_password = "2212"
    local_host = "localhost"
    local_port = "5432"
    local_db = "crypto"
    local_url = f"postgresql://{local_user}:{local_password}@{local_host}:{local_port}/{local_db}"
    
    try:
        # Test connection to local database
        import psycopg2
        test_conn = psycopg2.connect(local_url)
        test_conn.close()
        print(f"Successfully connected to local PostgreSQL database")
        DATABASE_URL = local_url
    except Exception as local_err:
        print(f"Unable to connect to local PostgreSQL database: {local_err}")
        # Fall back to Neon PostgreSQL configuration
        DATABASE_URL = os.getenv("DATABASE_URL_NEON", "postgresql://neondb_owner:npg_8TbOZL2KvPxB@ep-silent-star-a5groo1f.us-east-2.aws.neon.tech/neondb?sslmode=require")
    
    print(f"Updated DATABASE_URL: {DATABASE_URL}")

# Parse DATABASE_URL if available
if DATABASE_URL:
    # Parse the URL to get components
    try:
        # Handle postgresql:// urls
        if DATABASE_URL.startswith("postgresql://"):
            # Strip off postgres:// part
            user_pass_host_port_db = DATABASE_URL[len("postgresql://"):]
            user_pass, host_port_db = user_pass_host_port_db.split("@", 1)
            
            if ":" in user_pass:
                DB_USER, DB_PASS = user_pass.split(":", 1)
            else:
                DB_USER = user_pass
                DB_PASS = ""
                
            if "/" in host_port_db:
                host_port, DB_NAME = host_port_db.split("/", 1)
                # Remove any query parameters
                if "?" in DB_NAME:
                    DB_NAME = DB_NAME.split("?", 1)[0]
            else:
                host_port = host_port_db
                DB_NAME = ""
                
            if ":" in host_port:
                DB_HOST, DB_PORT = host_port.split(":", 1)
            else:
                DB_HOST = host_port
                DB_PORT = "5432"
            
            print(f"Parsed URL - Host: {DB_HOST}, Port: {DB_PORT}, DB: {DB_NAME}, User: {DB_USER}")
        else:
            # Non-postgresql:// URL format
            print(f"Non-standard DATABASE_URL format: {DATABASE_URL}")
            # Fallback to environment variables
            DB_HOST = os.getenv("PGHOST", "localhost")
            DB_PORT = os.getenv("PGPORT", "5432")
            DB_NAME = os.getenv("PGDATABASE", "crypto")
            DB_USER = os.getenv("PGUSER", "postgres")
            DB_PASS = os.getenv("PGPASSWORD", "postgres")
            print(f"Using environment variables - Host: {DB_HOST}, Port: {DB_PORT}, DB: {DB_NAME}, User: {DB_USER}")
    except Exception as e:
        print(f"Error parsing DATABASE_URL: {e}, using environment variables instead")
        DB_HOST = os.getenv("PGHOST", "localhost")
        DB_PORT = os.getenv("PGPORT", "5432")
        DB_NAME = os.getenv("PGDATABASE", "crypto")
        DB_USER = os.getenv("PGUSER", "postgres")
        DB_PASS = os.getenv("PGPASSWORD", "postgres")
        print(f"After error - Host: {DB_HOST}, Port: {DB_PORT}, DB: {DB_NAME}, User: {DB_USER}")
else:
    # Use environment variables
    print("No DATABASE_URL found, using individual environment variables")
    DB_HOST = os.getenv("PGHOST", "localhost")
    DB_PORT = os.getenv("PGPORT", "5432")
    DB_NAME = os.getenv("PGDATABASE", "crypto")
    DB_USER = os.getenv("PGUSER", "postgres")
    DB_PASS = os.getenv("PGPASSWORD", "postgres")
    print(f"Environment variables - Host: {DB_HOST}, Port: {DB_PORT}, DB: {DB_NAME}, User: {DB_USER}")

def close_db_connection(conn):
    """
    Safely close a database connection
    
    Args:
        conn: Database connection to close
    """
    if conn:
        try:
            conn.close()
            print("Database connection closed successfully")
        except Exception as e:
            print(f"Error closing database connection: {e}")
    
def get_db_connection():
    """Create a database connection with retry capability for various environments"""
    import time
    
    # Maximum number of connection attempts
    max_attempts = 5
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        try:
            # Fix for Docker remnants - check if DATABASE_URL is Docker format
            if DATABASE_URL == "db:5432/crypto" or DATABASE_URL == "postgresql://postgres:postgres@db:5432/crypto":
                print("Detected Docker database URL in connection function, trying local PostgreSQL database")
                
                # Try local PostgreSQL database first (with user's credentials)
                local_user = "postgres"
                local_password = "2212"
                local_host = "localhost"
                local_port = "5432"
                local_db = "crypto"
                local_url = f"postgresql://{local_user}:{local_password}@{local_host}:{local_port}/{local_db}"
                
                try:
                    # Try connecting to local database
                    conn = psycopg2.connect(local_url)
                    print("Successfully connected to local PostgreSQL database")
                    return conn
                except Exception as local_err:
                    print(f"Unable to connect to local PostgreSQL database: {local_err}")
                    # Fall back to Neon PostgreSQL
                    actual_url = os.getenv("DATABASE_URL_NEON", 
                        "postgresql://neondb_owner:npg_8TbOZL2KvPxB@ep-silent-star-a5groo1f.us-east-2.aws.neon.tech/neondb?sslmode=require")
                    print(f"Using alternative URL: {actual_url.split('@')[1] if '@' in actual_url else 'custom-url'}")
                    conn = psycopg2.connect(actual_url)
                    print("Successfully connected using alternative DATABASE_URL")
                    return conn
            # Try connecting using DATABASE_URL directly if available and not Docker format
            elif DATABASE_URL:
                print(f"Connecting to database using DATABASE_URL: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'custom-url'}")
                try:
                    conn = psycopg2.connect(DATABASE_URL)
                    print("Successfully connected using DATABASE_URL")
                    return conn
                except Exception as url_error:
                    print(f"Error connecting with DATABASE_URL: {url_error}, trying individual parameters")
                    # Fall back to individual parameters
                    print(f"Connecting to database at {DB_HOST}:{DB_PORT}/{DB_NAME} with user {DB_USER}")
                    conn = psycopg2.connect(
                        host=DB_HOST,
                        port=DB_PORT,
                        database=DB_NAME,
                        user=DB_USER,
                        password=DB_PASS
                    )
                    print("Successfully connected using individual parameters")
                    return conn
            else:
                # Fall back to individual parameters
                print(f"Connecting to database at {DB_HOST}:{DB_PORT}/{DB_NAME} with user {DB_USER}")
                conn = psycopg2.connect(
                    host=DB_HOST,
                    port=DB_PORT,
                    database=DB_NAME,
                    user=DB_USER,
                    password=DB_PASS
                )
                print("Successfully connected using individual parameters")
                return conn
        except psycopg2.Error as e:
            print(f"Attempt {attempt}/{max_attempts} - Error connecting to database: {e}")
            if attempt < max_attempts:
                # Wait before retrying (exponential backoff)
                wait_time = 2 ** attempt
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Maximum connection attempts reached. Could not connect to database.")
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
        
        # Create indicators table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS technical_indicators (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            interval VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            rsi NUMERIC,
            macd NUMERIC,
            macd_signal NUMERIC,
            macd_histogram NUMERIC,
            bb_upper NUMERIC,
            bb_middle NUMERIC,
            bb_lower NUMERIC,
            bb_percent NUMERIC,
            ema_9 NUMERIC,
            ema_21 NUMERIC,
            ema_50 NUMERIC,
            ema_200 NUMERIC,
            buy_signal BOOLEAN DEFAULT FALSE,
            sell_signal BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, interval, timestamp)
        );
        """)
        
        # Create index for indicators table
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_indicators_symbol_interval_timestamp 
        ON technical_indicators(symbol, interval, timestamp);
        """)
        
        # Create trades table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            interval VARCHAR(10) NOT NULL,
            type VARCHAR(10) NOT NULL,  -- BUY, SELL, SELL (EOT)
            entry_timestamp TIMESTAMP,
            exit_timestamp TIMESTAMP,
            entry_price NUMERIC,
            exit_price NUMERIC,
            quantity NUMERIC,
            profit NUMERIC,
            profit_pct NUMERIC,
            holding_time_hours NUMERIC,
            trade_status VARCHAR(10) NOT NULL, -- OPEN, CLOSED
            strategy_params JSONB,  -- Stores the strategy parameters used
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Create index for trades table
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_trades_symbol_interval 
        ON trades(symbol, interval);
        """)
        
        # Create portfolio table for tracking crypto holdings
        cur.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            quantity NUMERIC NOT NULL,
            purchase_price NUMERIC NOT NULL,
            purchase_date TIMESTAMP NOT NULL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, purchase_date)
        );
        """)
        
        # Create index for portfolio table
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_portfolio_symbol 
        ON portfolio(symbol);
        """)
        
        # Create benchmark data table for tracking comparative indices
        cur.execute("""
        CREATE TABLE IF NOT EXISTS benchmarks (
            id SERIAL PRIMARY KEY,
            name VARCHAR(50) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            value NUMERIC NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name, timestamp)
        );
        """)
        
        # Create index for benchmarks table
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_benchmarks_name_timestamp 
        ON benchmarks(name, timestamp);
        """)
        
        # Create sentiment data table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            source VARCHAR(50) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            sentiment_score NUMERIC NOT NULL,
            volume INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, source, timestamp)
        );
        """)
        
        # Create index for sentiment data
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_source_timestamp
        ON sentiment_data(symbol, source, timestamp);
        """)
        
        # Create news data table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS news_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20),
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            source VARCHAR(100) NOT NULL,
            url TEXT,
            author VARCHAR(100),
            published_at TIMESTAMP NOT NULL,
            sentiment_score NUMERIC,
            relevance_score NUMERIC,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Create index for news data
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_news_symbol_published
        ON news_data(symbol, published_at);
        """)
        
        # Create ML predictions table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS ml_predictions (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            interval VARCHAR(10) NOT NULL,
            prediction_timestamp TIMESTAMP NOT NULL,
            target_timestamp TIMESTAMP NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            predicted_price NUMERIC,
            predicted_change_pct NUMERIC,
            confidence_score NUMERIC,
            features_used JSONB,
            prediction_type VARCHAR(20) NOT NULL,  -- price, direction, pattern
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, interval, prediction_timestamp, target_timestamp, model_name)
        );
        """)
        
        # Create index for ML predictions
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol_interval
        ON ml_predictions(symbol, interval, prediction_timestamp);
        """)
        
        # Create ML model performance tracking table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS ml_model_performance (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            interval VARCHAR(10) NOT NULL,
            training_timestamp TIMESTAMP NOT NULL,
            accuracy NUMERIC,
            precision NUMERIC,
            recall NUMERIC,
            f1_score NUMERIC,
            mse NUMERIC,
            mae NUMERIC,
            training_params JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(model_name, symbol, interval, training_timestamp)
        );
        """)
        
        # Create index for ML model performance
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_ml_performance_model
        ON ml_model_performance(model_name, symbol, interval);
        """)
        
        # Create detected patterns table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS detected_patterns (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            interval VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            pattern_type VARCHAR(50) NOT NULL,
            pattern_strength NUMERIC NOT NULL,
            expected_outcome VARCHAR(20) NOT NULL,  -- bullish, bearish, neutral
            confidence_score NUMERIC NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, interval, timestamp, pattern_type)
        );
        """)
        
        # Create index for detected patterns
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_patterns_symbol_interval
        ON detected_patterns(symbol, interval, timestamp);
        """)
        
        # Create US Dollar Index (DXY) table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS dollar_index (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            close NUMERIC NOT NULL,
            open NUMERIC,
            high NUMERIC,
            low NUMERIC,
            volume NUMERIC,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp)
        );
        """)
        
        # Create index for Dollar Index
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_dollar_index_timestamp
        ON dollar_index(timestamp);
        """)
        
        # Create Global Liquidity Indicators table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS global_liquidity (
            id SERIAL PRIMARY KEY,
            indicator_name VARCHAR(50) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            value NUMERIC NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(indicator_name, timestamp)
        );
        """)
        
        # Create index for Global Liquidity
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_global_liquidity_timestamp
        ON global_liquidity(indicator_name, timestamp);
        """)
        
        # Create trading signals table for historical tracking
        cur.execute("""
        CREATE TABLE IF NOT EXISTS trading_signals (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            interval VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            signal_type VARCHAR(10) NOT NULL,  -- 'buy', 'sell', or 'neutral'
            signal_strength NUMERIC,           -- 0.0 to 1.0 indicating signal strength
            price NUMERIC NOT NULL,            -- price at signal generation
            stop_loss NUMERIC,                 -- suggested stop loss price level
            take_profit NUMERIC,               -- suggested take profit price level
            stop_loss_method VARCHAR(50),      -- method used to calculate stop loss
            take_profit_method VARCHAR(50),    -- method used to calculate take profit
            risk_reward_ratio NUMERIC,         -- risk to reward ratio (e.g., 1:2, 1:3)
            bb_signal BOOLEAN,                 -- individual indicator signals
            rsi_signal BOOLEAN,
            macd_signal BOOLEAN,
            ema_signal BOOLEAN,                -- ema crossover signals
            strategy_name VARCHAR(100),        -- which strategy generated this signal
            strategy_params JSONB,             -- strategy parameters used
            notes TEXT,                        -- additional context
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, interval, timestamp, strategy_name)
        );
        """)
        
        # Create index for trading signals
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol_interval_timestamp 
        ON trading_signals(symbol, interval, timestamp);
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
        
        # Use custom function instead of pandas read_sql_query to avoid SQLAlchemy dependency
        df = execute_sql_to_df(query, conn, params=(symbol, interval, start_time, end_time))
        
        return df
    except psycopg2.Error as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def get_existing_data_months(symbol, interval, start_year, start_month, end_year, end_month):
    """
    Get a list of year-month combinations that already have data in the database.
    This helps optimize data downloading by skipping months that are already stored.
    
    Returns:
        List of tuples (year, month) that have data
    """
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        query = """
        SELECT DISTINCT 
            EXTRACT(YEAR FROM timestamp)::INTEGER as year,
            EXTRACT(MONTH FROM timestamp)::INTEGER as month
        FROM historical_data
        WHERE symbol = %s
        AND interval = %s
        AND timestamp BETWEEN %s AND %s
        ORDER BY year, month
        """
        
        # Create date range
        start_date = datetime(start_year, start_month, 1)
        if end_month == 12:
            end_date = datetime(end_year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(end_year, end_month + 1, 1) - timedelta(days=1)
        
        # Use custom function instead of pandas read_sql_query to avoid SQLAlchemy dependency
        df = execute_sql_to_df(query, conn, params=(symbol, interval, start_date, end_date))
        
        if df.empty:
            return []
        
        # Convert to list of tuples
        return list(zip(df['year'].astype(int).tolist(), df['month'].astype(int).tolist()))
    except psycopg2.Error as e:
        print(f"Error getting existing data months: {e}")
        return []
    finally:
        if conn:
            conn.close()

def has_complete_month_data(symbol, interval, year, month):
    """
    Check if a specific month has complete data in the database.
    Returns True if the month has expected number of records.
    """
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        query = """
        SELECT COUNT(*) as record_count
        FROM historical_data
        WHERE symbol = %s
        AND interval = %s
        AND EXTRACT(YEAR FROM timestamp) = %s
        AND EXTRACT(MONTH FROM timestamp) = %s
        """
        
        cur = conn.cursor()
        cur.execute(query, (symbol, interval, year, month))
        record_count = cur.fetchone()[0]
        cur.close()
        
        # Calculate expected records
        days_in_month = 31  # Approximate for most months
        if month == 2:
            # February
            if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                days_in_month = 29  # Leap year
            else:
                days_in_month = 28
        elif month in [4, 6, 9, 11]:
            # April, June, September, November
            days_in_month = 30
        
        # Expected records based on interval
        if interval == '1m':
            expected = days_in_month * 24 * 60
        elif interval == '3m':
            expected = days_in_month * 24 * 20
        elif interval == '5m':
            expected = days_in_month * 24 * 12
        elif interval == '15m':
            expected = days_in_month * 24 * 4
        elif interval == '30m':
            expected = days_in_month * 24 * 2
        elif interval == '1h':
            expected = days_in_month * 24
        elif interval == '2h':
            expected = days_in_month * 12
        elif interval == '4h':
            expected = days_in_month * 6
        elif interval == '6h':
            expected = days_in_month * 4
        elif interval == '8h':
            expected = days_in_month * 3
        elif interval == '12h':
            expected = days_in_month * 2
        elif interval == '1d':
            expected = days_in_month
        elif interval == '3d':
            expected = days_in_month // 3
        elif interval == '1w':
            expected = days_in_month // 7
        else:
            expected = 0
        
        # Check if we have at least 90% of expected records
        # This accounts for partial months, market closures, etc.
        threshold = expected * 0.9
        
        print(f"Month {year}-{month} for {symbol}/{interval}: {record_count} records (expected ~{expected})")
        return record_count >= threshold
    except psycopg2.Error as e:
        print(f"Error checking month completeness: {e}")
        return False
    finally:
        if conn:
            conn.close()

def save_indicators(df, symbol, interval):
    """Save technical indicators to database"""
    if df.empty:
        return False
    
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        # Insert data row by row
        insert_query = """
        INSERT INTO technical_indicators 
        (symbol, interval, timestamp, rsi, macd, macd_signal, macd_histogram,
         bb_upper, bb_middle, bb_lower, bb_percent, 
         ema_9, ema_21, ema_50, ema_200, 
         buy_signal, sell_signal)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, interval, timestamp) 
        DO UPDATE SET
            rsi = EXCLUDED.rsi,
            macd = EXCLUDED.macd,
            macd_signal = EXCLUDED.macd_signal,
            macd_histogram = EXCLUDED.macd_histogram,
            bb_upper = EXCLUDED.bb_upper,
            bb_middle = EXCLUDED.bb_middle,
            bb_lower = EXCLUDED.bb_lower,
            bb_percent = EXCLUDED.bb_percent,
            ema_9 = EXCLUDED.ema_9,
            ema_21 = EXCLUDED.ema_21,
            ema_50 = EXCLUDED.ema_50,
            ema_200 = EXCLUDED.ema_200,
            buy_signal = EXCLUDED.buy_signal,
            sell_signal = EXCLUDED.sell_signal,
            created_at = CURRENT_TIMESTAMP
        """
        
        data_tuples = []
        for _, row in df.iterrows():
            data_tuples.append((
                symbol,
                interval,
                row['timestamp'],
                float(row['RSI']) if 'RSI' in row and not pd.isna(row['RSI']) else None,
                float(row['MACD']) if 'MACD' in row and not pd.isna(row['MACD']) else None,
                float(row['MACD_signal']) if 'MACD_signal' in row and not pd.isna(row['MACD_signal']) else None,
                float(row['MACD_histogram']) if 'MACD_histogram' in row and not pd.isna(row['MACD_histogram']) else None,
                float(row['BB_upper']) if 'BB_upper' in row and not pd.isna(row['BB_upper']) else None,
                float(row['BB_middle']) if 'BB_middle' in row and not pd.isna(row['BB_middle']) else None,
                float(row['BB_lower']) if 'BB_lower' in row and not pd.isna(row['BB_lower']) else None,
                float(row['BB_percent']) if 'BB_percent' in row and not pd.isna(row['BB_percent']) else None,
                float(row['EMA_9']) if 'EMA_9' in row and not pd.isna(row['EMA_9']) else None,
                float(row['EMA_21']) if 'EMA_21' in row and not pd.isna(row['EMA_21']) else None,
                float(row['EMA_50']) if 'EMA_50' in row and not pd.isna(row['EMA_50']) else None,
                float(row['EMA_200']) if 'EMA_200' in row and not pd.isna(row['EMA_200']) else None,
                bool(row['buy_signal']) if 'buy_signal' in row else False,
                bool(row['sell_signal']) if 'sell_signal' in row else False
            ))
        
        cur.executemany(insert_query, data_tuples)
        conn.commit()
        
        return True
    except psycopg2.Error as e:
        print(f"Error saving indicators: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_indicators(symbol, interval, start_time, end_time):
    """Get indicators from database"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT timestamp, rsi, macd, macd_signal, macd_histogram,
               bb_upper, bb_middle, bb_lower, bb_percent,
               ema_9, ema_21, ema_50, ema_200, 
               buy_signal, sell_signal
        FROM technical_indicators
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
        
        # Use custom function instead of pandas read_sql_query to avoid SQLAlchemy dependency
        df = execute_sql_to_df(query, conn, params=(symbol, interval, start_time, end_time))
        
        return df
    except psycopg2.Error as e:
        print(f"Error fetching indicators: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def save_trade(trade_data):
    """Save a trade to the database"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        # Create query based on trade type (buy or sell)
        if trade_data['type'] == 'BUY':
            query = """
            INSERT INTO trades 
            (symbol, interval, type, entry_timestamp, entry_price, quantity, trade_status, strategy_params)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            RETURNING id
            """
            params = (
                trade_data['symbol'],
                trade_data['interval'],
                trade_data['type'],
                trade_data['timestamp'],
                trade_data['price'],
                trade_data['quantity'],
                'OPEN',
                trade_data.get('strategy_params', '{}')
            )
        else:  # SELL
            # First, get the open trade
            open_trade_query = """
            SELECT id, entry_price, entry_timestamp
            FROM trades
            WHERE symbol = %s
            AND interval = %s
            AND trade_status = 'OPEN'
            ORDER BY entry_timestamp DESC
            LIMIT 1
            """
            
            cur.execute(open_trade_query, (trade_data['symbol'], trade_data['interval']))
            open_trade = cur.fetchone()
            
            if not open_trade:
                print(f"No open trade found for {trade_data['symbol']}/{trade_data['interval']}")
                return False
            
            trade_id, entry_price, entry_timestamp = open_trade
            
            # Calculate profit
            exit_price = trade_data['price']
            profit = (exit_price - entry_price) * trade_data['quantity']
            profit_pct = ((exit_price / entry_price) - 1) * 100
            
            # Calculate holding time in hours
            exit_timestamp = trade_data['timestamp']
            holding_time = (exit_timestamp - entry_timestamp).total_seconds() / 3600
            
            # Update the trade
            query = """
            UPDATE trades
            SET exit_timestamp = %s,
                exit_price = %s,
                profit = %s,
                profit_pct = %s,
                holding_time_hours = %s,
                trade_status = 'CLOSED',
                type = %s
            WHERE id = %s
            RETURNING id
            """
            
            params = (
                exit_timestamp,
                exit_price,
                profit,
                profit_pct,
                holding_time,
                trade_data['type'],  # This could be 'SELL' or 'SELL (EOT)'
                trade_id
            )
        
        cur.execute(query, params)
        trade_id = cur.fetchone()[0]
        conn.commit()
        
        return trade_id
    except psycopg2.Error as e:
        print(f"Error saving trade: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_open_trade(symbol, interval):
    """Get the currently open trade for a symbol/interval if one exists"""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        query = """
        SELECT id, symbol, interval, type, entry_timestamp, entry_price, quantity
        FROM trades
        WHERE symbol = %s
        AND interval = %s
        AND trade_status = 'OPEN'
        ORDER BY entry_timestamp DESC
        LIMIT 1
        """
        
        # Use custom function instead of pandas read_sql_query to avoid SQLAlchemy dependency
        df = execute_sql_to_df(query, conn, params=(symbol, interval))
        
        if df.empty:
            return None
        
        # Convert to dictionary
        trade = df.iloc[0].to_dict()
        return trade
    except psycopg2.Error as e:
        print(f"Error getting open trade: {e}")
        return None
    finally:
        if conn:
            conn.close()

def add_portfolio_item(symbol, quantity, purchase_price, purchase_date, notes=None):
    """Add a new cryptocurrency to the portfolio"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        # Convert purchase_date to datetime if it's a string
        if isinstance(purchase_date, str):
            purchase_date = datetime.fromisoformat(purchase_date.replace('Z', '+00:00'))
        
        query = """
        INSERT INTO portfolio 
        (symbol, quantity, purchase_price, purchase_date, notes)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """
        
        cur.execute(query, (symbol, quantity, purchase_price, purchase_date, notes))
        item_id = cur.fetchone()[0]
        conn.commit()
        
        return item_id
    except psycopg2.Error as e:
        print(f"Error adding portfolio item: {e}")
        return False
    finally:
        if conn:
            conn.close()

def update_portfolio_item(item_id, quantity=None, purchase_price=None, purchase_date=None, notes=None):
    """Update an existing portfolio item"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        # Build the update query dynamically based on what's provided
        update_fields = []
        params = []
        
        if quantity is not None:
            update_fields.append("quantity = %s")
            params.append(quantity)
        
        if purchase_price is not None:
            update_fields.append("purchase_price = %s")
            params.append(purchase_price)
        
        if purchase_date is not None:
            if isinstance(purchase_date, str):
                purchase_date = datetime.fromisoformat(purchase_date.replace('Z', '+00:00'))
            update_fields.append("purchase_date = %s")
            params.append(purchase_date)
        
        if notes is not None:
            update_fields.append("notes = %s")
            params.append(notes)
        
        if not update_fields:
            return False  # Nothing to update
        
        # Add the item_id parameter
        params.append(item_id)
        
        query = f"""
        UPDATE portfolio 
        SET {', '.join(update_fields)}
        WHERE id = %s
        RETURNING id
        """
        
        cur.execute(query, params)
        updated_id = cur.fetchone()
        conn.commit()
        
        return updated_id is not None
    except psycopg2.Error as e:
        print(f"Error updating portfolio item: {e}")
        return False
    finally:
        if conn:
            conn.close()

def delete_portfolio_item(item_id):
    """Delete a portfolio item"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        query = """
        DELETE FROM portfolio
        WHERE id = %s
        RETURNING id
        """
        
        cur.execute(query, (item_id,))
        deleted_id = cur.fetchone()
        conn.commit()
        
        return deleted_id is not None
    except psycopg2.Error as e:
        print(f"Error deleting portfolio item: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_portfolio():
    """Get all portfolio items"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT id, symbol, quantity, purchase_price, purchase_date, notes, created_at
        FROM portfolio
        ORDER BY symbol, purchase_date
        """
        
        # Use custom function instead of pandas read_sql_query to avoid SQLAlchemy dependency
        df = execute_sql_to_df(query, conn)
        
        if df.empty:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['id', 'symbol', 'quantity', 'purchase_price', 'purchase_date', 'notes', 'created_at'])
        return df
    except psycopg2.Error as e:
        print(f"Error fetching portfolio: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def save_benchmark_data(name, symbol, timestamp, value):
    """Save benchmark data point"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        query = """
        INSERT INTO benchmarks 
        (name, symbol, timestamp, value)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (name, timestamp) 
        DO UPDATE SET
            value = EXCLUDED.value,
            created_at = CURRENT_TIMESTAMP
        RETURNING id
        """
        
        cur.execute(query, (name, symbol, timestamp, value))
        benchmark_id = cur.fetchone()[0]
        conn.commit()
        
        return benchmark_id
    except psycopg2.Error as e:
        print(f"Error saving benchmark data: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_benchmark_data(name, start_time, end_time):
    """Get benchmark data"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT name, symbol, timestamp, value
        FROM benchmarks
        WHERE name = %s
        AND timestamp BETWEEN %s AND %s
        ORDER BY timestamp ASC
        """
        
        # Convert timestamps to datetime objects if they're not already
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        # Use custom function instead of pandas read_sql_query to avoid SQLAlchemy dependency
        df = execute_sql_to_df(query, conn, params=(name, start_time, end_time))
        
        return df
    except psycopg2.Error as e:
        print(f"Error fetching benchmark data: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def save_sentiment_data(symbol, source, timestamp, sentiment_score, post_volume=0, sentiment_ratio=0.0, discussion_intensity=0.0):
    """
    Save sentiment data to database
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTC')
        source: Source of sentiment data (e.g., 'twitter', 'reddit')
        timestamp: Timestamp for the sentiment data
        sentiment_score: Sentiment score (0.0-1.0)
        post_volume: Volume of posts analyzed (optional)
        sentiment_ratio: Ratio of positive to negative sentiment (-1.0 to 1.0) (optional)
        discussion_intensity: Intensity of discussion (0.0-1.0) (optional)
    """
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        # Check if the sentiment_data table has the needed columns
        # If not, add them with ALTER TABLE
        try:
            cur.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'sentiment_data' AND column_name = 'sentiment_ratio'
            """)
            if not cur.fetchone():
                # Add sentiment_ratio column
                cur.execute("ALTER TABLE sentiment_data ADD COLUMN sentiment_ratio FLOAT DEFAULT 0.0")
            
            cur.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'sentiment_data' AND column_name = 'discussion_intensity'
            """)
            if not cur.fetchone():
                # Add discussion_intensity column
                cur.execute("ALTER TABLE sentiment_data ADD COLUMN discussion_intensity FLOAT DEFAULT 0.0")
                
            # Rename volume to post_volume if needed
            cur.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'sentiment_data' AND column_name = 'volume'
            """)
            if cur.fetchone():
                # Check if post_volume already exists
                cur.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'sentiment_data' AND column_name = 'post_volume'
                """)
                if not cur.fetchone():
                    # Rename volume to post_volume
                    cur.execute("ALTER TABLE sentiment_data RENAME COLUMN volume TO post_volume")
        except Exception as e:
            print(f"Error checking or altering sentiment_data table: {e}")
        
        # New query with updated columns
        query = """
        INSERT INTO sentiment_data 
        (symbol, source, timestamp, sentiment_score, post_volume, sentiment_ratio, discussion_intensity)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, source, timestamp) 
        DO UPDATE SET
            sentiment_score = EXCLUDED.sentiment_score,
            post_volume = EXCLUDED.post_volume,
            sentiment_ratio = EXCLUDED.sentiment_ratio,
            discussion_intensity = EXCLUDED.discussion_intensity,
            created_at = CURRENT_TIMESTAMP
        RETURNING id
        """
        
        cur.execute(query, (symbol, source, timestamp, sentiment_score, post_volume, sentiment_ratio, discussion_intensity))
        sentiment_id = cur.fetchone()[0]
        conn.commit()
        
        return sentiment_id
    except psycopg2.Error as e:
        print(f"Error saving sentiment data: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_sentiment_data(symbol, sources=None, start_time=None, end_time=None):
    """Get sentiment data from database"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        # Base query
        query = """
        SELECT symbol, source, timestamp, sentiment_score, 
               COALESCE(post_volume, 0) as post_volume,
               COALESCE(sentiment_ratio, 0) as sentiment_ratio, 
               COALESCE(discussion_intensity, 0) as discussion_intensity
        FROM sentiment_data
        WHERE symbol = %s
        """
        
        params = [symbol]
        
        # Add source filter if specified
        if sources:
            if isinstance(sources, str):
                sources = [sources]
            placeholders = ', '.join(['%s'] * len(sources))
            query += f" AND source IN ({placeholders})"
            params.extend(sources)
        
        # Add time range if specified
        if start_time:
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            query += " AND timestamp >= %s"
            params.append(start_time)
        
        if end_time:
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            query += " AND timestamp <= %s"
            params.append(end_time)
        
        # Add order by
        query += " ORDER BY timestamp ASC"
        
        # Use custom function instead of pandas read_sql_query to avoid SQLAlchemy dependency
        df = execute_sql_to_df(query, conn, params=tuple(params))
        
        return df
    except psycopg2.Error as e:
        print(f"Error fetching sentiment data: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def get_profitable_trades(symbol, interval, min_profit_pct=0, limit=50):
    """Get the most profitable closed trades"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT 
            id, symbol, interval, type, 
            entry_timestamp, exit_timestamp, 
            entry_price, exit_price, quantity,
            profit, profit_pct, holding_time_hours
        FROM trades
        WHERE symbol = %s
        AND interval = %s
        AND trade_status = 'CLOSED'
        AND profit_pct >= %s
        ORDER BY profit_pct DESC
        LIMIT %s
        """
        
        # Use custom function instead of pandas read_sql_query to avoid SQLAlchemy dependency
        df = execute_sql_to_df(query, conn, params=(symbol, interval, min_profit_pct, limit))
        
        return df
    except psycopg2.Error as e:
        print(f"Error fetching profitable trades: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()
def save_dxy_data(df):
    """
    Save US Dollar Index (DXY) data to database
    
    Args:
        df: DataFrame with DXY data (must have timestamp and close columns at minimum)
        
    Returns:
        Boolean indicating success
    """
    if df.empty:
        return False
    
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        # Insert data row by row
        insert_query = """
        INSERT INTO dollar_index 
        (timestamp, close, open, high, low, volume)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (timestamp) 
        DO UPDATE SET
            close = EXCLUDED.close,
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            volume = EXCLUDED.volume,
            created_at = CURRENT_TIMESTAMP
        """
        
        data_tuples = []
        for _, row in df.iterrows():
            # Get values with appropriate null handling
            open_val = float(row['open']) if 'open' in row and not pd.isna(row['open']) else None
            high_val = float(row['high']) if 'high' in row and not pd.isna(row['high']) else None
            low_val = float(row['low']) if 'low' in row and not pd.isna(row['low']) else None
            volume_val = float(row['volume']) if 'volume' in row and not pd.isna(row['volume']) else None
            
            data_tuples.append((
                row['timestamp'],
                float(row['close']),
                open_val,
                high_val,
                low_val,
                volume_val
            ))
        
        cur.executemany(insert_query, data_tuples)
        conn.commit()
        
        return True
    except psycopg2.Error as e:
        print(f"Error saving DXY data: {e}")
        return False
    finally:
        if conn:
            conn.close()


def get_dxy_data(start_time=None, end_time=None):
    """
    Get US Dollar Index data from database
    
    Args:
        start_time: Start time for data (datetime or string YYYY-MM-DD)
        end_time: End time for data (datetime or string YYYY-MM-DD)
        
    Returns:
        DataFrame with DXY data
    """
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        # Build query based on parameters
        query = "SELECT timestamp, close, open, high, low, volume FROM dollar_index "
        params = []
        
        if start_time or end_time:
            query += "WHERE "
            
            if start_time:
                # Convert string to datetime if needed
                if isinstance(start_time, str):
                    start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                query += "timestamp >= %s "
                params.append(start_time)
                
                if end_time:
                    query += "AND "
            
            if end_time:
                # Convert string to datetime if needed
                if isinstance(end_time, str):
                    end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                query += "timestamp <= %s "
                params.append(end_time)
        
        query += "ORDER BY timestamp ASC"
        
        # Use custom function to execute query
        df = execute_sql_to_df(query, conn, params=tuple(params) if params else None)
        
        return df
    except psycopg2.Error as e:
        print(f"Error fetching DXY data: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()


def save_liquidity_data(df):
    """
    Save global liquidity indicator data to database
    
    Args:
        df: DataFrame with global liquidity data 
            (must have indicator_name, timestamp, and value columns)
        
    Returns:
        Boolean indicating success
    """
    if df.empty:
        return False
    
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        # Insert data row by row
        insert_query = """
        INSERT INTO global_liquidity 
        (indicator_name, timestamp, value)
        VALUES (%s, %s, %s)
        ON CONFLICT (indicator_name, timestamp) 
        DO UPDATE SET
            value = EXCLUDED.value,
            created_at = CURRENT_TIMESTAMP
        """
        
        data_tuples = []
        for _, row in df.iterrows():
            data_tuples.append((
                row['indicator_name'],
                row['timestamp'],
                float(row['value'])
            ))
        
        cur.executemany(insert_query, data_tuples)
        conn.commit()
        
        return True
    except psycopg2.Error as e:
        print(f"Error saving liquidity data: {e}")
        return False
    finally:
        if conn:
            conn.close()


def get_liquidity_data(indicator_name=None, start_time=None, end_time=None):
    """
    Get global liquidity data from database
    
    Args:
        indicator_name: Specific indicator name to filter by (or None for all)
        start_time: Start time for data (datetime or string YYYY-MM-DD)
        end_time: End time for data (datetime or string YYYY-MM-DD)
        
    Returns:
        DataFrame with liquidity data
    """
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        # Build query based on parameters
        query = "SELECT indicator_name, timestamp, value FROM global_liquidity "
        params = []
        where_clauses = []
        
        if indicator_name:
            where_clauses.append("indicator_name = %s")
            params.append(indicator_name)
        
        if start_time:
            # Convert string to datetime if needed
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            where_clauses.append("timestamp >= %s")
            params.append(start_time)
        
        if end_time:
            # Convert string to datetime if needed
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            where_clauses.append("timestamp <= %s")
            params.append(end_time)
        
        if where_clauses:
            query += "WHERE " + " AND ".join(where_clauses)
        
        query += " ORDER BY indicator_name, timestamp ASC"
        
        # Use custom function to execute query
        df = execute_sql_to_df(query, conn, params=tuple(params) if params else None)
        
        return df
    except psycopg2.Error as e:
        print(f"Error fetching liquidity data: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()
