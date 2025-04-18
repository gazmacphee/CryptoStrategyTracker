import os
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
from psycopg2 import sql

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
        cursor = conn.cursor()
        query = """
        SELECT DISTINCT 
            EXTRACT(YEAR FROM timestamp)::integer as year,
            EXTRACT(MONTH FROM timestamp)::integer as month
        FROM historical_data
        WHERE symbol = %s
        AND interval = %s
        AND timestamp BETWEEN %s AND %s
        ORDER BY year, month
        """
        
        # Create start and end dates from year and month
        start_date = datetime(start_year, start_month, 1)
        # Last day of end month
        if end_month == 12:
            end_date = datetime(end_year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(end_year, end_month + 1, 1) - timedelta(days=1)
        
        cursor.execute(query, (symbol, interval, start_date, end_date))
        results = cursor.fetchall()
        
        # Convert results to list of (year, month) tuples
        existing_months = [(int(year), int(month)) for year, month in results]
        
        return existing_months
    except Exception as e:
        print(f"Error checking existing data months: {e}")
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
        cursor = conn.cursor()
        query = """
        SELECT COUNT(*) 
        FROM historical_data
        WHERE symbol = %s
        AND interval = %s
        AND EXTRACT(YEAR FROM timestamp)::integer = %s
        AND EXTRACT(MONTH FROM timestamp)::integer = %s
        """
        
        cursor.execute(query, (symbol, interval, year, month))
        count = cursor.fetchone()[0]
        
        # Estimate expected candles for the month based on interval
        from calendar import monthrange
        days_in_month = monthrange(year, month)[1]
        
        expected_count = 0
        if interval == '1h':
            expected_count = days_in_month * 24
        elif interval == '4h':
            expected_count = days_in_month * 6
        elif interval == '1d':
            expected_count = days_in_month
        
        # Consider complete if at least 90% of expected candles are present
        # (allows for some missing candles due to exchange maintenance, etc.)
        completeness_threshold = 0.90
        
        return count >= expected_count * completeness_threshold
    except Exception as e:
        print(f"Error checking month data completeness: {e}")
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
        
        # Insert or update indicators
        insert_query = """
        INSERT INTO technical_indicators 
        (symbol, interval, timestamp, rsi, macd, macd_signal, macd_histogram, 
         bb_upper, bb_middle, bb_lower, bb_percent, ema_9, ema_21, ema_50, ema_200,
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
                float(row.get('rsi', None)) if 'rsi' in row and pd.notna(row['rsi']) else None,
                float(row.get('macd', None)) if 'macd' in row and pd.notna(row['macd']) else None,
                float(row.get('macd_signal', None)) if 'macd_signal' in row and pd.notna(row['macd_signal']) else None,
                float(row.get('macd_histogram', None)) if 'macd_histogram' in row and pd.notna(row['macd_histogram']) else None,
                float(row.get('bb_upper', None)) if 'bb_upper' in row and pd.notna(row['bb_upper']) else None,
                float(row.get('bb_middle', None)) if 'bb_middle' in row and pd.notna(row['bb_middle']) else None,
                float(row.get('bb_lower', None)) if 'bb_lower' in row and pd.notna(row['bb_lower']) else None,
                float(row.get('bb_percent', None)) if 'bb_percent' in row and pd.notna(row['bb_percent']) else None,
                float(row.get('ema_9', None)) if 'ema_9' in row and pd.notna(row['ema_9']) else None,
                float(row.get('ema_21', None)) if 'ema_21' in row and pd.notna(row['ema_21']) else None,
                float(row.get('ema_50', None)) if 'ema_50' in row and pd.notna(row['ema_50']) else None,
                float(row.get('ema_200', None)) if 'ema_200' in row and pd.notna(row['ema_200']) else None,
                bool(row.get('buy_signal', False)),
                bool(row.get('sell_signal', False))
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
        
        df = pd.read_sql_query(
            query, 
            conn, 
            params=(symbol, interval, start_time, end_time)
        )
        
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
        
        # Convert strategy_params to JSON string if present
        strategy_params = trade_data.get('strategy_params', {})
        import json
        strategy_params_json = json.dumps(strategy_params)
        
        if trade_data.get('type') == 'BUY':
            # Insert a new open trade
            insert_query = """
            INSERT INTO trades 
            (symbol, interval, type, entry_timestamp, entry_price, quantity, trade_status, strategy_params)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """
            
            cur.execute(insert_query, (
                trade_data.get('symbol'),
                trade_data.get('interval'),
                trade_data.get('type'),
                trade_data.get('timestamp'),
                trade_data.get('price'),
                trade_data.get('coins'),
                'OPEN',
                strategy_params_json
            ))
            
            # Get the ID of the inserted trade
            trade_id = cur.fetchone()[0]
            conn.commit()
            return trade_id
            
        elif trade_data.get('type') in ['SELL', 'SELL (EOT)']:
            # Update an existing trade
            update_query = """
            UPDATE trades 
            SET exit_timestamp = %s,
                exit_price = %s,
                profit = %s,
                profit_pct = %s,
                holding_time_hours = %s,
                trade_status = 'CLOSED'
            WHERE id = %s
            """
            
            cur.execute(update_query, (
                trade_data.get('timestamp'),
                trade_data.get('price'),
                trade_data.get('profit'),
                trade_data.get('profit_pct'),
                trade_data.get('holding_time'),
                trade_data.get('trade_id')
            ))
            
            conn.commit()
            return True
            
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
        SELECT id, symbol, interval, type, entry_timestamp, entry_price, quantity, strategy_params
        FROM trades
        WHERE symbol = %s
        AND interval = %s
        AND trade_status = 'OPEN'
        ORDER BY entry_timestamp DESC
        LIMIT 1
        """
        
        cur = conn.cursor()
        cur.execute(query, (symbol, interval))
        
        result = cur.fetchone()
        if result:
            # Convert to dictionary
            columns = ['id', 'symbol', 'interval', 'type', 'entry_timestamp', 'entry_price', 'quantity', 'strategy_params']
            trade = dict(zip(columns, result))
            
            # Parse JSON
            import json
            if trade['strategy_params']:
                trade['strategy_params'] = json.loads(trade['strategy_params'])
                
            return trade
        
        return None
    except psycopg2.Error as e:
        print(f"Error fetching open trade: {e}")
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
        
        # Insert a new portfolio item
        insert_query = """
        INSERT INTO portfolio 
        (symbol, quantity, purchase_price, purchase_date, notes)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """
        
        # Convert purchase_date to datetime if it's a string
        if isinstance(purchase_date, str):
            purchase_date = datetime.fromisoformat(purchase_date.replace('Z', '+00:00'))
            
        cur.execute(insert_query, (
            symbol,
            quantity,
            purchase_price,
            purchase_date,
            notes
        ))
        
        # Get the ID of the inserted item
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
        
        # First, get the current item to preserve any values not being updated
        cur.execute("SELECT * FROM portfolio WHERE id = %s", (item_id,))
        current_item = cur.fetchone()
        
        if not current_item:
            print(f"Portfolio item with ID {item_id} not found")
            return False
            
        # Build update query based on which fields are provided
        update_parts = []
        params = []
        
        if quantity is not None:
            update_parts.append("quantity = %s")
            params.append(quantity)
            
        if purchase_price is not None:
            update_parts.append("purchase_price = %s")
            params.append(purchase_price)
            
        if purchase_date is not None:
            # Convert purchase_date to datetime if it's a string
            if isinstance(purchase_date, str):
                purchase_date = datetime.fromisoformat(purchase_date.replace('Z', '+00:00'))
            update_parts.append("purchase_date = %s")
            params.append(purchase_date)
            
        if notes is not None:
            update_parts.append("notes = %s")
            params.append(notes)
            
        if not update_parts:
            # Nothing to update
            return True
            
        # Complete the query and add the ID parameter
        update_query = f"UPDATE portfolio SET {', '.join(update_parts)} WHERE id = %s"
        params.append(item_id)
        
        cur.execute(update_query, params)
        conn.commit()
        return True
            
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
        cur.execute("DELETE FROM portfolio WHERE id = %s", (item_id,))
        conn.commit()
        return True
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
        return pd.DataFrame(columns=['id', 'symbol', 'quantity', 'purchase_price', 'purchase_date', 'notes', 'created_at'])
    
    try:
        query = """
        SELECT id, symbol, quantity, purchase_price, purchase_date, notes, created_at
        FROM portfolio
        ORDER BY symbol, purchase_date
        """
        
        df = pd.read_sql_query(query, conn)
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
        
        # Insert or update benchmark data point
        insert_query = """
        INSERT INTO benchmarks 
        (name, symbol, timestamp, value)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (name, timestamp) 
        DO UPDATE SET
            value = EXCLUDED.value,
            created_at = CURRENT_TIMESTAMP
        """
        
        # Convert timestamp to datetime if it's a string
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
        cur.execute(insert_query, (
            name,
            symbol,
            timestamp,
            value
        ))
        
        conn.commit()
        return True
            
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
        SELECT timestamp, value
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
        
        df = pd.read_sql_query(
            query, 
            conn, 
            params=(name, start_time, end_time)
        )
        
        return df
    except psycopg2.Error as e:
        print(f"Error fetching benchmark data: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def save_sentiment_data(symbol, source, timestamp, sentiment_score, volume):
    """Save sentiment data to database"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        # Insert or update sentiment data
        insert_query = """
        INSERT INTO sentiment_data 
        (symbol, source, timestamp, sentiment_score, volume)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (symbol, source, timestamp) 
        DO UPDATE SET
            sentiment_score = EXCLUDED.sentiment_score,
            volume = EXCLUDED.volume,
            created_at = CURRENT_TIMESTAMP
        """
        
        # Convert timestamp to datetime if it's a string
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
        cur.execute(insert_query, (
            symbol,
            source,
            timestamp,
            sentiment_score,
            volume
        ))
        
        conn.commit()
        return True
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
        # Build query based on parameters
        query = """
        SELECT symbol, source, timestamp, sentiment_score, volume
        FROM sentiment_data
        WHERE symbol = %s
        """
        
        params = [symbol]
        
        # Add source filter if specified
        if sources:
            source_placeholders = ', '.join(['%s'] * len(sources))
            query += f" AND source IN ({source_placeholders})"
            params.extend(sources)
        
        # Add time filters if specified
        if start_time:
            # Convert to datetime if it's a string
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            query += " AND timestamp >= %s"
            params.append(start_time)
            
        if end_time:
            # Convert to datetime if it's a string
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            query += " AND timestamp <= %s"
            params.append(end_time)
        
        # Order by timestamp
        query += " ORDER BY timestamp ASC"
        
        # Execute query
        df = pd.read_sql_query(query, conn, params=params)
        
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
        return []
    
    try:
        query = """
        SELECT id, symbol, interval, type, 
               entry_timestamp, exit_timestamp, 
               entry_price, exit_price, 
               quantity, profit, profit_pct, holding_time_hours, strategy_params
        FROM trades
        WHERE symbol = %s
        AND interval = %s
        AND trade_status = 'CLOSED'
        AND profit_pct >= %s
        ORDER BY profit_pct DESC
        LIMIT %s
        """
        
        cur = conn.cursor()
        cur.execute(query, (symbol, interval, min_profit_pct, limit))
        
        rows = cur.fetchall()
        trades = []
        
        if rows:
            # Convert to dictionaries
            columns = ['id', 'symbol', 'interval', 'type', 
                     'entry_timestamp', 'exit_timestamp', 
                     'entry_price', 'exit_price', 
                     'quantity', 'profit', 'profit_pct', 'holding_time_hours', 'strategy_params']
            
            import json
            for row in rows:
                trade = dict(zip(columns, row))
                
                # Parse JSON
                if trade['strategy_params']:
                    trade['strategy_params'] = json.loads(trade['strategy_params'])
                    
                trades.append(trade)
        
        return trades
    except psycopg2.Error as e:
        print(f"Error fetching profitable trades: {e}")
        return []
    finally:
        if conn:
            conn.close()
