import os
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
from psycopg2 import sql

# Database configuration - get from environment variables with defaults
DATABASE_URL = os.getenv("DATABASE_URL", "")
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
            else:
                host_port = host_port_db
                DB_NAME = ""
                
            if ":" in host_port:
                DB_HOST, DB_PORT = host_port.split(":", 1)
            else:
                DB_HOST = host_port
                DB_PORT = "5432"
        else:
            # Fallback to environment variables
            DB_HOST = os.getenv("PGHOST", "localhost")
            DB_PORT = os.getenv("PGPORT", "5432")
            DB_NAME = os.getenv("PGDATABASE", "crypto")
            DB_USER = os.getenv("PGUSER", "postgres")
            DB_PASS = os.getenv("PGPASSWORD", "postgres")
    except Exception as e:
        print(f"Error parsing DATABASE_URL: {e}, using environment variables instead")
        DB_HOST = os.getenv("PGHOST", "localhost")
        DB_PORT = os.getenv("PGPORT", "5432")
        DB_NAME = os.getenv("PGDATABASE", "crypto")
        DB_USER = os.getenv("PGUSER", "postgres")
        DB_PASS = os.getenv("PGPASSWORD", "postgres")
else:
    # Use environment variables
    DB_HOST = os.getenv("PGHOST", "localhost")
    DB_PORT = os.getenv("PGPORT", "5432")
    DB_NAME = os.getenv("PGDATABASE", "crypto")
    DB_USER = os.getenv("PGUSER", "postgres")
    DB_PASS = os.getenv("PGPASSWORD", "postgres")

def get_db_connection():
    """Create a database connection"""
    try:
        # Try connecting using DATABASE_URL directly first if available
        if DATABASE_URL:
            conn = psycopg2.connect(DATABASE_URL)
            return conn
        else:
            # Fall back to individual parameters
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
        return []
    
    try:
        query = """
        SELECT id, symbol, quantity, purchase_price, purchase_date, notes, created_at
        FROM portfolio
        ORDER BY symbol, purchase_date
        """
        
        df = pd.read_sql_query(query, conn)
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
