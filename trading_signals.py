"""
Module for handling trading signals - creation, storage, and retrieval
"""

import os
import psycopg2
import pandas as pd
import json
from datetime import datetime
from database import get_db_connection

def create_signals_table():
    """Create the trading signals table if it doesn't exist"""
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database, cannot create trading signals table")
        return False
    
    try:
        cur = conn.cursor()
        
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
        
        # Create index for faster queries
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol_interval_timestamp 
        ON trading_signals(symbol, interval, timestamp);
        """)
        
        conn.commit()
        print("Trading signals table created successfully")
        return True
    except Exception as e:
        print(f"Error creating trading signals table: {e}")
        return False
    finally:
        if conn:
            conn.close()

def save_trading_signals(df, symbol, interval, strategy_name="default_strategy", strategy_params=None):
    """
    Save trading signals to database
    
    Args:
        df: DataFrame with OHLCV data, indicators, and signal columns
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '4h', '1d')
        strategy_name: Name of the strategy used
        strategy_params: Dictionary of strategy parameters
    
    Returns:
        Boolean indicating success
    """
    if df.empty:
        return False
    
    # Ensure trading signals table exists
    create_signals_table()
    
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        # Insert signals where buy_signal or sell_signal is True
        insert_query = """
        INSERT INTO trading_signals 
        (symbol, interval, timestamp, signal_type, signal_strength, price, 
         bb_signal, rsi_signal, macd_signal, ema_signal, strategy_name, strategy_params, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, interval, timestamp, strategy_name) 
        DO UPDATE SET
            signal_type = EXCLUDED.signal_type,
            signal_strength = EXCLUDED.signal_strength,
            price = EXCLUDED.price,
            bb_signal = EXCLUDED.bb_signal,
            rsi_signal = EXCLUDED.rsi_signal,
            macd_signal = EXCLUDED.macd_signal,
            ema_signal = EXCLUDED.ema_signal,
            strategy_params = EXCLUDED.strategy_params,
            notes = EXCLUDED.notes,
            created_at = CURRENT_TIMESTAMP
        """
        
        signal_count = 0
        
        for idx, row in df.iterrows():
            # Only save rows with signals
            if row.get('buy_signal', False) or row.get('sell_signal', False):
                signal_type = 'buy' if row.get('buy_signal', False) else 'sell'
                
                # Calculate signal strength - default to 1.0 if not available
                signal_strength = 1.0
                
                # Determine which indicators contributed to the signal
                bb_signal = 'bb_percent' in row and (
                    (signal_type == 'buy' and row['bb_percent'] < -0.8) or 
                    (signal_type == 'sell' and row['bb_percent'] > 0.8)
                )
                
                rsi_signal = 'rsi' in row and (
                    (signal_type == 'buy' and row['rsi'] < 30) or 
                    (signal_type == 'sell' and row['rsi'] > 70)
                )
                
                macd_signal = 'macd_histogram' in row and (
                    (signal_type == 'buy' and row['macd_histogram'] > 0 and row.get('macd_histogram_prev', 0) <= 0) or
                    (signal_type == 'sell' and row['macd_histogram'] < 0 and row.get('macd_histogram_prev', 0) >= 0)
                )
                
                ema_signal = 'ema_9' in row and 'ema_21' in row and (
                    (signal_type == 'buy' and row['ema_9'] > row['ema_21'] and row.get('ema_9_prev', 0) <= row.get('ema_21_prev', 0)) or
                    (signal_type == 'sell' and row['ema_9'] < row['ema_21'] and row.get('ema_9_prev', 0) >= row.get('ema_21_prev', 0))
                )
                
                # Convert strategy params to JSON
                strategy_params_json = json.dumps(strategy_params) if strategy_params else None
                
                # Create notes
                notes = f"{signal_type.upper()} signal at {row['close']} based on "
                signal_reasons = []
                if bb_signal: signal_reasons.append("Bollinger Bands")
                if rsi_signal: signal_reasons.append("RSI")
                if macd_signal: signal_reasons.append("MACD")
                if ema_signal: signal_reasons.append("EMA Crossover")
                notes += ", ".join(signal_reasons) if signal_reasons else "combined indicators"
                
                # Insert the signal
                cur.execute(insert_query, (
                    symbol,
                    interval,
                    row['timestamp'],
                    signal_type,
                    signal_strength,
                    float(row['close']),
                    bb_signal,
                    rsi_signal,
                    macd_signal,
                    ema_signal,
                    strategy_name,
                    strategy_params_json,
                    notes
                ))
                
                signal_count += 1
        
        conn.commit()
        print(f"Saved {signal_count} trading signals for {symbol}/{interval}")
        return True
    except Exception as e:
        print(f"Error saving trading signals: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def get_recent_signals(symbol=None, interval=None, start_time=None, limit=100):
    """
    Get recent trading signals from the database
    
    Args:
        symbol: Trading pair symbol (optional)
        interval: Time interval (optional)
        start_time: Starting timestamp (optional)
        limit: Maximum number of signals to return
    
    Returns:
        DataFrame with signals
    """
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        sql_conditions = []
        params = []
        
        if symbol:
            sql_conditions.append("symbol = %s")
            params.append(symbol)
        
        if interval:
            sql_conditions.append("interval = %s")
            params.append(interval)
            
        if start_time:
            sql_conditions.append("timestamp >= %s")
            params.append(start_time)
        
        # Construct the WHERE clause if conditions exist
        where_clause = ""
        if sql_conditions:
            where_clause = "WHERE " + " AND ".join(sql_conditions)
        
        # Build the query
        query = f"""
        SELECT id, symbol, interval, timestamp, signal_type, signal_strength, price, 
               bb_signal, rsi_signal, macd_signal, ema_signal, 
               strategy_name, strategy_params, notes, created_at
        FROM trading_signals
        {where_clause}
        ORDER BY timestamp DESC
        LIMIT {limit}
        """
        
        # Execute the query
        import pandas as pd
        df = pd.read_sql_query(query, conn, params=params)
        return df
    except Exception as e:
        print(f"Error getting trading signals: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def get_signal_statistics(symbol=None, interval=None, days=30):
    """
    Get statistics about trading signals
    
    Args:
        symbol: Trading pair symbol (optional)
        interval: Time interval (optional)
        days: Number of days to look back
    
    Returns:
        Dictionary with statistics
    """
    conn = get_db_connection()
    if not conn:
        return {}
    
    try:
        sql_conditions = []
        params = []
        
        if symbol:
            sql_conditions.append("symbol = %s")
            params.append(symbol)
        
        if interval:
            sql_conditions.append("interval = %s")
            params.append(interval)
        
        if days:
            sql_conditions.append("timestamp >= NOW() - INTERVAL '%s days'")
            params.append(days)
        
        # Construct the WHERE clause
        where_clause = ""
        if sql_conditions:
            where_clause = "WHERE " + " AND ".join(sql_conditions)
        
        # Count signals by type
        query = f"""
        SELECT signal_type, COUNT(*) as count
        FROM trading_signals
        {where_clause}
        GROUP BY signal_type
        """
        
        # Execute the query
        import pandas as pd
        df = pd.read_sql_query(query, conn, params=params)
        
        # Convert to dictionary
        stats = {"total": df['count'].sum() if not df.empty else 0}
        for _, row in df.iterrows():
            stats[row['signal_type']] = row['count']
        
        # Get distribution by indicator
        query = f"""
        SELECT 
            SUM(CASE WHEN bb_signal THEN 1 ELSE 0 END) as bb_count,
            SUM(CASE WHEN rsi_signal THEN 1 ELSE 0 END) as rsi_count,
            SUM(CASE WHEN macd_signal THEN 1 ELSE 0 END) as macd_count,
            SUM(CASE WHEN ema_signal THEN 1 ELSE 0 END) as ema_count,
            COUNT(*) as total
        FROM trading_signals
        {where_clause}
        """
        
        df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            total = df['total'].iloc[0]
            if total > 0:
                stats['indicators'] = {
                    'bollinger': df['bb_count'].iloc[0],
                    'rsi': df['rsi_count'].iloc[0],
                    'macd': df['macd_count'].iloc[0],
                    'ema': df['ema_count'].iloc[0]
                }
        
        return stats
    except Exception as e:
        print(f"Error getting signal statistics: {e}")
        return {}
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Create the table if running this module directly
    create_signals_table()