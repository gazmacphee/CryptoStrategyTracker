"""
Module for handling trading signals - creation, storage, and retrieval
"""

import os
import psycopg2
import pandas as pd
import numpy as np
import json
from datetime import datetime
from database import get_db_connection

# Constants for risk management
DEFAULT_STOP_LOSS_PERCENT = 0.03  # 3% default stop loss
DEFAULT_RISK_REWARD_RATIO = 2.0   # 2:1
DEFAULT_ATR_MULTIPLIER = 2.0      # 2 * ATR for stop loss


def calculate_support_level(df, lookback_periods=20):
    """
    Find support level based on recent lows
    
    Args:
        df: DataFrame with OHLCV data
        lookback_periods: Number of periods to look back to find support
        
    Returns:
        Support price level
    """
    if len(df) < lookback_periods + 1:
        return None
    
    # Get recent data for support calculation
    recent_data = df.iloc[-lookback_periods:]
    
    # Find local minimums (lows surrounded by higher lows)
    lows = []
    for i in range(1, len(recent_data)-1):
        if recent_data.iloc[i]['low'] < recent_data.iloc[i-1]['low'] and \
           recent_data.iloc[i]['low'] < recent_data.iloc[i+1]['low']:
            lows.append(recent_data.iloc[i]['low'])
    
    # If no clear support, use lowest low
    if not lows:
        return recent_data['low'].min()
    
    # Return the most recent support level or average of supports
    return lows[-1] if len(lows) > 0 else None


def calculate_resistance_level(df, lookback_periods=20):
    """
    Find resistance level based on recent highs
    
    Args:
        df: DataFrame with OHLCV data
        lookback_periods: Number of periods to look back to find resistance
        
    Returns:
        Resistance price level
    """
    if len(df) < lookback_periods + 1:
        return None
    
    # Get recent data for resistance calculation
    recent_data = df.iloc[-lookback_periods:]
    
    # Find local maximums (highs surrounded by lower highs)
    highs = []
    for i in range(1, len(recent_data)-1):
        if recent_data.iloc[i]['high'] > recent_data.iloc[i-1]['high'] and \
           recent_data.iloc[i]['high'] > recent_data.iloc[i+1]['high']:
            highs.append(recent_data.iloc[i]['high'])
    
    # If no clear resistance, use highest high
    if not highs:
        return recent_data['high'].max()
    
    # Return the most recent resistance level or average of resistances
    return highs[-1] if len(highs) > 0 else None


def calculate_atr(df, period=14):
    """
    Calculate Average True Range (ATR)
    
    Args:
        df: DataFrame with OHLCV data
        period: ATR period
        
    Returns:
        ATR value
    """
    if len(df) < period + 1:
        return None
    
    # Calculate True Range
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    # Calculate ATR
    atr = true_range.rolling(period).mean().iloc[-1]
    return atr


def get_stop_loss(df, entry_price, method='support', percent=DEFAULT_STOP_LOSS_PERCENT, 
                 atr_multiplier=DEFAULT_ATR_MULTIPLIER):
    """
    Calculate stop loss level using various methods
    
    Args:
        df: DataFrame with OHLCV data
        entry_price: Entry price for the trade
        method: Method to use for stop loss - 'support', 'percent', or 'atr'
        percent: Percentage below entry for stop loss (if using 'percent' method)
        atr_multiplier: Multiplier for ATR value (if using 'atr' method)
        
    Returns:
        Tuple of (stop_loss_price, method_used)
    """
    # Default fallback
    percent_stop = entry_price * (1 - percent)
    
    if method == 'support':
        # Try to find support level
        support = calculate_support_level(df)
        
        # If support is too far (more than 5% loss), use percent method instead
        if support and support < entry_price and support > entry_price * 0.95:
            return support, 'support'
        else:
            # Fall back to percent method
            return percent_stop, 'percent'
    
    elif method == 'atr':
        atr = calculate_atr(df)
        if atr:
            atr_stop = entry_price - (atr * atr_multiplier)
            
            # If ATR stop loss is too far (more than 5% loss), use percent method
            if atr_stop < entry_price * 0.95:
                return percent_stop, 'percent'
            else:
                return atr_stop, 'atr'
        else:
            # Fall back to percent method
            return percent_stop, 'percent'
    
    # Default percent method
    return percent_stop, 'percent'


def get_take_profit(entry_price, stop_loss, method='risk_reward', risk_reward_ratio=DEFAULT_RISK_REWARD_RATIO, 
                   resistance_level=None):
    """
    Calculate take profit level using various methods
    
    Args:
        entry_price: Entry price for the trade
        stop_loss: Stop loss level
        method: Method to use for take profit - 'risk_reward' or 'resistance'
        risk_reward_ratio: Risk-to-reward ratio (if using 'risk_reward' method)
        resistance_level: Resistance level (if using 'resistance' method)
        
    Returns:
        Tuple of (take_profit_price, method_used)
    """
    # Default fallback using risk-reward ratio
    risk = entry_price - stop_loss
    rr_take_profit = entry_price + (risk * risk_reward_ratio)
    
    if method == 'resistance' and resistance_level:
        # If resistance is higher than entry and provides at least 1:1 risk-reward
        if resistance_level > entry_price and resistance_level >= (entry_price + risk):
            return resistance_level, 'resistance'
        else:
            # Fall back to risk-reward method
            return rr_take_profit, 'risk_reward'
    
    # Default risk-reward method
    return rr_take_profit, 'risk_reward'


def calculate_exit_levels(df, entry_price, stop_loss_method='support', 
                         take_profit_method='risk_reward', risk_reward_ratio=DEFAULT_RISK_REWARD_RATIO):
    """
    Calculate both stop loss and take profit levels
    
    Args:
        df: DataFrame with OHLCV data
        entry_price: Entry price for the trade
        stop_loss_method: Method to use for stop loss calculation
        take_profit_method: Method to use for take profit calculation
        risk_reward_ratio: Risk-to-reward ratio for take profit calculation
        
    Returns:
        Dictionary with stop loss and take profit information
    """
    # Calculate stop loss
    stop_loss, sl_method = get_stop_loss(df, entry_price, method=stop_loss_method)
    
    # Calculate resistance level if needed
    resistance = None
    if take_profit_method == 'resistance':
        resistance = calculate_resistance_level(df)
    
    # Calculate take profit
    take_profit, tp_method = get_take_profit(entry_price, stop_loss, 
                                            method=take_profit_method, 
                                            risk_reward_ratio=risk_reward_ratio,
                                            resistance_level=resistance)
    
    # Calculate actual risk-reward ratio achieved
    risk = entry_price - stop_loss
    reward = take_profit - entry_price
    actual_rr = reward / risk if risk > 0 else 0
    
    return {
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'stop_loss_method': sl_method,
        'take_profit_method': tp_method,
        'risk_reward_ratio': actual_rr
    }

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

def get_last_signal(conn, symbol, interval, strategy_name="default_strategy"):
    """
    Get the most recent signal for a symbol and interval
    
    Returns:
        Tuple of (timestamp, signal_type) or (None, None) if no signals exist
    """
    try:
        cur = conn.cursor()
        query = """
        SELECT timestamp, signal_type 
        FROM trading_signals 
        WHERE symbol = %s AND interval = %s AND strategy_name = %s
        ORDER BY timestamp DESC 
        LIMIT 1
        """
        cur.execute(query, (symbol, interval, strategy_name))
        result = cur.fetchone()
        
        if result:
            return result[0], result[1]
        return None, None
    except Exception as e:
        print(f"Error getting last signal: {e}")
        return None, None

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
        
        # Get the last signal for this symbol/interval
        last_timestamp, last_signal_type = get_last_signal(conn, symbol, interval, strategy_name)
        
        # Insert signals where buy_signal or sell_signal is True
        insert_query = """
        INSERT INTO trading_signals 
        (symbol, interval, timestamp, signal_type, signal_strength, price, 
         stop_loss, take_profit, stop_loss_method, take_profit_method, risk_reward_ratio,
         bb_signal, rsi_signal, macd_signal, ema_signal, strategy_name, strategy_params, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, interval, timestamp, strategy_name) 
        DO UPDATE SET
            signal_type = EXCLUDED.signal_type,
            signal_strength = EXCLUDED.signal_strength,
            price = EXCLUDED.price,
            stop_loss = EXCLUDED.stop_loss,
            take_profit = EXCLUDED.take_profit,
            stop_loss_method = EXCLUDED.stop_loss_method,
            take_profit_method = EXCLUDED.take_profit_method,
            risk_reward_ratio = EXCLUDED.risk_reward_ratio,
            bb_signal = EXCLUDED.bb_signal,
            rsi_signal = EXCLUDED.rsi_signal,
            macd_signal = EXCLUDED.macd_signal,
            ema_signal = EXCLUDED.ema_signal,
            strategy_params = EXCLUDED.strategy_params,
            notes = EXCLUDED.notes,
            created_at = CURRENT_TIMESTAMP
        """
        
        signal_count = 0
        in_position = last_signal_type == 'buy' if last_signal_type else False
        
        for idx, row in df.iterrows():
            # Skip rows earlier than our last saved signal
            if last_timestamp and row['timestamp'] <= last_timestamp:
                continue
                
            # Only save rows with signals that follow the correct sequence
            has_buy_signal = row.get('buy_signal', False)
            has_sell_signal = row.get('sell_signal', False)
            
            # Only process buy signals if we're not in a position
            # Only process sell signals if we are in a position
            if (has_buy_signal and not in_position) or (has_sell_signal and in_position):
                signal_type = 'buy' if has_buy_signal else 'sell'
                
                # Calculate signal strength - default to 1.0 if not available
                signal_strength = 1.0
                
                # Determine which indicators contributed to the signal
                bb_signal = 'bb_percent' in row and (
                    (signal_type == 'buy' and row['bb_percent'] < -0.75) or 
                    (signal_type == 'sell' and row['bb_percent'] > 0.75)
                )
                
                rsi_signal = 'rsi' in row and (
                    (signal_type == 'buy' and row['rsi'] < 35) or 
                    (signal_type == 'sell' and row['rsi'] > 65)
                )
                
                macd_signal = 'macd_histogram' in row and (
                    (signal_type == 'buy' and row['macd_histogram'] > 0 and row.get('macd_histogram_prev', 0) <= 0) or
                    (signal_type == 'sell' and row['macd_histogram'] < 0 and row.get('macd_histogram_prev', 0) >= 0)
                )
                
                ema_signal = 'ema_9' in row and 'ema_21' in row and (
                    (signal_type == 'buy' and row['ema_9'] > row['ema_21'] and row.get('ema_9_prev', 0) <= row.get('ema_21_prev', 0)) or
                    (signal_type == 'sell' and row['ema_9'] < row['ema_21'] and row.get('ema_9_prev', 0) >= row.get('ema_21_prev', 0))
                )
                
                # Get strategy used if available in the row
                if 'strategy_used' in row:
                    current_strategy_name = f"{strategy_name}_{row['strategy_used']}"
                else:
                    current_strategy_name = strategy_name
                
                # Convert strategy params to JSON
                strategy_params_json = json.dumps(strategy_params) if strategy_params else None
                
                # Create notes
                notes = f"{signal_type.upper()} signal at {row['close']} based on "
                signal_reasons = []
                if bb_signal: signal_reasons.append("Bollinger Bands")
                if rsi_signal: signal_reasons.append("RSI")
                if macd_signal: signal_reasons.append("MACD")
                if ema_signal: signal_reasons.append("EMA Crossover")
                
                # Add strategy information to notes
                if 'strategy_used' in row:
                    signal_reasons.append(f"Strategy: {row['strategy_used']}")
                    
                notes += ", ".join(signal_reasons) if signal_reasons else "combined indicators"
                
                # Calculate stop loss and take profit values for buy signals
                stop_loss = None
                take_profit = None
                stop_loss_method = None
                take_profit_method = None
                risk_reward_ratio = None
                
                if signal_type == 'buy':
                    # Get enough historical data for exit level calculations
                    recent_rows = df.iloc[max(0, idx-30):idx+1].copy()
                    
                    # Calculate entry price (current closing price)
                    entry_price = float(row['close'])
                    
                    # Calculate exit levels using support for stop loss and resistance for take profit
                    exit_levels = calculate_exit_levels(
                        recent_rows, 
                        entry_price, 
                        stop_loss_method='support',  # Try support level first, with fallback
                        take_profit_method='resistance',  # Try resistance level first, with fallback
                        risk_reward_ratio=DEFAULT_RISK_REWARD_RATIO
                    )
                    
                    # Extract values
                    stop_loss = exit_levels['stop_loss']
                    take_profit = exit_levels['take_profit']
                    stop_loss_method = exit_levels['stop_loss_method']
                    take_profit_method = exit_levels['take_profit_method']
                    risk_reward_ratio = exit_levels['risk_reward_ratio']
                    
                    # Add stop loss and take profit info to notes
                    notes += f". Stop loss: {stop_loss:.2f} ({stop_loss_method}), "
                    notes += f"Take profit: {take_profit:.2f} ({take_profit_method}), "
                    notes += f"Risk-reward: 1:{risk_reward_ratio:.2f}"
                
                # Insert the signal
                cur.execute(insert_query, (
                    symbol,
                    interval,
                    row['timestamp'],
                    signal_type,
                    signal_strength,
                    float(row['close']),
                    stop_loss,
                    take_profit,
                    stop_loss_method,
                    take_profit_method,
                    risk_reward_ratio,
                    bb_signal,
                    rsi_signal,
                    macd_signal,
                    ema_signal,
                    current_strategy_name,
                    strategy_params_json,
                    notes
                ))
                
                # Update our in_position status after saving this signal
                in_position = (signal_type == 'buy')
                
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

def save_trading_signals_from_multiple_strategies(dataframes, symbol, interval):
    """
    Save trading signals from multiple strategies to the database
    
    Args:
        dataframes: Dictionary of DataFrames, each with a different strategy applied
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '4h', '1d')
    
    Returns:
        Total number of signals saved
    """
    total_signals = 0
    
    # Process each strategy's dataframe
    for strategy_name, df in dataframes.items():
        if not df.empty and ('buy_signal' in df.columns or 'sell_signal' in df.columns):
            # Save signals from this strategy
            if save_trading_signals(df, symbol, interval, strategy_name):
                # Count approximate number of signals (rough estimate)
                signal_count = df['buy_signal'].sum() + df['sell_signal'].sum()
                total_signals += int(signal_count)
    
    return total_signals

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
               stop_loss, take_profit, stop_loss_method, take_profit_method, risk_reward_ratio,
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