"""
Database Access Module

Provides core database functionality, connection management, 
and table definitions with optimized indexes.
"""

import logging
import psycopg2
import psycopg2.extras
from psycopg2.extensions import connection
from typing import Optional, List, Dict, Any

from src.config.container import container


def get_db_connection() -> connection:
    """
    Get a database connection from the container
    
    Returns:
        Active database connection
    """
    return container.get("db_connection")


def create_tables() -> None:
    """
    Create database tables with optimized indexes
    """
    logger = container.get("logger")
    logger.info("Creating database tables if they don't exist...")
    
    create_statements = [
        # Historical price data with optimized indexes for querying
        """
        CREATE TABLE IF NOT EXISTS historical_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            interval VARCHAR(5) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            close NUMERIC NOT NULL,
            volume NUMERIC NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        
        # Index on symbol + interval + timestamp for rapid lookups
        """
        CREATE INDEX IF NOT EXISTS idx_historical_data_lookup 
        ON historical_data (symbol, interval, timestamp)
        """,
        
        # Index on timestamp alone for time-based queries
        """
        CREATE INDEX IF NOT EXISTS idx_historical_data_timestamp
        ON historical_data (timestamp)
        """,
        
        # Technical indicators with optimized indexes
        """
        CREATE TABLE IF NOT EXISTS technical_indicators (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            interval VARCHAR(5) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            
            -- Bollinger Bands
            bb_upper NUMERIC,
            bb_middle NUMERIC,
            bb_lower NUMERIC,
            
            -- RSI
            rsi NUMERIC,
            
            -- MACD
            macd NUMERIC,
            macd_signal NUMERIC,
            macd_histogram NUMERIC,
            
            -- EMAs
            ema_9 NUMERIC,
            ema_21 NUMERIC,
            ema_50 NUMERIC,
            ema_200 NUMERIC,
            
            -- Stochastic
            stoch_k NUMERIC,
            stoch_d NUMERIC,
            
            -- ATR
            atr NUMERIC,
            
            -- ADX
            adx NUMERIC,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        
        # Index on symbol + interval + timestamp for rapid lookups
        """
        CREATE INDEX IF NOT EXISTS idx_indicators_lookup
        ON technical_indicators (symbol, interval, timestamp)
        """,
        
        # Trades table
        """
        CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            interval VARCHAR(5) NOT NULL,
            strategy VARCHAR(50) NOT NULL,
            direction VARCHAR(10) NOT NULL,
            entry_time TIMESTAMP NOT NULL,
            entry_price NUMERIC NOT NULL,
            exit_time TIMESTAMP,
            exit_price NUMERIC,
            quantity NUMERIC NOT NULL,
            profit_loss NUMERIC,
            profit_loss_pct NUMERIC,
            status VARCHAR(20) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        
        # Portfolio table
        """
        CREATE TABLE IF NOT EXISTS portfolio (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            cash_value NUMERIC NOT NULL,
            asset_value NUMERIC NOT NULL,
            total_value NUMERIC NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        
        # Benchmarks table
        """
        CREATE TABLE IF NOT EXISTS benchmarks (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            price NUMERIC NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        
        # Sentiment data
        """
        CREATE TABLE IF NOT EXISTS sentiment_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            source VARCHAR(50) NOT NULL,
            sentiment_score NUMERIC NOT NULL,
            volume NUMERIC,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        
        # Index on sentiment data for rapid lookups
        """
        CREATE INDEX IF NOT EXISTS idx_sentiment_lookup
        ON sentiment_data (symbol, timestamp)
        """
    ]
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    for statement in create_statements:
        try:
            cursor.execute(statement)
        except Exception as e:
            logger.error(f"Error creating table: {e}")
            raise
    
    cursor.close()
    logger.info("Database tables created successfully")


def save_batch(table: str, columns: List[str], values: List[tuple]) -> int:
    """
    Efficiently save multiple records in a single query
    
    Args:
        table: Target table name
        columns: Column names
        values: List of value tuples (each tuple corresponds to one row)
    
    Returns:
        Number of rows inserted
    """
    if not values:
        return 0
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    placeholders = ', '.join(['%s'] * len(columns))
    column_str = ', '.join(columns)
    
    query = f"INSERT INTO {table} ({column_str}) VALUES %s"
    
    try:
        psycopg2.extras.execute_values(cursor, query, values)
        count = cursor.rowcount
        cursor.close()
        return count
    except Exception as e:
        logger.error(f"Error saving batch to {table}: {e}")
        cursor.close()
        raise


def query(sql: str, params: tuple = None, fetch_all: bool = True) -> List[Dict[str, Any]]:
    """
    Execute a database query and return results as dictionaries
    
    Args:
        sql: SQL query to execute
        params: Query parameters
        fetch_all: Whether to fetch all results or just the first one
    
    Returns:
        Query results as dictionaries
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    try:
        cursor.execute(sql, params)
        
        if fetch_all:
            results = cursor.fetchall()
        else:
            results = cursor.fetchone()
            
        # Convert to regular dictionaries
        if results is not None:
            if fetch_all:
                results = [dict(row) for row in results]
            else:
                results = dict(results)
                
        cursor.close()
        return results
    except Exception as e:
        logger.error(f"Query error: {e}")
        cursor.close()
        raise


def execute(sql: str, params: tuple = None) -> int:
    """
    Execute a database command (INSERT, UPDATE, DELETE)
    
    Args:
        sql: SQL command to execute
        params: Command parameters
    
    Returns:
        Number of affected rows
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(sql, params)
        rowcount = cursor.rowcount
        cursor.close()
        return rowcount
    except Exception as e:
        logger.error(f"Execute error: {e}")
        cursor.close()
        raise