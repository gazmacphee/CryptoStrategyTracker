"""
Data Repository Classes

These classes provide a clean interface to data access operations
for each entity type, encapsulating SQL queries and data transformations.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

from src.data.database import query, execute, save_batch
from src.config.container import container


class HistoricalDataRepository:
    """Repository for historical price data operations"""
    
    def __init__(self, db_connection=None, logger=None):
        """
        Initialize the repository
        
        Args:
            db_connection: Database connection
            logger: Logger instance
        """
        self.db_connection = db_connection
        self.logger = logger or container.get("logger")
    
    def save_candles(self, symbol: str, interval: str, candles: List[Dict[str, Any]]) -> int:
        """
        Save multiple candles in a batch operation
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            candles: List of candle dictionaries
            
        Returns:
            Number of candles saved
        """
        if not candles:
            return 0
        
        columns = ['symbol', 'interval', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        values = [
            (
                symbol,
                interval,
                candle['timestamp'],
                candle['open'],
                candle['high'],
                candle['low'],
                candle['close'],
                candle['volume']
            )
            for candle in candles
        ]
        
        try:
            return save_batch('historical_data', columns, values)
        except Exception as e:
            self.logger.error(f"Error saving candles for {symbol}/{interval}: {e}")
            raise
    
    def get_historical_data(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get historical price data
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            start_time: Start time for data range
            end_time: End time for data range
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with historical price data
        """
        params = [symbol, interval]
        sql = """
            SELECT * FROM historical_data
            WHERE symbol = %s AND interval = %s
        """
        
        if start_time:
            sql += " AND timestamp >= %s"
            params.append(start_time)
        
        if end_time:
            sql += " AND timestamp <= %s"
            params.append(end_time)
        
        sql += " ORDER BY timestamp ASC"
        
        if limit:
            sql += " LIMIT %s"
            params.append(limit)
        
        try:
            result = query(sql, tuple(params))
            return pd.DataFrame(result) if result else pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}/{interval}: {e}")
            return pd.DataFrame()
    
    def get_last_timestamp(self, symbol: str, interval: str) -> Optional[datetime]:
        """
        Get the timestamp of the most recent data point
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            
        Returns:
            Timestamp of most recent data or None
        """
        sql = """
            SELECT MAX(timestamp) as last_timestamp
            FROM historical_data
            WHERE symbol = %s AND interval = %s
        """
        
        try:
            result = query(sql, (symbol, interval), fetch_all=False)
            return result['last_timestamp'] if result and result['last_timestamp'] else None
        except Exception as e:
            self.logger.error(f"Error getting last timestamp for {symbol}/{interval}: {e}")
            return None
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of symbols available in the database
        
        Returns:
            List of symbol strings
        """
        sql = """
            SELECT DISTINCT symbol FROM historical_data
            ORDER BY symbol
        """
        
        try:
            result = query(sql)
            return [row['symbol'] for row in result] if result else []
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []
    
    def delete_data_for_range(self, symbol: str, interval: str, start_time: datetime, end_time: datetime) -> int:
        """
        Delete data for a specific time range
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            start_time: Start of range to delete
            end_time: End of range to delete
            
        Returns:
            Number of records deleted
        """
        sql = """
            DELETE FROM historical_data
            WHERE symbol = %s 
            AND interval = %s 
            AND timestamp >= %s 
            AND timestamp <= %s
        """
        
        try:
            return execute(sql, (symbol, interval, start_time, end_time))
        except Exception as e:
            self.logger.error(f"Error deleting data for {symbol}/{interval}: {e}")
            raise


class TechnicalIndicatorsRepository:
    """Repository for technical indicators data operations"""
    
    def __init__(self, db_connection=None, logger=None):
        """
        Initialize the repository
        
        Args:
            db_connection: Database connection
            logger: Logger instance
        """
        self.db_connection = db_connection
        self.logger = logger or container.get("logger")
    
    def save_indicators(self, indicators_data: List[Dict[str, Any]]) -> int:
        """
        Save technical indicators in a batch operation
        
        Args:
            indicators_data: List of indicator dictionaries
            
        Returns:
            Number of records saved
        """
        if not indicators_data:
            return 0
        
        # Dynamically build columns from the first record to handle different indicators
        first_record = indicators_data[0]
        columns = list(first_record.keys())
        
        # Prepare values for batch insertion
        values = [
            tuple(record.get(col, None) for col in columns)
            for record in indicators_data
        ]
        
        try:
            return save_batch('technical_indicators', columns, values)
        except Exception as e:
            self.logger.error(f"Error saving technical indicators: {e}")
            raise
    
    def get_indicators(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get technical indicators
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            start_time: Start time for data range
            end_time: End time for data range
            
        Returns:
            DataFrame with technical indicators
        """
        params = [symbol, interval]
        sql = """
            SELECT * FROM technical_indicators
            WHERE symbol = %s AND interval = %s
        """
        
        if start_time:
            sql += " AND timestamp >= %s"
            params.append(start_time)
        
        if end_time:
            sql += " AND timestamp <= %s"
            params.append(end_time)
        
        sql += " ORDER BY timestamp ASC"
        
        try:
            result = query(sql, tuple(params))
            return pd.DataFrame(result) if result else pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error getting indicators for {symbol}/{interval}: {e}")
            return pd.DataFrame()


class SentimentRepository:
    """Repository for sentiment data operations"""
    
    def __init__(self, db_connection=None, logger=None):
        """
        Initialize the repository
        
        Args:
            db_connection: Database connection
            logger: Logger instance
        """
        self.db_connection = db_connection
        self.logger = logger or container.get("logger")
    
    def save_sentiment(self, sentiment_data: List[Dict[str, Any]]) -> int:
        """
        Save sentiment data in a batch operation
        
        Args:
            sentiment_data: List of sentiment dictionaries
            
        Returns:
            Number of records saved
        """
        if not sentiment_data:
            return 0
        
        columns = ['symbol', 'timestamp', 'source', 'sentiment_score', 'volume']
        values = [
            (
                item['symbol'],
                item['timestamp'],
                item['source'],
                item['sentiment_score'],
                item.get('volume', 1)  # Default volume to 1 if not specified
            )
            for item in sentiment_data
        ]
        
        try:
            return save_batch('sentiment_data', columns, values)
        except Exception as e:
            self.logger.error(f"Error saving sentiment data: {e}")
            raise
    
    def get_sentiment(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get sentiment data
        
        Args:
            symbol: Trading pair symbol or currency name
            start_time: Start time for data range
            end_time: End time for data range
            
        Returns:
            DataFrame with sentiment data
        """
        # Allow partial matching for base currencies (e.g., BTC for BTCUSDT)
        symbol_condition = "symbol = %s OR symbol LIKE %s"
        params = [symbol, f"{symbol}%"]
        
        sql = f"""
            SELECT * FROM sentiment_data
            WHERE ({symbol_condition})
        """
        
        if start_time:
            sql += " AND timestamp >= %s"
            params.append(start_time)
        
        if end_time:
            sql += " AND timestamp <= %s"
            params.append(end_time)
        
        sql += " ORDER BY timestamp ASC"
        
        try:
            result = query(sql, tuple(params))
            return pd.DataFrame(result) if result else pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error getting sentiment data for {symbol}: {e}")
            return pd.DataFrame()


class TradeRepository:
    """Repository for trade data operations"""
    
    def __init__(self, db_connection=None, logger=None):
        """
        Initialize the repository
        
        Args:
            db_connection: Database connection
            logger: Logger instance
        """
        self.db_connection = db_connection
        self.logger = logger or container.get("logger")
    
    def save_trade(self, trade_data: Dict[str, Any]) -> int:
        """
        Save a trade record
        
        Args:
            trade_data: Trade dictionary
            
        Returns:
            New trade ID
        """
        columns = list(trade_data.keys())
        placeholders = ', '.join(['%s'] * len(columns))
        column_str = ', '.join(columns)
        
        sql = f"""
            INSERT INTO trades ({column_str})
            VALUES ({placeholders})
            RETURNING id
        """
        
        values = tuple(trade_data.values())
        
        try:
            result = query(sql, values, fetch_all=False)
            return result['id'] if result else None
        except Exception as e:
            self.logger.error(f"Error saving trade: {e}")
            raise
    
    def update_trade(self, trade_id: int, update_data: Dict[str, Any]) -> bool:
        """
        Update an existing trade
        
        Args:
            trade_id: ID of trade to update
            update_data: Dictionary of fields to update
            
        Returns:
            True if update was successful
        """
        set_clauses = []
        values = []
        
        for key, value in update_data.items():
            set_clauses.append(f"{key} = %s")
            values.append(value)
        
        set_clause = ', '.join(set_clauses)
        values.append(trade_id)
        
        sql = f"""
            UPDATE trades
            SET {set_clause}
            WHERE id = %s
        """
        
        try:
            rowcount = execute(sql, tuple(values))
            return rowcount > 0
        except Exception as e:
            self.logger.error(f"Error updating trade {trade_id}: {e}")
            raise
    
    def get_trades(
        self,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
        strategy: Optional[str] = None,
        status: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get trades with optional filtering
        
        Args:
            symbol: Filter by symbol
            interval: Filter by interval
            strategy: Filter by strategy
            status: Filter by status
            start_time: Start time for data range
            end_time: End time for data range
            
        Returns:
            DataFrame with trade data
        """
        conditions = []
        params = []
        
        if symbol:
            conditions.append("symbol = %s")
            params.append(symbol)
        
        if interval:
            conditions.append("interval = %s")
            params.append(interval)
        
        if strategy:
            conditions.append("strategy = %s")
            params.append(strategy)
        
        if status:
            conditions.append("status = %s")
            params.append(status)
        
        if start_time:
            conditions.append("entry_time >= %s")
            params.append(start_time)
        
        if end_time:
            conditions.append("entry_time <= %s")
            params.append(end_time)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        sql = f"""
            SELECT * FROM trades
            WHERE {where_clause}
            ORDER BY entry_time DESC
        """
        
        try:
            result = query(sql, tuple(params) if params else None)
            return pd.DataFrame(result) if result else pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error getting trades: {e}")
            return pd.DataFrame()


# Note: We don't register repositories directly here anymore
# They are now registered in src/config/initialize.py with proper dependency injection
# This avoids circular dependencies and makes the code more testable