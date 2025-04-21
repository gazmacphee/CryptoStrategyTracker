"""
Data Management Service

This service handles data retrieval, processing, and storage,
coordinating between external data sources and the database.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import time
import requests
import io
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config.container import container
from src.config import settings


class DataService:
    """Service for managing cryptocurrency data"""
    
    def __init__(self):
        """Initialize the data service"""
        self.logger = container.get("logger")
        self.historical_repo = container.get("historical_data_repo")
        self.indicators_repo = container.get("indicators_repo")
        
        # Cache for in-memory data that doesn't need to be fetched repeatedly
        self._cache = {}
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading symbols
        
        Returns:
            List of symbol strings
        """
        # Try to get symbols from database first
        db_symbols = self.historical_repo.get_available_symbols()
        
        if db_symbols:
            return db_symbols
        
        # Fall back to default symbols if database is empty
        return settings.DEFAULT_SYMBOLS
    
    def get_klines_data(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get klines (candlestick) data for a symbol
        
        Tries to get data from the database first, then falls back to
        external API if needed.
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            start_time: Start time for data range
            end_time: End time for data range
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with klines data
        """
        # Try to get from database first
        df = self.historical_repo.get_historical_data(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        # If we got data from the database, return it
        if not df.empty:
            return df
        
        # Fall back to downloading from Binance
        self.logger.info(f"No data in database for {symbol}/{interval}. Downloading from Binance...")
        
        # Download data
        try:
            downloaded_df = self._download_from_binance(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            if not downloaded_df.empty:
                # Save to database for future use
                self._save_dataframe_to_db(downloaded_df, symbol, interval)
                return downloaded_df
        except Exception as e:
            self.logger.error(f"Error downloading data from Binance: {e}")
        
        # Return empty DataFrame if we couldn't get data
        return pd.DataFrame()
    
    def _download_from_binance(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Download data from Binance API
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            start_time: Start time for data range
            end_time: End time for data range
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with downloaded data
        """
        # Define API endpoint
        api_endpoints = settings.BINANCE_API_ENDPOINTS
        
        # Try each endpoint
        for endpoint in api_endpoints:
            try:
                url = f"{endpoint}/klines"
                
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'limit': limit
                }
                
                if start_time:
                    params['startTime'] = int(start_time.timestamp() * 1000)
                
                if end_time:
                    params['endTime'] = int(end_time.timestamp() * 1000)
                
                # Add API key if available
                if settings.BINANCE_API_KEY:
                    params['api_key'] = settings.BINANCE_API_KEY
                
                # Make the request
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(data, columns=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Convert to correct types
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col])
                    
                    # Convert timestamps to datetime
                    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
                    
                    # Select and rename columns
                    result = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
                    
                    # Add symbol and interval
                    result['symbol'] = symbol
                    result['interval'] = interval
                    
                    return result
                
                # If we got any response, even an error, break the loop
                # This avoids trying multiple endpoints for rate limits or invalid symbols
                break
            
            except Exception as e:
                self.logger.warning(f"Error accessing Binance API endpoint {endpoint}: {e}")
                continue
        
        # Fall back to Binance Data Vision
        try:
            return self._download_from_binance_data_vision(symbol, interval, start_time, end_time)
        except Exception as e:
            self.logger.error(f"Error downloading from Binance Data Vision: {e}")
            return pd.DataFrame()
    
    def _download_from_binance_data_vision(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Download historical data from Binance Data Vision archive
        
        This is a fallback when the API is not available or rate-limited.
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            start_time: Start time for data range
            end_time: End time for data range
            
        Returns:
            DataFrame with downloaded data
        """
        # Default to the last 30 days if no time range specified
        if not end_time:
            end_time = datetime.now()
        
        if not start_time:
            start_time = end_time - timedelta(days=30)
        
        # Generate monthly and daily files to download
        files_to_download = self._get_binance_archive_files(symbol, interval, start_time, end_time)
        
        # Download files in parallel
        all_data = []
        
        with ThreadPoolExecutor(max_workers=settings.MAX_DOWNLOAD_WORKERS) as executor:
            # Submit download tasks
            future_to_file = {
                executor.submit(self._download_and_parse_file, file_url): file_url
                for file_url in files_to_download
            }
            
            # Process results as they complete
            for future in as_completed(future_to_file):
                file_url = future_to_file[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        all_data.append(data)
                except Exception as e:
                    self.logger.error(f"Error downloading {file_url}: {e}")
        
        # Combine all data
        if not all_data:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Filter to requested time range
        combined_df = combined_df[
            (combined_df['timestamp'] >= pd.Timestamp(start_time)) &
            (combined_df['timestamp'] <= pd.Timestamp(end_time))
        ]
        
        # Sort by timestamp
        combined_df = combined_df.sort_values('timestamp')
        
        # Drop duplicates
        combined_df = combined_df.drop_duplicates(subset=['timestamp'])
        
        return combined_df
    
    def _get_binance_archive_files(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[str]:
        """
        Generate URLs for Binance Data Vision archive files
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            start_time: Start time for data range
            end_time: End time for data range
            
        Returns:
            List of file URLs to download
        """
        base_url = settings.BINANCE_DATA_VISION_BASE_URL
        data_type = "spot"  # Use spot market data
        files = []
        
        # Generate monthly files first
        current_month = start_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_month = end_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        while current_month <= end_month:
            year = current_month.year
            month = current_month.month
            
            # Monthly URL pattern
            monthly_url = f"{base_url}/data/{data_type}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"
            files.append(monthly_url)
            
            # Move to next month
            if current_month.month == 12:
                current_month = current_month.replace(year=current_month.year + 1, month=1)
            else:
                current_month = current_month.replace(month=current_month.month + 1)
        
        # Also add daily files for the current month
        # This helps when the monthly file isn't yet available
        current_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_current_month = current_day.replace(day=1)
        
        # Only add daily files for the current month
        if end_time >= start_current_month:
            day = start_current_month
            
            while day <= min(end_time, current_day):
                year = day.year
                month = day.month
                day_of_month = day.day
                
                # Daily URL pattern
                daily_url = f"{base_url}/data/{data_type}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}-{day_of_month:02d}.zip"
                files.append(daily_url)
                
                # Move to next day
                day += timedelta(days=1)
        
        return files
    
    def _download_and_parse_file(self, file_url: str) -> Optional[pd.DataFrame]:
        """
        Download and parse a Binance Data Vision file
        
        Args:
            file_url: URL of file to download
            
        Returns:
            DataFrame with parsed data or None on error
        """
        self.logger.debug(f"Downloading {file_url}")
        
        # Extract symbol and interval from URL
        parts = file_url.split('/')
        symbol = parts[-3]
        interval = parts[-2]
        
        try:
            # Download the file
            response = requests.get(file_url, timeout=settings.DOWNLOAD_TIMEOUT_SECONDS)
            
            # Check if file exists
            if response.status_code == 404:
                self.logger.debug(f"File not found: {file_url}")
                return None
            
            # Check for other errors
            response.raise_for_status()
            
            # Process the zip file
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # There should be a single CSV file inside
                csv_filename = z.namelist()[0]
                
                with z.open(csv_filename) as f:
                    # Read the CSV
                    df = pd.read_csv(f, header=None, names=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Convert to correct types
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col])
                    
                    # Try different timestamp formats
                    # Some Binance files use milliseconds, others use seconds
                    try:
                        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
                    except:
                        try:
                            # Try seconds
                            df['timestamp'] = pd.to_datetime(df['open_time'], unit='s')
                        except:
                            self.logger.warning(f"Could not parse timestamps in {file_url}")
                            return None
                    
                    # Select and rename columns
                    result = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
                    
                    # Add symbol and interval
                    result['symbol'] = symbol
                    result['interval'] = interval
                    
                    return result
        
        except Exception as e:
            self.logger.warning(f"Error processing {file_url}: {e}")
            return None
    
    def _save_dataframe_to_db(self, df: pd.DataFrame, symbol: str, interval: str) -> int:
        """
        Save a DataFrame to the database
        
        Args:
            df: DataFrame with data to save
            symbol: Trading pair symbol
            interval: Time interval
            
        Returns:
            Number of records saved
        """
        if df.empty:
            return 0
        
        # Convert DataFrame to list of dictionaries
        candles = df.to_dict('records')
        
        # Save to database
        return self.historical_repo.save_candles(symbol, interval, candles)
    
    def detect_data_gaps(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Detect gaps in historical data
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            start_time: Start time for data range
            end_time: End time for data range
            
        Returns:
            List of gap dictionaries with start and end times
        """
        # Get data for the period
        df = self.historical_repo.get_historical_data(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time
        )
        
        if df.empty:
            # If no data, the entire range is a gap
            return [{
                'symbol': symbol,
                'interval': interval,
                'start_time': start_time,
                'end_time': end_time,
                'expected_candles': self._calculate_expected_candles(interval, start_time, end_time)
            }]
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Calculate expected time between candles
        seconds = self._interval_to_seconds(interval)
        
        # Initialize gaps list
        gaps = []
        
        # Check if there's a gap at the start
        if df['timestamp'].min() > start_time:
            gaps.append({
                'symbol': symbol,
                'interval': interval,
                'start_time': start_time,
                'end_time': df['timestamp'].min(),
                'expected_candles': self._calculate_expected_candles(interval, start_time, df['timestamp'].min())
            })
        
        # Check for gaps between candles
        for i in range(len(df) - 1):
            current_time = df.iloc[i]['timestamp']
            next_time = df.iloc[i + 1]['timestamp']
            
            expected_next = current_time + timedelta(seconds=seconds)
            
            if (next_time - expected_next).total_seconds() > seconds * 0.5:  # Allow some tolerance
                gaps.append({
                    'symbol': symbol,
                    'interval': interval,
                    'start_time': expected_next,
                    'end_time': next_time,
                    'expected_candles': self._calculate_expected_candles(interval, expected_next, next_time)
                })
        
        # Check if there's a gap at the end
        if df['timestamp'].max() < end_time:
            gaps.append({
                'symbol': symbol,
                'interval': interval,
                'start_time': df['timestamp'].max() + timedelta(seconds=seconds),
                'end_time': end_time,
                'expected_candles': self._calculate_expected_candles(interval, df['timestamp'].max() + timedelta(seconds=seconds), end_time)
            })
        
        return gaps
    
    def fill_data_gaps(
        self,
        gaps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Fill detected gaps in data
        
        Args:
            gaps: List of gap dictionaries with start and end times
            
        Returns:
            Dictionary with results
        """
        results = {
            'total_gaps': len(gaps),
            'filled_gaps': 0,
            'total_candles_filled': 0,
            'failed_gaps': []
        }
        
        for gap in gaps:
            symbol = gap['symbol']
            interval = gap['interval']
            start_time = gap['start_time']
            end_time = gap['end_time']
            
            try:
                # Download data for the gap
                df = self._download_from_binance(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if not df.empty:
                    # Save to database
                    candles_saved = self._save_dataframe_to_db(df, symbol, interval)
                    
                    if candles_saved > 0:
                        results['filled_gaps'] += 1
                        results['total_candles_filled'] += candles_saved
                    else:
                        results['failed_gaps'].append(gap)
                else:
                    results['failed_gaps'].append(gap)
            
            except Exception as e:
                self.logger.error(f"Error filling gap for {symbol}/{interval}: {e}")
                results['failed_gaps'].append(gap)
        
        return results
    
    def _interval_to_seconds(self, interval: str) -> int:
        """
        Convert interval string to seconds
        
        Args:
            interval: Interval string (e.g., '1h', '15m')
            
        Returns:
            Number of seconds
        """
        unit = interval[-1]
        value = int(interval[:-1])
        
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 3600
        elif unit == 'd':
            return value * 86400
        elif unit == 'w':
            return value * 604800
        else:
            return value
    
    def _calculate_expected_candles(
        self,
        interval: str,
        start_time: datetime,
        end_time: datetime
    ) -> int:
        """
        Calculate expected number of candles in a time range
        
        Args:
            interval: Time interval
            start_time: Start time
            end_time: End time
            
        Returns:
            Expected number of candles
        """
        seconds = self._interval_to_seconds(interval)
        time_diff = (end_time - start_time).total_seconds()
        
        # Add 1 to include both endpoints
        return max(0, int(time_diff / seconds) + 1)


# Register in the container
def data_service_factory(container):
    return DataService()

container.register_service("data_service", data_service_factory)