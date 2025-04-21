"""
Backfill Service Module

This module provides functionality to download and process historical data
for multiple symbol/interval combinations in parallel, with managed concurrency.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import threading
from datetime import datetime, timedelta
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from queue import Queue, Empty
import json

from src.config.container import container
from src.config import settings


class BackfillService:
    """Service for backfilling historical data"""
    
    def __init__(self, logger=None, data_service=None, indicators_service=None):
        """
        Initialize the backfill service
        
        Args:
            logger: Logger instance
            data_service: Data service instance
            indicators_service: Indicators service instance
        """
        # Allow injected dependencies or get from container
        self.logger = logger or container.get("logger")
        self.data_service = data_service or container.get("data_service")
        self.indicators_service = indicators_service or container.get("indicators_service")
        
        # Running state
        self.is_running = False
        self.stop_requested = False
        self.progress = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'current_tasks': {},
            'start_time': None,
            'last_update': None
        }
        
        # Backfill lock file path
        self.lock_file = '.backfill_lock'
    
    def start_backfill(
        self,
        symbols: Optional[List[str]] = None,
        intervals: Optional[List[str]] = None,
        days_back: int = 180,
        max_workers: Optional[int] = None,
        update_indicators: bool = True
    ) -> Dict[str, Any]:
        """
        Start the backfill process in the background
        
        Args:
            symbols: List of symbols to process
            intervals: List of intervals to process
            days_back: How many days to backfill
            max_workers: Maximum number of concurrent workers
            update_indicators: Whether to update indicators after backfilling data
            
        Returns:
            Dictionary with start status
        """
        if self.is_running:
            return {'status': 'already_running', 'message': 'Backfill process is already running'}
        
        if self.lock_exists():
            return {'status': 'locked', 'message': 'Backfill is locked by another process'}
        
        # Use default symbols/intervals if none provided
        if symbols is None:
            symbols = settings.DEFAULT_SYMBOLS
        
        if intervals is None:
            intervals = settings.DEFAULT_INTERVALS
        
        # Use default max workers if none provided
        if max_workers is None:
            max_workers = settings.MAX_DOWNLOAD_WORKERS
        
        # Create the lock file
        self._create_lock_file()
        
        # Initialize progress
        self.progress = {
            'total_tasks': len(symbols) * len(intervals),
            'completed_tasks': 0,
            'current_tasks': {},
            'start_time': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat()
        }
        
        # Start the backfill thread
        thread = threading.Thread(
            target=self._run_backfill,
            args=(symbols, intervals, days_back, max_workers, update_indicators),
            daemon=True
        )
        thread.start()
        
        return {
            'status': 'started',
            'message': f'Backfill process started in background for {len(symbols)} symbols and {len(intervals)} intervals',
            'total_tasks': self.progress['total_tasks']
        }
    
    def _run_backfill(
        self,
        symbols: List[str],
        intervals: List[str],
        days_back: int,
        max_workers: int,
        update_indicators: bool
    ) -> None:
        """
        Run the backfill process
        
        Args:
            symbols: List of symbols to process
            intervals: List of intervals to process
            days_back: How many days to backfill
            max_workers: Maximum number of concurrent workers
            update_indicators: Whether to update indicators after backfilling data
        """
        self.is_running = True
        self.stop_requested = False
        
        # Log backfill start
        self.logger.info(f"Starting backfill process for {len(symbols)} symbols and {len(intervals)} intervals")
        self.logger.info(f"Symbols: {', '.join(symbols)}")
        self.logger.info(f"Intervals: {', '.join(intervals)}")
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        # Create tasks
        tasks = []
        for symbol in symbols:
            for interval in intervals:
                tasks.append((symbol, interval))
        
        # Track task results
        results = {}
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit initial batch of tasks
            future_to_task = {}
            for task in tasks[:max_workers]:
                symbol, interval = task
                future = executor.submit(
                    self._process_symbol_interval,
                    symbol, interval, start_time, end_time
                )
                future_to_task[future] = task
                
                # Update progress
                self.progress['current_tasks'][f"{symbol}_{interval}"] = {
                    'symbol': symbol,
                    'interval': interval,
                    'status': 'processing',
                    'start_time': datetime.now().isoformat()
                }
            
            # Process completed tasks and submit new ones
            tasks_index = max_workers
            while future_to_task and not self.stop_requested:
                # Wait for the next task to complete
                done, _ = concurrent.futures.wait(
                    future_to_task.keys(), 
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                for future in done:
                    symbol, interval = future_to_task[future]
                    
                    try:
                        result = future.result()
                        results[f"{symbol}_{interval}"] = result
                        
                        # Log completion
                        self.logger.info(f"Completed {symbol}/{interval}: {result['candles_processed']} candles processed")
                        
                        # Update indicators if requested
                        if update_indicators:
                            self._update_indicators(symbol, interval, start_time, end_time)
                    
                    except Exception as e:
                        self.logger.error(f"Error processing {symbol}/{interval}: {e}")
                        results[f"{symbol}_{interval}"] = {
                            'status': 'error',
                            'error': str(e)
                        }
                    
                    # Update progress
                    self.progress['completed_tasks'] += 1
                    if f"{symbol}_{interval}" in self.progress['current_tasks']:
                        del self.progress['current_tasks'][f"{symbol}_{interval}"]
                    
                    self.progress['last_update'] = datetime.now().isoformat()
                    
                    # Remove the future from our tracking
                    del future_to_task[future]
                    
                    # Submit a new task if available
                    if tasks_index < len(tasks) and not self.stop_requested:
                        symbol, interval = tasks[tasks_index]
                        future = executor.submit(
                            self._process_symbol_interval,
                            symbol, interval, start_time, end_time
                        )
                        future_to_task[future] = (symbol, interval)
                        
                        # Update progress
                        self.progress['current_tasks'][f"{symbol}_{interval}"] = {
                            'symbol': symbol,
                            'interval': interval,
                            'status': 'processing',
                            'start_time': datetime.now().isoformat()
                        }
                        
                        tasks_index += 1
        
        # Log backfill completion
        self.logger.info(f"Backfill process completed: {self.progress['completed_tasks']}/{self.progress['total_tasks']} tasks")
        
        # Clear the lock file
        self._clear_lock_file()
        
        # Update running state
        self.is_running = False
        
        # Save final results
        self._save_backfill_results(results)
    
    def _process_symbol_interval(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        Process a single symbol/interval combination
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            start_time: Start time for backfill
            end_time: End time for backfill
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing {symbol}/{interval}")
        
        # Initialize result
        result = {
            'symbol': symbol,
            'interval': interval,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'processing_start': datetime.now().isoformat(),
            'status': 'processing'
        }
        
        try:
            # Download data
            df = self.data_service._download_from_binance_data_vision(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time
            )
            
            if df.empty:
                self.logger.warning(f"No data available for {symbol}/{interval}")
                result.update({
                    'status': 'no_data',
                    'candles_processed': 0,
                    'processing_end': datetime.now().isoformat()
                })
                return result
            
            # Save to database
            candles_processed = self.data_service._save_dataframe_to_db(df, symbol, interval)
            
            # Update result
            result.update({
                'status': 'success',
                'candles_processed': candles_processed,
                'processing_end': datetime.now().isoformat()
            })
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error processing {symbol}/{interval}: {e}")
            result.update({
                'status': 'error',
                'error': str(e),
                'processing_end': datetime.now().isoformat()
            })
            return result
    
    def _update_indicators(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        Update indicators for a symbol/interval
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            start_time: Start time for update
            end_time: End time for update
            
        Returns:
            Dictionary with update results
        """
        self.logger.info(f"Updating indicators for {symbol}/{interval}")
        
        try:
            # Update indicators
            result = self.indicators_service.update_indicators(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time
            )
            
            self.logger.info(f"Indicators updated for {symbol}/{interval}: {result}")
            return result
        
        except Exception as e:
            self.logger.error(f"Error updating indicators for {symbol}/{interval}: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def stop_backfill(self) -> Dict[str, Any]:
        """
        Stop the backfill process
        
        Returns:
            Dictionary with stop status
        """
        if not self.is_running:
            return {'status': 'not_running', 'message': 'Backfill process is not running'}
        
        # Set stop flag
        self.stop_requested = True
        
        return {'status': 'stopping', 'message': 'Stop signal sent to backfill process'}
    
    def get_backfill_progress(self) -> Dict[str, Any]:
        """
        Get the current backfill progress
        
        Returns:
            Dictionary with progress information
        """
        if not self.is_running and not self.lock_exists():
            return {'status': 'not_running', 'message': 'Backfill process is not running'}
        
        # Calculate overall progress percentage
        percentage = 0
        if self.progress['total_tasks'] > 0:
            percentage = (self.progress['completed_tasks'] / self.progress['total_tasks']) * 100
        
        # Calculate elapsed time
        elapsed_seconds = 0
        if self.progress['start_time']:
            start_time = datetime.fromisoformat(self.progress['start_time'])
            elapsed_seconds = (datetime.now() - start_time).total_seconds()
        
        # Calculate estimated remaining time
        remaining_seconds = 0
        if percentage > 0:
            remaining_seconds = (elapsed_seconds / percentage) * (100 - percentage)
        
        return {
            'status': 'running' if self.is_running else 'locked',
            'progress': {
                'percentage': round(percentage, 1),
                'completed_tasks': self.progress['completed_tasks'],
                'total_tasks': self.progress['total_tasks'],
                'current_tasks': self.progress['current_tasks'],
                'elapsed_seconds': round(elapsed_seconds),
                'estimated_remaining_seconds': round(remaining_seconds)
            }
        }
    
    def lock_exists(self) -> bool:
        """
        Check if a backfill lock file exists
        
        Returns:
            True if lock exists, False otherwise
        """
        return os.path.exists(self.lock_file)
    
    def _create_lock_file(self) -> None:
        """Create a backfill lock file"""
        with open(self.lock_file, 'w') as f:
            f.write(datetime.now().isoformat())
        
        self.logger.info(f"Created backfill lock file: {self.lock_file}")
    
    def _clear_lock_file(self) -> None:
        """Remove the backfill lock file"""
        if os.path.exists(self.lock_file):
            os.remove(self.lock_file)
            self.logger.info(f"Removed backfill lock file: {self.lock_file}")
    
    def _save_backfill_results(self, results: Dict[str, Any]) -> None:
        """
        Save backfill results to a file
        
        Args:
            results: Dictionary with backfill results
        """
        filename = f"backfill_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, default=str)
            
            self.logger.info(f"Saved backfill results to: {filename}")
        except Exception as e:
            self.logger.error(f"Error saving backfill results: {e}")


# Factory function for dependency injection
def backfill_service_factory(container):
    """
    Create and return a BackfillService instance with injected dependencies
    
    Args:
        container: The dependency container
        
    Returns:
        Configured BackfillService instance
    """
    # Get required dependencies from container
    logger = container.get("logger")
    data_service = container.get("data_service")
    indicators_service = container.get("indicators_service")
    
    # Create and return service with explicit dependencies
    return BackfillService(
        logger=logger,
        data_service=data_service,
        indicators_service=indicators_service
    )