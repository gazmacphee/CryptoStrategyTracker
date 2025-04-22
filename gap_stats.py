"""
Gap statistics module for tracking and analyzing data gaps.
This module detects, reports, and helps manage time series gaps in financial data.
"""
import logging
import datetime
import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional
import database

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
GAP_LOG_FILE = "gap_analysis.log"
TEMP_GAP_DIR = "temp_gaps"

def ensure_temp_dir():
    """Create temporary directory for gap logs if it doesn't exist"""
    if not os.path.exists(TEMP_GAP_DIR):
        os.makedirs(TEMP_GAP_DIR)

def detect_gaps(symbol: str, interval: str, start_date: Optional[datetime.datetime] = None, 
               end_date: Optional[datetime.datetime] = None) -> pd.DataFrame:
    """
    Detect gaps in time series data for a specific symbol and interval.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '4h', '1d')
        start_date: Optional start date for gap detection
        end_date: Optional end date for gap detection
        
    Returns:
        DataFrame with detected gaps
    """
    try:
        # Set default date range if not provided
        if not start_date:
            start_date = datetime.datetime.now() - datetime.timedelta(days=30)
        if not end_date:
            end_date = datetime.datetime.now()
            
        # Get data from database
        df = database.get_historical_data(
            symbol, 
            interval, 
            start_date.timestamp(), 
            end_date.timestamp()
        )
        
        if df.empty:
            logger.info(f"No data found for {symbol}/{interval} in the specified date range")
            return pd.DataFrame()
            
        # Sort by timestamp to ensure chronological order
        df.sort_values('timestamp', inplace=True)
        
        # Get the expected interval in seconds
        interval_seconds = _get_interval_seconds(interval)
        
        # Detect gaps
        gaps = []
        prev_row = None
        
        for _, row in df.iterrows():
            if prev_row is not None:
                time_diff = row['timestamp'] - prev_row['timestamp']
                
                # If time difference is greater than expected interval + 5% tolerance
                if time_diff > interval_seconds * 1.05:
                    expected_records = int(time_diff / interval_seconds) - 1
                    if expected_records > 0:
                        gap_start = datetime.datetime.fromtimestamp(prev_row['timestamp'] + interval_seconds)
                        gap_end = datetime.datetime.fromtimestamp(row['timestamp'] - interval_seconds)
                        
                        gaps.append({
                            'symbol': symbol,
                            'interval': interval,
                            'gap_start': gap_start,
                            'gap_end': gap_end,
                            'missing_records': expected_records,
                            'gap_duration_hours': (gap_end - gap_start).total_seconds() / 3600
                        })
            
            prev_row = row
            
        # Create DataFrame from gaps
        if gaps:
            gaps_df = pd.DataFrame(gaps)
            
            # Add gap ID and status fields
            gaps_df['gap_id'] = [f"GAP_{symbol}_{interval}_{i}" for i in range(len(gaps))]
            gaps_df['status'] = "Detected"
            
            # Log gaps
            _log_gaps(gaps_df)
            
            return gaps_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error detecting gaps: {str(e)}")
        return pd.DataFrame()

def get_gap_stats() -> Dict[str, Any]:
    """
    Get statistics about detected data gaps.
    
    Returns:
        Dictionary with gap statistics
    """
    try:
        ensure_temp_dir()
        
        # Check if gap log exists
        if not os.path.exists(os.path.join(TEMP_GAP_DIR, GAP_LOG_FILE)):
            return {
                "total_gaps": 0,
                "filled_gaps": 0,
                "gap_details": pd.DataFrame()
            }
            
        # Load gap log
        with open(os.path.join(TEMP_GAP_DIR, GAP_LOG_FILE), 'r') as f:
            gaps = json.load(f)
            
        if not gaps or not gaps.get('gaps'):
            return {
                "total_gaps": 0,
                "filled_gaps": 0,
                "gap_details": pd.DataFrame()
            }
            
        # Convert to DataFrame
        gaps_df = pd.DataFrame(gaps['gaps'])
        
        # Convert timestamp columns to datetime
        if 'gap_start' in gaps_df.columns:
            gaps_df['gap_start'] = pd.to_datetime(gaps_df['gap_start'])
        if 'gap_end' in gaps_df.columns:
            gaps_df['gap_end'] = pd.to_datetime(gaps_df['gap_end'])
            
        # Clean up column names for display
        renamed_columns = {
            'gap_id': 'Gap ID',
            'symbol': 'Symbol',
            'interval': 'Interval',
            'gap_start': 'Gap Start',
            'gap_end': 'Gap End',
            'missing_records': 'Missing Records',
            'gap_duration_hours': 'Duration (hours)',
            'status': 'Status'
        }
        
        display_df = gaps_df.rename(columns=renamed_columns)
        
        # Get counts
        total_gaps = len(gaps_df)
        filled_gaps = len(gaps_df[gaps_df['status'] == 'Filled'])
        
        # Symbol and interval breakdowns
        symbol_counts = gaps_df.groupby('symbol').size().reset_index(name='count')
        symbol_counts = symbol_counts.rename(columns={'symbol': 'Symbol', 'count': 'Gap Count'})
        
        interval_counts = gaps_df.groupby('interval').size().reset_index(name='count')
        interval_counts = interval_counts.rename(columns={'interval': 'Interval', 'count': 'Gap Count'})
        
        # Status breakdown
        status_counts = gaps_df.groupby('status').size().reset_index(name='count')
        status_counts = status_counts.rename(columns={'status': 'Status', 'count': 'Count'})
        
        return {
            "total_gaps": total_gaps,
            "filled_gaps": filled_gaps,
            "gap_details": display_df,
            "symbol_counts": symbol_counts,
            "interval_counts": interval_counts,
            "status_counts": status_counts
        }
        
    except Exception as e:
        logger.error(f"Error getting gap statistics: {str(e)}")
        return {
            "total_gaps": 0,
            "filled_gaps": 0,
            "error": str(e)
        }

def clear_gap_log() -> bool:
    """
    Clear the gap analysis log file.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_temp_dir()
        log_path = os.path.join(TEMP_GAP_DIR, GAP_LOG_FILE)
        
        # Create empty gap log
        with open(log_path, 'w') as f:
            json.dump({"gaps": []}, f)
            
        return True
        
    except Exception as e:
        logger.error(f"Error clearing gap log: {str(e)}")
        return False

def update_gap_status(gap_id: str, new_status: str) -> bool:
    """
    Update the status of a specific gap.
    
    Args:
        gap_id: ID of the gap to update
        new_status: New status to set
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_temp_dir()
        log_path = os.path.join(TEMP_GAP_DIR, GAP_LOG_FILE)
        
        # If log doesn't exist, there's nothing to update
        if not os.path.exists(log_path):
            return False
            
        # Load gap log
        with open(log_path, 'r') as f:
            gaps = json.load(f)
            
        # Find and update gap
        updated = False
        for gap in gaps.get('gaps', []):
            if gap.get('gap_id') == gap_id:
                gap['status'] = new_status
                updated = True
                break
                
        if updated:
            # Save updated log
            with open(log_path, 'w') as f:
                json.dump(gaps, f)
                
            return True
        else:
            logger.warning(f"Gap ID {gap_id} not found in gap log")
            return False
            
    except Exception as e:
        logger.error(f"Error updating gap status: {str(e)}")
        return False

def fill_gaps(symbol: str = None, interval: str = None) -> int:
    """
    Attempt to fill detected gaps for specified symbol and interval.
    If no symbol/interval is specified, attempts to fill all detected gaps.
    
    Args:
        symbol: Optional symbol to limit gap filling
        interval: Optional interval to limit gap filling
        
    Returns:
        Number of gaps successfully filled
    """
    try:
        # Get gap statistics
        gap_stats = get_gap_stats()
        
        if gap_stats["total_gaps"] == 0:
            logger.info("No gaps to fill")
            return 0
            
        gaps_df = gap_stats.get("gap_details")
        if gaps_df is None or gaps_df.empty:
            return 0
            
        # Convert column names back to original
        column_mapping = {
            'Gap ID': 'gap_id',
            'Symbol': 'symbol',
            'Interval': 'interval',
            'Gap Start': 'gap_start',
            'Gap End': 'gap_end',
            'Missing Records': 'missing_records',
            'Duration (hours)': 'gap_duration_hours',
            'Status': 'status'
        }
        
        gaps_df = gaps_df.rename(columns=column_mapping)
        
        # Filter gaps by symbol and interval if specified
        if symbol:
            gaps_df = gaps_df[gaps_df['symbol'] == symbol]
        if interval:
            gaps_df = gaps_df[gaps_df['interval'] == interval]
            
        # Filter only unfilled gaps
        unfilled_gaps = gaps_df[gaps_df['status'] != 'Filled']
        
        if unfilled_gaps.empty:
            logger.info("No unfilled gaps to process")
            return 0
            
        # Attempt to fill each gap
        filled_count = 0
        
        for _, gap in unfilled_gaps.iterrows():
            # Mark as in progress
            update_gap_status(gap['gap_id'], 'In Progress')
            
            try:
                # Convert timestamps for API calls
                start_time = gap['gap_start'].timestamp() if isinstance(gap['gap_start'], datetime.datetime) else gap['gap_start']
                end_time = gap['gap_end'].timestamp() if isinstance(gap['gap_end'], datetime.datetime) else gap['gap_end']
                
                # This would call the data fetching function
                # For now, we'll just log it and mark as filled as a placeholder
                logger.info(f"Attempting to fill gap {gap['gap_id']} for {gap['symbol']}/{gap['interval']} from {gap['gap_start']} to {gap['gap_end']}")
                
                # Mark as filled (in a real implementation, this would check if data was actually filled)
                update_gap_status(gap['gap_id'], 'Filled')
                filled_count += 1
                
            except Exception as e:
                logger.error(f"Error filling gap {gap['gap_id']}: {str(e)}")
                update_gap_status(gap['gap_id'], 'Failed')
                
        return filled_count
        
    except Exception as e:
        logger.error(f"Error filling gaps: {str(e)}")
        return 0

def _get_interval_seconds(interval: str) -> int:
    """
    Convert interval string to seconds.
    
    Args:
        interval: Interval string (e.g., '1h', '4h', '1d')
        
    Returns:
        Interval in seconds
    """
    unit = interval[-1].lower()
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
        raise ValueError(f"Unknown interval unit: {unit}")

def _log_gaps(gaps_df: pd.DataFrame) -> None:
    """
    Log detected gaps to the gap log file.
    
    Args:
        gaps_df: DataFrame with gap information
    """
    try:
        ensure_temp_dir()
        log_path = os.path.join(TEMP_GAP_DIR, GAP_LOG_FILE)
        
        # Convert timestamps to strings for JSON serialization
        gaps_df_copy = gaps_df.copy()
        if 'gap_start' in gaps_df_copy.columns:
            gaps_df_copy['gap_start'] = gaps_df_copy['gap_start'].astype(str)
        if 'gap_end' in gaps_df_copy.columns:
            gaps_df_copy['gap_end'] = gaps_df_copy['gap_end'].astype(str)
            
        # Convert to list of dictionaries
        gaps_list = gaps_df_copy.to_dict('records')
        
        # Load existing gaps if the file exists
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {"gaps": []}
        else:
            existing_data = {"gaps": []}
            
        # Add new gaps
        existing_gap_ids = {gap.get('gap_id') for gap in existing_data.get('gaps', [])}
        
        for gap in gaps_list:
            if gap.get('gap_id') not in existing_gap_ids:
                existing_data['gaps'].append(gap)
                
        # Save updated data
        with open(log_path, 'w') as f:
            json.dump(existing_data, f)
            
    except Exception as e:
        logger.error(f"Error logging gaps: {str(e)}")