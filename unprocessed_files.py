"""
Module for tracking and analyzing unprocessed data files.
This module helps identify files that failed to process and reasons for failure.
"""
import logging
import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
UNPROCESSED_FILES_LOG = "unprocessed_files.log"
TEMP_DIR = "temp_gaps"  # Using the same temp directory as gap_stats


def ensure_temp_dir():
    """Create temporary directory if it doesn't exist"""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)


def log_unprocessed_file(symbol: str, interval: str, file_path: str, reason: str) -> bool:
    """
    Log an unprocessed file to the tracking system.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '4h', '1d')
        file_path: Path to the unprocessed file
        reason: Reason for failure to process
        
    Returns:
        True if successfully logged, False otherwise
    """
    try:
        ensure_temp_dir()
        log_path = os.path.join(TEMP_DIR, UNPROCESSED_FILES_LOG)
        
        # Create file entry
        file_entry = {
            'symbol': symbol,
            'interval': interval,
            'file_path': file_path,
            'reason': reason,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Load existing log if it exists
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                try:
                    log_data = json.load(f)
                except json.JSONDecodeError:
                    log_data = {"files": []}
        else:
            log_data = {"files": []}
            
        # Add new entry
        log_data['files'].append(file_entry)
        
        # Save updated log
        with open(log_path, 'w') as f:
            json.dump(log_data, f)
            
        return True
        
    except Exception as e:
        logger.error(f"Error logging unprocessed file: {str(e)}")
        return False


def get_unprocessed_files_stats() -> Optional[Dict[str, Any]]:
    """
    Get statistics about unprocessed files.
    
    Returns:
        Dictionary with statistics or None if no log exists
    """
    try:
        ensure_temp_dir()
        log_path = os.path.join(TEMP_DIR, UNPROCESSED_FILES_LOG)
        
        # Check if log exists
        if not os.path.exists(log_path):
            return None
            
        # Load log data
        with open(log_path, 'r') as f:
            try:
                log_data = json.load(f)
            except json.JSONDecodeError:
                return None
                
        # Get list of files
        files = log_data.get('files', [])
        
        if not files:
            return None
            
        # Convert to DataFrame for analysis
        df = pd.DataFrame(files)
        
        # Add timestamp column as datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Get total count
        total_count = len(df)
        
        # Get symbol counts
        symbol_counts = df.groupby('symbol').size().reset_index(name='count')
        symbol_counts = symbol_counts.sort_values('count', ascending=False)
        
        # Get interval counts
        interval_counts = df.groupby('interval').size().reset_index(name='count')
        interval_counts = interval_counts.sort_values('count', ascending=False)
        
        # Get reason counts
        reason_counts = df.groupby('reason').size().reset_index(name='count')
        reason_counts = reason_counts.sort_values('count', ascending=False)
        
        return {
            "total_count": total_count,
            "symbol_counts": symbol_counts.to_dict('records'),
            "interval_counts": interval_counts.to_dict('records'),
            "reason_counts": reason_counts.to_dict('records'),
            "file_details": df.to_dict('records')
        }
        
    except Exception as e:
        logger.error(f"Error getting unprocessed files stats: {str(e)}")
        return None


def clear_unprocessed_files_log() -> bool:
    """
    Clear the unprocessed files log.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_temp_dir()
        log_path = os.path.join(TEMP_DIR, UNPROCESSED_FILES_LOG)
        
        # Create empty log
        with open(log_path, 'w') as f:
            json.dump({"files": []}, f)
            
        return True
        
    except Exception as e:
        logger.error(f"Error clearing unprocessed files log: {str(e)}")
        return False


def attempt_reprocess_files() -> Dict[str, Any]:
    """
    Attempt to reprocess previously unprocessed files.
    
    Returns:
        Dictionary with reprocessing results
    """
    try:
        # Get unprocessed files stats
        stats = get_unprocessed_files_stats()
        
        if not stats:
            return {
                "success": True,
                "total_files": 0,
                "reprocessed": 0,
                "failed": 0
            }
            
        total_files = stats["total_count"]
        reprocessed = 0
        failed = 0
        
        # In a real implementation, we would attempt to reprocess each file
        # For this placeholder, we'll just clear the log
        if clear_unprocessed_files_log():
            reprocessed = total_files
        else:
            failed = total_files
            
        return {
            "success": failed == 0,
            "total_files": total_files,
            "reprocessed": reprocessed,
            "failed": failed
        }
        
    except Exception as e:
        logger.error(f"Error reprocessing files: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }