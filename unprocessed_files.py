#!/usr/bin/env python3
"""
Helper module to manage and report unprocessed files and gaps in data.
"""

import os
import pandas as pd
import logging
from typing import Dict, Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
UNPROCESSED_FILES_LOG = "unprocessed_files.log"


def log_unprocessed_file(symbol: str, interval: str, year: int, month: int, reason: str) -> None:
    """
    Log an unprocessed file to the tracking log.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '15m', '1h', '4h', '1d')
        year: Year of the file
        month: Month of the file
        reason: Reason why the file could not be processed
    """
    try:
        with open(UNPROCESSED_FILES_LOG, "a") as f:
            f.write(f"{symbol}/{interval}/{year}-{month:02d}: {reason}\n")
    except Exception as e:
        logging.error(f"Error logging unprocessed file: {e}")


def get_unprocessed_files_stats() -> Optional[Dict]:
    """
    Get statistics about unprocessed files from the log.
    
    Returns:
        Dictionary with statistics or None if no data is available
    """
    try:
        if not os.path.exists(UNPROCESSED_FILES_LOG) or os.path.getsize(UNPROCESSED_FILES_LOG) == 0:
            return None
            
        # Read the log file
        with open(UNPROCESSED_FILES_LOG, "r") as f:
            unprocessed_files = f.readlines()
        
        # Count by symbol and interval
        symbol_counts = {}
        interval_counts = {}
        reasons = {}
        
        # File details for the table
        file_details = []
        
        for file_entry in unprocessed_files:
            parts = file_entry.strip().split('/')
            if len(parts) >= 2:
                symbol = parts[0]
                interval = parts[1]
                
                # Extract date and reason
                date_reason = parts[2] if len(parts) > 2 else "Unknown"
                
                if ":" in date_reason:
                    date_part = date_reason.split(":")[0].strip()
                    reason_part = date_reason.split(":", 1)[1].strip()
                else:
                    date_part = date_reason
                    reason_part = "Unknown reason"
                
                # Update counts
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
                interval_counts[interval] = interval_counts.get(interval, 0) + 1
                
                # Categorize reasons
                reason_category = "Other"
                
                if "Failed to process ZIP content" in reason_part:
                    reason_category = "ZIP Processing Error"
                elif "Error downloading" in reason_part:
                    reason_category = "Download Error"
                elif "No data available" in reason_part:
                    reason_category = "No Data Available"
                
                reasons[reason_category] = reasons.get(reason_category, 0) + 1
                
                # Add to file details
                file_details.append({
                    "Symbol": symbol,
                    "Interval": interval,
                    "Date": date_part,
                    "Reason": reason_category
                })
        
        # Prepare statistics
        stats = {
            "total_count": len(unprocessed_files),
            "symbol_counts": pd.DataFrame({
                "Symbol": list(symbol_counts.keys()),
                "Count": list(symbol_counts.values())
            }).sort_values("Count", ascending=False),
            "interval_counts": pd.DataFrame({
                "Interval": list(interval_counts.keys()),
                "Count": list(interval_counts.values())
            }).sort_values("Count", ascending=False),
            "reason_counts": pd.DataFrame({
                "Reason": list(reasons.keys()),
                "Count": list(reasons.values())
            }).sort_values("Count", ascending=False),
            "file_details": pd.DataFrame(file_details)
        }
        
        return stats
    
    except Exception as e:
        logging.error(f"Error getting unprocessed files statistics: {e}")
        return None


def clear_unprocessed_files_log() -> bool:
    """
    Clear the unprocessed files log.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(UNPROCESSED_FILES_LOG, "w") as f:
            f.write("")  # Clear the file
        return True
    except Exception as e:
        logging.error(f"Error clearing unprocessed files log: {e}")
        return False


if __name__ == "__main__":
    # Get statistics
    stats = get_unprocessed_files_stats()
    
    if stats:
        print(f"Unprocessed Files Statistics:")
        print(f"Total unprocessed files: {stats['total_count']}")
        
        print("\nBy Symbol:")
        print(stats["symbol_counts"])
        
        print("\nBy Interval:")
        print(stats["interval_counts"])
        
        print("\nBy Reason:")
        print(stats["reason_counts"])
    else:
        print("No unprocessed files found.")