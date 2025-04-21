#!/usr/bin/env python3
"""
Gap statistics module for cryptocurrency data.

This module analyzes and reports on gaps in cryptocurrency time series data.
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
GAP_FILLER_LOG = "gap_filler.log"


def get_gap_stats() -> Optional[Dict]:
    """
    Retrieve statistics about detected and filled gaps from the log file.
    
    Returns:
        Dictionary with gap statistics or None if no data is available
    """
    try:
        if not os.path.exists(GAP_FILLER_LOG):
            return None
            
        # Read the log file
        with open(GAP_FILLER_LOG, 'r') as f:
            log_content = f.readlines()
        
        # Initialize stats
        stats = {
            'total_gaps': 0,
            'filled_gaps': 0,
            'failed_gaps': 0,
            'gap_details': []
        }
        
        # Process log to extract information
        for line in log_content:
            if 'Detected gap in' in line:
                # Extract gap information
                parts = line.split(' - ')
                if len(parts) >= 2:
                    info = parts[1]
                    
                    # Extract symbol, interval and dates
                    if 'Detected gap in ' in info:
                        gap_info = info.split('Detected gap in ')[1]
                        symbol_interval = gap_info.split(' from ')[0]
                        dates = gap_info.split(' from ')[1].split(' to ')
                        
                        if len(dates) >= 2:
                            start_date = dates[0]
                            end_date = dates[1]
                            
                            # Add to the details list
                            stats['gap_details'].append({
                                'Symbol/Interval': symbol_interval,
                                'Start Date': start_date,
                                'End Date': end_date,
                                'Status': 'Detected'
                            })
                            
                            stats['total_gaps'] += 1
            
            elif 'Successfully filled gap' in line:
                stats['filled_gaps'] += 1
                
                # Update the status of the corresponding gap in details
                parts = line.split(' - ')
                if len(parts) >= 2:
                    info = parts[1]
                    if 'Successfully filled gap in ' in info:
                        gap_info = info.split('Successfully filled gap in ')[1]
                        symbol_interval = gap_info.split(' from ')[0]
                        dates = gap_info.split(' from ')[1].split(' to ')
                        
                        if len(dates) >= 2:
                            start_date = dates[0]
                            end_date = dates[1]
                            
                            # Update status in gap_details
                            for detail in stats['gap_details']:
                                if (detail['Symbol/Interval'] == symbol_interval and
                                    detail['Start Date'] == start_date and
                                    detail['End Date'] == end_date):
                                    detail['Status'] = 'Filled'
            
            elif 'Failed to fill gap' in line:
                stats['failed_gaps'] += 1
                
                # Update the status of the corresponding gap in details
                parts = line.split(' - ')
                if len(parts) >= 2:
                    info = parts[1]
                    if 'Failed to fill gap in ' in info:
                        gap_info = info.split('Failed to fill gap in ')[1]
                        symbol_interval = gap_info.split(' from ')[0]
                        dates = gap_info.split(' from ')[1].split(' to ')
                        
                        if len(dates) >= 2:
                            start_date = dates[0]
                            end_date = dates[1]
                            
                            # Update status in gap_details
                            for detail in stats['gap_details']:
                                if (detail['Symbol/Interval'] == symbol_interval and
                                    detail['Start Date'] == start_date and
                                    detail['End Date'] == end_date):
                                    detail['Status'] = 'Failed'
        
        # Convert gap_details to DataFrame for display
        if stats['gap_details']:
            stats['gap_details'] = pd.DataFrame(stats['gap_details'])
            
        return stats
    
    except Exception as e:
        logging.error(f"Error getting gap statistics: {e}")
        return None


def clear_gap_log() -> bool:
    """
    Clear the gap filler log file.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(GAP_FILLER_LOG, "w") as f:
            f.write("")  # Clear the file
        return True
    except Exception as e:
        logging.error(f"Error clearing gap log: {e}")
        return False


if __name__ == "__main__":
    # Get gap statistics
    stats = get_gap_stats()
    
    if stats:
        print(f"\nGap Statistics:")
        print(f"Total gaps detected: {stats['total_gaps']}")
        print(f"Successfully filled: {stats['filled_gaps']}")
        print(f"Failed to fill: {stats['failed_gaps']}")
        
        if 'gap_details' in stats and not stats['gap_details'].empty:
            print("\nGap Details:")
            print(stats['gap_details'])
    else:
        print("No gap statistics available.")