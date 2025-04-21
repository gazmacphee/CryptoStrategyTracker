#!/usr/bin/env python3
"""
Gap Filler module for cryptocurrency data.

This module identifies and fills gaps in historical cryptocurrency data,
ensuring completeness for technical analysis.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, date
import calendar
from typing import List, Tuple, Dict, Optional, Set

# Local imports
from database import (
    get_db_connection, execute_sql_to_df, has_complete_month_data, 
    get_existing_data_months, save_historical_data, save_indicators
)
from download_single_pair import download_and_process
from binance_api import get_klines_data
import indicators
from strategy import evaluate_buy_sell_signals

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gap_filler.log')
    ]
)

# Constants
DEFAULT_INTERVALS = ['15m', '30m', '1h', '4h', '1d']
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT",
    "SOLUSDT", "DOTUSDT", "LTCUSDT", "LINKUSDT", "MATICUSDT", "AVAXUSDT"
]

def identify_data_gaps(symbol: str, interval: str, start_date: datetime, end_date: datetime) -> List[Tuple[datetime, datetime]]:
    """
    Identify gaps in historical data for a specific symbol and interval.
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        interval: Time interval (e.g., "15m", "1h")
        start_date: Start date to check for gaps
        end_date: End date to check for gaps
        
    Returns:
        List of tuples representing the start and end of each gap period
    """
    logging.info(f"Identifying data gaps for {symbol}/{interval} from {start_date} to {end_date}")
    
    # Get existing data from database
    conn = get_db_connection()
    if not conn:
        logging.error("Failed to connect to database")
        return []
    
    try:
        # Get timestamp values at expected interval frequency
        query = """
        SELECT timestamp
        FROM historical_data
        WHERE symbol = %s
        AND interval = %s
        AND timestamp BETWEEN %s AND %s
        ORDER BY timestamp ASC
        """
        
        df = execute_sql_to_df(query, conn, params=(symbol, interval, start_date, end_date))
        
        if df.empty:
            # If no data exists in the entire range, return it as one gap
            logging.warning(f"No data found for {symbol}/{interval} in the specified range")
            return [(start_date, end_date)]
        
        # Convert interval to timedelta
        interval_td = get_interval_timedelta(interval)
        if not interval_td:
            logging.error(f"Invalid interval format: {interval}")
            return []
            
        # Create a continuous range of expected timestamps
        expected_timestamps = []
        current = start_date
        while current <= end_date:
            expected_timestamps.append(current)
            current += interval_td
            
        expected_df = pd.DataFrame(expected_timestamps, columns=['timestamp'])
        
        # Find missing timestamps by comparing expected with actual
        merged = expected_df.merge(df, on='timestamp', how='left', indicator=True)
        missing = merged[merged['_merge'] == 'left_only']['timestamp']
        
        if len(missing) == 0:
            logging.info(f"No gaps found for {symbol}/{interval}")
            return []
            
        # Group consecutive missing timestamps into gaps
        missing_list = sorted(missing.tolist())
        gaps = []
        gap_start = missing_list[0]
        prev_ts = gap_start
        
        for ts in missing_list[1:]:
            # If timestamps are not consecutive, close current gap and start a new one
            if ts - prev_ts > interval_td * 1.5:  # Allow some tolerance
                gaps.append((gap_start, prev_ts))
                gap_start = ts
            prev_ts = ts
            
        # Add the last gap
        gaps.append((gap_start, prev_ts))
        
        # Log the gaps found
        for i, (gap_start, gap_end) in enumerate(gaps):
            logging.info(f"Gap #{i+1}: {gap_start} to {gap_end} ({(gap_end - gap_start).total_seconds() / 3600:.1f} hours)")
            
        return gaps
        
    except Exception as e:
        logging.error(f"Error identifying data gaps: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_interval_timedelta(interval: str) -> Optional[timedelta]:
    """
    Convert interval string to timedelta.
    
    Args:
        interval: Time interval string (e.g., "15m", "1h", "1d")
        
    Returns:
        Equivalent timedelta or None if format is invalid
    """
    try:
        unit = interval[-1]
        value = int(interval[:-1])
        
        if unit == 'm':
            return timedelta(minutes=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'd':
            return timedelta(days=value)
        elif unit == 'w':
            return timedelta(weeks=value)
        else:
            return None
    except:
        return None

def fill_data_gap(symbol: str, interval: str, gap_start: datetime, gap_end: datetime) -> bool:
    """
    Fill a gap in historical data.
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        interval: Time interval (e.g., "15m", "1h")
        gap_start: Start of the gap period
        gap_end: End of the gap period
        
    Returns:
        True if gap was successfully filled, False otherwise
    """
    logging.info(f"Filling data gap for {symbol}/{interval} from {gap_start} to {gap_end}")
    
    # If gap spans more than one month, break it down by months
    if gap_start.month != gap_end.month or gap_start.year != gap_end.year:
        success = True
        current_date = gap_start
        
        while current_date <= gap_end:
            year = current_date.year
            month = current_date.month
            
            # Calculate the end of the month
            days_in_month = calendar.monthrange(year, month)[1]
            month_end = datetime(year, month, days_in_month, 23, 59, 59)
            
            # If month_end is beyond gap_end, use gap_end instead
            month_end = min(month_end, gap_end)
            
            # Fill this month's portion of the gap
            month_success = fill_month_data(symbol, interval, year, month)
            success = success and month_success
            
            # Move to first day of next month
            if month == 12:
                current_date = datetime(year + 1, 1, 1)
            else:
                current_date = datetime(year, month + 1, 1)
                
        return success
    else:
        # Gap is within a single month
        return fill_month_data(symbol, interval, gap_start.year, gap_start.month)

def fill_month_data(symbol: str, interval: str, year: int, month: int) -> bool:
    """
    Fill data for a specific month.
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        interval: Time interval (e.g., "15m", "1h")
        year: Year
        month: Month
        
    Returns:
        True if data was successfully filled, False otherwise
    """
    logging.info(f"Filling month data for {symbol}/{interval} for {year}-{month}")
    
    # Check if month already has complete data
    if has_complete_month_data(symbol, interval, year, month):
        logging.info(f"Month {year}-{month} already has complete data for {symbol}/{interval}")
        return True
        
    try:
        # Download data for the month
        result = download_and_process(symbol, interval, year, month)
        
        if result and result.get('downloaded'):
            candles_count = result.get('candles', 0)
            logging.info(f"Successfully downloaded {candles_count} candles for {symbol}/{interval} {year}-{month}")
            
            # Calculate indicators for the month
            time_range = (
                datetime(year, month, 1), 
                datetime(year, month, calendar.monthrange(year, month)[1], 23, 59, 59)
            )
            
            calculate_indicators_for_range(symbol, interval, time_range[0], time_range[1])
            
            return True
        else:
            error = result.get('error', 'Unknown error') if result else 'Failed to download data'
            logging.error(f"Failed to fill month data: {error}")
            return False
            
    except Exception as e:
        logging.error(f"Error filling month data: {e}")
        return False

def calculate_indicators_for_range(symbol: str, interval: str, start_time: datetime, end_time: datetime) -> bool:
    """
    Calculate technical indicators for a specific date range.
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        interval: Time interval (e.g., "15m", "1h")
        start_time: Start of the range
        end_time: End of the range
        
    Returns:
        True if indicators were successfully calculated, False otherwise
    """
    logging.info(f"Calculating indicators for {symbol}/{interval} from {start_time} to {end_time}")
    
    conn = get_db_connection()
    if not conn:
        logging.error("Failed to connect to database")
        return False
        
    try:
        # Get historical data for the range
        query = """
        SELECT timestamp, open, high, low, close, volume
        FROM historical_data
        WHERE symbol = %s
        AND interval = %s
        AND timestamp BETWEEN %s AND %s
        ORDER BY timestamp ASC
        """
        
        df = execute_sql_to_df(query, conn, params=(symbol, interval, start_time, end_time))
        
        if df.empty:
            logging.warning(f"No historical data found for calculating indicators")
            return False
            
        # Calculate indicators
        df = indicators.add_bollinger_bands(df)
        df = indicators.add_rsi(df)
        df = indicators.add_macd(df)
        df = indicators.add_ema(df)
        df = indicators.add_stochastic(df)
        
        # Calculate trading signals
        df = evaluate_buy_sell_signals(df)
        
        # Save indicators to database
        result = save_indicators(df, symbol, interval)
        
        if result:
            logging.info(f"Successfully calculated and saved indicators for {len(df)} data points")
            return True
        else:
            logging.error("Failed to save indicators to database")
            return False
            
    except Exception as e:
        logging.error(f"Error calculating indicators: {e}")
        return False
    finally:
        if conn:
            conn.close()

def run_gap_analysis(symbols: List[str]=None, intervals: List[str]=None, 
                   lookback_days: int=90, max_gaps: int=10) -> Dict:
    """
    Run a full gap analysis for multiple symbols and intervals.
    
    Args:
        symbols: List of symbols to check, uses DEFAULT_SYMBOLS if None
        intervals: List of intervals to check, uses DEFAULT_INTERVALS if None
        lookback_days: Number of days to look back for gaps
        max_gaps: Maximum number of gaps to fill per symbol/interval
        
    Returns:
        Dictionary with results of the gap analysis
    """
    symbols = symbols or DEFAULT_SYMBOLS
    intervals = intervals or DEFAULT_INTERVALS
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    results = {
        'total_gaps': 0,
        'filled_gaps': 0,
        'failed_gaps': 0,
        'details': {}
    }
    
    for symbol in symbols:
        results['details'][symbol] = {}
        
        for interval in intervals:
            logging.info(f"Running gap analysis for {symbol}/{interval}")
            
            # Identify gaps
            gaps = identify_data_gaps(symbol, interval, start_date, end_date)
            
            # Limit to max_gaps
            if len(gaps) > max_gaps:
                logging.warning(f"Found {len(gaps)} gaps, limiting to {max_gaps}")
                gaps = gaps[:max_gaps]
                
            results['total_gaps'] += len(gaps)
            
            # Store details
            interval_results = {
                'gaps_found': len(gaps),
                'gaps_filled': 0,
                'gaps_failed': 0,
                'gap_details': []
            }
            
            # Fill each gap
            for gap_start, gap_end in gaps:
                gap_info = {
                    'start': gap_start.isoformat(),
                    'end': gap_end.isoformat(),
                    'duration_hours': (gap_end - gap_start).total_seconds() / 3600,
                    'filled': False
                }
                
                success = fill_data_gap(symbol, interval, gap_start, gap_end)
                
                gap_info['filled'] = success
                interval_results['gap_details'].append(gap_info)
                
                if success:
                    interval_results['gaps_filled'] += 1
                    results['filled_gaps'] += 1
                else:
                    interval_results['gaps_failed'] += 1
                    results['failed_gaps'] += 1
            
            results['details'][symbol][interval] = interval_results
            
    logging.info(f"Gap analysis complete. Found {results['total_gaps']} gaps, filled {results['filled_gaps']}, failed {results['failed_gaps']}")
    return results

def check_technical_indicators_completeness(symbols: List[str]=None, intervals: List[str]=None,
                                          lookback_days: int=90) -> Dict:
    """
    Check if technical indicators are complete for all data points.
    
    Args:
        symbols: List of symbols to check
        intervals: List of intervals to check
        lookback_days: Number of days to look back
        
    Returns:
        Dictionary with details of missing indicators
    """
    symbols = symbols or DEFAULT_SYMBOLS
    intervals = intervals or DEFAULT_INTERVALS
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    results = {
        'total_missing': 0,
        'details': {}
    }
    
    for symbol in symbols:
        results['details'][symbol] = {}
        
        for interval in intervals:
            logging.info(f"Checking indicators for {symbol}/{interval}")
            
            conn = get_db_connection()
            if not conn:
                logging.error("Failed to connect to database")
                continue
                
            try:
                # Get count of historical data points
                hist_query = """
                SELECT COUNT(*) as hist_count
                FROM historical_data
                WHERE symbol = %s
                AND interval = %s
                AND timestamp BETWEEN %s AND %s
                """
                
                hist_df = execute_sql_to_df(hist_query, conn, params=(symbol, interval, start_date, end_date))
                hist_count = hist_df.iloc[0]['hist_count'] if not hist_df.empty else 0
                
                # Get count of indicator data points
                ind_query = """
                SELECT COUNT(*) as ind_count
                FROM technical_indicators
                WHERE symbol = %s
                AND interval = %s
                AND timestamp BETWEEN %s AND %s
                """
                
                ind_df = execute_sql_to_df(ind_query, conn, params=(symbol, interval, start_date, end_date))
                ind_count = ind_df.iloc[0]['ind_count'] if not ind_df.empty else 0
                
                # Calculate missing indicators
                missing = hist_count - ind_count
                if missing > 0:
                    results['total_missing'] += missing
                    
                # Store details
                results['details'][symbol][interval] = {
                    'historical_points': int(hist_count),
                    'indicator_points': int(ind_count),
                    'missing': int(missing),
                    'completeness_pct': round(100 * ind_count / max(1, hist_count), 2)
                }
                
                # If indicators are missing, calculate them
                if missing > 0:
                    logging.info(f"Calculating {missing} missing indicators for {symbol}/{interval}")
                    calculate_indicators_for_range(symbol, interval, start_date, end_date)
                    
            except Exception as e:
                logging.error(f"Error checking indicators: {e}")
            finally:
                if conn:
                    conn.close()
                    
    logging.info(f"Indicator completeness check complete. Found {results['total_missing']} missing indicators.")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fill gaps in cryptocurrency data")
    parser.add_argument("--symbol", help="Single symbol to check (default: all)")
    parser.add_argument("--interval", help="Single interval to check (default: all)")
    parser.add_argument("--lookback", type=int, default=90, help="Lookback days")
    parser.add_argument("--max-gaps", type=int, default=10, help="Maximum gaps to fill per pair")
    parser.add_argument("--indicators-only", action="store_true", help="Only check/fill indicators")
    
    args = parser.parse_args()
    
    symbols = [args.symbol] if args.symbol else DEFAULT_SYMBOLS
    intervals = [args.interval] if args.interval else DEFAULT_INTERVALS
    
    print(f"=== Cryptocurrency Data Gap Filler ===")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Intervals: {', '.join(intervals)}")
    print(f"Lookback period: {args.lookback} days")
    print(f"Maximum gaps to fill: {args.max_gaps}")
    print("")
    
    if args.indicators_only:
        print("Checking technical indicators completeness...")
        results = check_technical_indicators_completeness(
            symbols=symbols,
            intervals=intervals,
            lookback_days=args.lookback
        )
        
        print("\nResults:")
        print(f"Total missing indicators: {results['total_missing']}")
        
        for symbol in results['details']:
            for interval, details in results['details'][symbol].items():
                print(f"{symbol}/{interval}: {details['missing']} missing indicators " +
                      f"({details['completeness_pct']}% complete)")
    else:
        print("Running full gap analysis...")
        results = run_gap_analysis(
            symbols=symbols,
            intervals=intervals,
            lookback_days=args.lookback,
            max_gaps=args.max_gaps
        )
        
        print("\nResults:")
        print(f"Total gaps found: {results['total_gaps']}")
        print(f"Gaps filled: {results['filled_gaps']}")
        print(f"Gaps failed: {results['failed_gaps']}")
        
        for symbol in results['details']:
            for interval, details in results['details'][symbol].items():
                print(f"{symbol}/{interval}: Found {details['gaps_found']} gaps, " +
                      f"filled {details['gaps_filled']}, failed {details['gaps_failed']}")
                
        # After filling gaps, also check indicators
        print("\nChecking technical indicators completeness...")
        check_technical_indicators_completeness(
            symbols=symbols,
            intervals=intervals,
            lookback_days=args.lookback
        )