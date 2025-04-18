"""
Helper module to fetch and parse file listings from Binance Data Vision.
This provides functions to determine available files for a given symbol and interval.
"""

import os
import requests
import logging
import re
import time
from datetime import datetime
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('binance_data_listing.log')
    ]
)

# Constants
BASE_URL = "https://data.binance.vision"
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

def fetch_directory_listing(url: str) -> Optional[str]:
    """
    Fetch the XML content of a directory listing from Binance Data Vision.
    Returns the XML content as a string or None if the request fails.
    """
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                return response.text
            
            logging.warning(f"Failed to fetch directory listing: {url}, status code: {response.status_code}")
            return None
        except Exception as e:
            retries += 1
            logging.warning(f"Error fetching directory listing (attempt {retries}/{MAX_RETRIES}): {e}")
            time.sleep(RETRY_DELAY)
    
    logging.error(f"Failed to fetch directory listing after {MAX_RETRIES} attempts: {url}")
    return None

def parse_xml_listing(xml_content: str) -> List[Dict]:
    """
    Parse an XML directory listing and extract file/directory information.
    Returns a list of dictionaries with keys: name, size, type, last_modified
    """
    if not xml_content:
        logging.error("Empty XML content provided to parse_xml_listing")
        return []
        
    try:
        # Parse XML content
        root = ET.fromstring(xml_content)
        namespace = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        
        # Extract items
        items = []
        for content in root.findall('.//s3:Contents', namespace):
            key_element = content.find('s3:Key', namespace)
            size_element = content.find('s3:Size', namespace) 
            last_modified_element = content.find('s3:LastModified', namespace)
            
            # Skip items with missing elements
            if not key_element or not size_element or not last_modified_element:
                continue
                
            # Get text values, defaulting to empty string if None
            key = key_element.text or ""
            size_text = size_element.text or "0"
            last_modified = last_modified_element.text or ""
            
            try:
                size = int(size_text)
            except (ValueError, TypeError):
                size = 0
                
            # Convert to readable format (safely)
            name = os.path.basename(key) if key else ""
            path = os.path.dirname(key) if key else ""
            is_directory = name == '' or name.endswith('/')
            
            items.append({
                'name': name,
                'path': path,
                'size': size,
                'type': 'directory' if is_directory else 'file',
                'last_modified': last_modified,
                'full_path': key
            })
        
        return items
    except Exception as e:
        logging.error(f"Error parsing XML listing: {e}")
        return []

def get_available_kline_files(symbol: str, interval: str, file_type: str = 'monthly') -> List[Dict]:
    """
    Get a list of available kline data files for a given symbol and interval.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '4h', '1d')
        file_type: Type of files to fetch ('monthly' or 'daily')
    
    Returns:
        List of dictionaries with file information
    """
    if file_type not in ['monthly', 'daily']:
        logging.error(f"Invalid file_type: {file_type}. Must be 'monthly' or 'daily'")
        return []
    
    # Build URL for directory listing
    url = f"{BASE_URL}/?prefix=data/spot/{file_type}/klines/{symbol}/{interval}/"
    logging.info(f"Fetching {file_type} kline file listing for {symbol}/{interval}")
    
    # Fetch directory listing
    xml_content = fetch_directory_listing(url)
    if not xml_content:
        logging.warning(f"No directory listing available for {symbol}/{interval} {file_type} klines")
        return []
    
    # Parse XML listing
    items = parse_xml_listing(xml_content)
    
    # Filter for zip files only
    zip_files = [item for item in items if item['name'].endswith('.zip')]
    
    if not zip_files:
        logging.warning(f"No zip files found for {symbol}/{interval} {file_type} klines")
    else:
        logging.info(f"Found {len(zip_files)} {file_type} kline files for {symbol}/{interval}")
    
    return zip_files

def extract_date_from_filename(filename: str) -> Optional[Tuple[int, int, Optional[int]]]:
    """
    Extract year, month, and day (if present) from a kline filename.
    Returns a tuple of (year, month, day) or None if the pattern doesn't match.
    For monthly files, day will be None.
    """
    # Pattern for monthly files: SYMBOL-INTERVAL-YYYY-MM.zip
    monthly_pattern = r'.*-(\d{4})-(\d{2})\.zip$'
    
    # Pattern for daily files: SYMBOL-INTERVAL-YYYY-MM-DD.zip
    daily_pattern = r'.*-(\d{4})-(\d{2})-(\d{2})\.zip$'
    
    # Try monthly pattern first
    monthly_match = re.match(monthly_pattern, filename)
    if monthly_match:
        year, month = map(int, monthly_match.groups())
        return (year, month, None)
    
    # Try daily pattern
    daily_match = re.match(daily_pattern, filename)
    if daily_match:
        year, month, day = map(int, daily_match.groups())
        return (year, month, day)
    
    return None

def get_date_range_for_symbol_interval(symbol: str, interval: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Get the earliest and latest dates available for a given symbol and interval.
    Returns a tuple of (min_date, max_date) or (None, None) if no files are found.
    """
    # First try monthly files
    monthly_files = get_available_kline_files(symbol, interval, 'monthly')
    
    # Extract dates from filenames
    dates = []
    for file_info in monthly_files:
        date_parts = extract_date_from_filename(file_info['name'])
        if date_parts:
            year, month, _ = date_parts
            file_date = datetime(year, month, 1)
            dates.append(file_date)
    
    # If no monthly files, try daily files
    if not dates:
        daily_files = get_available_kline_files(symbol, interval, 'daily')
        for file_info in daily_files:
            date_parts = extract_date_from_filename(file_info['name'])
            if date_parts:
                year, month, day = date_parts
                day = day or 1  # Default to first day if day is None
                file_date = datetime(year, month, day)
                dates.append(file_date)
    
    if not dates:
        logging.warning(f"No date information available for {symbol}/{interval}")
        return (None, None)
    
    min_date = min(dates)
    max_date = max(dates)
    
    logging.info(f"Date range for {symbol}/{interval}: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    return (min_date, max_date)

if __name__ == "__main__":
    # Example usage
    symbol = "BTCUSDT"
    interval = "1h"
    
    print(f"Checking available files for {symbol}/{interval}...")
    min_date, max_date = get_date_range_for_symbol_interval(symbol, interval)
    
    if min_date and max_date:
        print(f"Available data from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        
        # Get all monthly files
        monthly_files = get_available_kline_files(symbol, interval, 'monthly')
        print(f"Found {len(monthly_files)} monthly files")
        
        # Show sample file names
        for file in monthly_files[:5]:
            print(f"  - {file['name']} ({file['size']} bytes)")
        if len(monthly_files) > 5:
            print(f"  - ... and {len(monthly_files) - 5} more files")
    else:
        print(f"No data available for {symbol}/{interval}")