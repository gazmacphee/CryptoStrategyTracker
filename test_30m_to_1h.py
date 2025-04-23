"""
Test script to verify the enhanced 30m to 1h conversion functionality
"""

import os
import sys
import pandas as pd
import datetime

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
import database
import database_extensions
from database_extensions import ensure_float_df

def test_get_30m_data():
    """Test retrieving 30m data directly"""
    print("Testing direct 30m data retrieval...")
    
    symbol = "BTCUSDT"
    interval = "30m"
    
    # Get connection
    conn = database.get_db_connection()
    if conn is None:
        print("Failed to connect to database")
        return
        
    try:
        cursor = conn.cursor()
        check_query = "SELECT COUNT(*) FROM historical_data WHERE symbol = %s AND interval = %s"
        cursor.execute(check_query, (symbol, interval))
        count = cursor.fetchone()[0]
        print(f"Found {count} records for {symbol}/{interval}")
        
        if count > 0:
            # Get the data
            query = "SELECT * FROM historical_data WHERE symbol = %s AND interval = %s ORDER BY timestamp ASC"
            cursor.execute(query, (symbol, interval))
            
            # Get column names
            columns = [desc[0] for desc in cursor.description]
            print(f"Columns: {columns}")
            
            # Fetch rows
            rows = cursor.fetchall()
            print(f"Retrieved {len(rows)} rows")
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=columns)
            
            # Convert to float
            df = ensure_float_df(df)
            
            # Print some data
            print("\nSample data:")
            print(df.head())
            
            # Print stats
            print("\nData range:")
            print(f"Min timestamp: {df['timestamp'].min()}")
            print(f"Max timestamp: {df['timestamp'].max()}")
            
            return df
        else:
            print("No 30m data found")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())
        return None
    finally:
        conn.close()

def test_convert_30m_to_1h(df_30m):
    """Test converting 30m data to 1h"""
    print("\nTesting 30m to 1h conversion...")
    
    if df_30m is None or df_30m.empty:
        print("No 30m data to convert")
        return None
        
    try:
        # Convert to datetime
        if 'timestamp' in df_30m.columns:
            df_30m['timestamp'] = pd.to_datetime(df_30m['timestamp'])
        
        # Set timestamp as index for resampling
        df_30m = df_30m.set_index('timestamp')
        
        # Resample to 1h
        df_1h = df_30m.resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Reset index to get timestamp as column again
        df_1h = df_1h.reset_index()
        
        # Add back other columns
        if 'symbol' in df_30m.columns:
            df_1h['symbol'] = df_30m['symbol'].iloc[0]
            
        df_1h['interval'] = '1h'
        
        # Print result
        print("\nConverted 1h data:")
        print(df_1h.head())
        
        # Print stats
        print("\nConverted data range:")
        print(f"Min timestamp: {df_1h['timestamp'].min()}")
        print(f"Max timestamp: {df_1h['timestamp'].max()}")
        print(f"Total rows: {len(df_1h)}")
        
        return df_1h
    except Exception as e:
        print(f"Error converting data: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def test_using_extensions():
    """Test using the database_extensions module"""
    print("\nTesting database_extensions.get_historical_data with 1h interval...")
    
    # Use the enhanced get_historical_data function
    df = database_extensions.get_historical_data("BTCUSDT", "1h", lookback_days=30)
    
    if df is not None and not df.empty:
        print(f"Successfully retrieved {len(df)} rows using enhanced get_historical_data")
        print("\nSample data:")
        print(df.head())
        return df
    else:
        print("Failed to retrieve data using enhanced get_historical_data")
        return None

def main():
    """Main test function"""
    print("=== Testing 30m to 1h Conversion Functionality ===")
    
    # Get 30m data
    df_30m = test_get_30m_data()
    
    # Convert 30m to 1h
    if df_30m is not None and not df_30m.empty:
        df_1h = test_convert_30m_to_1h(df_30m)
    
    # Test using extensions
    df_ext = test_using_extensions()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()