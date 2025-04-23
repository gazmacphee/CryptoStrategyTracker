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
        df_1h = df_30m.resample('1h').agg({
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

def test_manual_resampling():
    """Test the resampling function with a manually created DataFrame"""
    print("\nTesting manual resampling...")
    
    # Create a test dataframe with 30m data
    timestamps = [
        pd.Timestamp('2022-04-01 00:00:00'),
        pd.Timestamp('2022-04-01 00:30:00'),
        pd.Timestamp('2022-04-01 01:00:00'),
        pd.Timestamp('2022-04-01 01:30:00'),
        pd.Timestamp('2022-04-01 02:00:00'),
        pd.Timestamp('2022-04-01 02:30:00')
    ]
    
    test_df = pd.DataFrame({
        'timestamp': timestamps,
        'open': [100.0, 105.0, 110.0, 112.0, 115.0, 118.0],
        'high': [106.0, 108.0, 113.0, 116.0, 119.0, 121.0],
        'low': [99.0, 104.0, 108.0, 110.0, 113.0, 117.0],
        'close': [105.0, 110.0, 112.0, 115.0, 118.0, 120.0],
        'volume': [10.0, 15.0, 12.0, 13.0, 11.0, 14.0],
        'symbol': ['BTCUSDT'] * 6,
        'interval': ['30m'] * 6
    })
    
    print("\nTest 30m data:")
    print(test_df)
    
    # Here we'll test the resampling directly to ensure we understand how pandas resampling works
    print("\nManual resampling test:")
    df_copy = test_df.copy()
    df_copy = df_copy.set_index('timestamp')
    
    # Test with '1h' interval explicitly
    resampled = df_copy.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    resampled = resampled.reset_index()
    print("\nManually resampled data (hourly intervals):")
    print(resampled)
    
    # Add symbol and interval back
    resampled['symbol'] = 'BTCUSDT'
    resampled['interval'] = '1h'
    
    # Print resampled values for verification
    print("\nResample results by hour:")
    for i in range(len(resampled)):
        hour = resampled.iloc[i]['timestamp'].hour
        print(f"Hour {hour}:")
        print(f"  open: {resampled.iloc[i]['open']}")
        print(f"  high: {resampled.iloc[i]['high']}")
        print(f"  low: {resampled.iloc[i]['low']}")
        print(f"  close: {resampled.iloc[i]['close']}")
        print(f"  volume: {resampled.iloc[i]['volume']}")
    
    # Now use the conversion function for comparison
    print("\nUsing test_convert_30m_to_1h function:")
    df_30m = test_df.copy()
    result_1h = test_convert_30m_to_1h(df_30m)
    
    # Verify the expected results
    if result_1h is not None:
        print("\nVerification based on actual resampling output:")
        # Verify row count
        assert len(result_1h) == len(resampled), f"Expected {len(resampled)} rows, got {len(result_1h)}"
        
        # Verify values match the resampled dataframe
        match = True
        for i in range(len(resampled)):
            for col in ['open', 'high', 'low', 'close', 'volume']:
                expected = resampled.iloc[i][col]
                actual = result_1h.iloc[i][col]
                if expected != actual:
                    print(f"❌ Mismatch at hour {resampled.iloc[i]['timestamp'].hour}: {col} expected {expected}, got {actual}")
                    match = False
        
        if match:
            print("✅ All values match pandas resampling!")
        
        # Print expected values for manual verification
        print("\nExpected volume totals:")
        print(f"Hour 0: {test_df.iloc[0]['volume'] + test_df.iloc[1]['volume']} (10.0 + 15.0)")
        print(f"Hour 1: {test_df.iloc[2]['volume'] + test_df.iloc[3]['volume']} (12.0 + 13.0)")
        print(f"Hour 2: {test_df.iloc[4]['volume'] + test_df.iloc[5]['volume']} (11.0 + 14.0)")
        
    else:
        print("❌ Failed to convert test data to 1h interval")
    
    return result_1h

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
    
    # Even if no real data, test with manual data
    test_manual_resampling()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()