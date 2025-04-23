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
    
    # Convert to 1h using our function
    df_30m = test_df.copy()
    result_1h = test_convert_30m_to_1h(df_30m)
    
    # Verify the expected results
    if result_1h is not None:
        print("\nVerification:")
        # Verify that the 1h interval has half as many rows as 30m
        assert len(result_1h) == len(test_df) // 2, f"Expected half as many rows ({len(test_df)//2}), got {len(result_1h)}"
        # Check that high values are the max of each hour's 30m values
        assert result_1h.iloc[0]['high'] == 108.0, f"Expected max high for first hour to be 108.0, got {result_1h.iloc[0]['high']}"
        assert result_1h.iloc[1]['high'] == 119.0, f"Expected max high for second hour to be 119.0, got {result_1h.iloc[1]['high']}"
        # Check that low values are the min of each hour's 30m values
        assert result_1h.iloc[0]['low'] == 99.0, f"Expected min low for first hour to be 99.0, got {result_1h.iloc[0]['low']}"
        assert result_1h.iloc[1]['low'] == 110.0, f"Expected min low for second hour to be 110.0, got {result_1h.iloc[1]['low']}"
        # Check that close values are the last of each hour's 30m values
        assert result_1h.iloc[0]['close'] == 110.0, f"Expected close for first hour to be 110.0, got {result_1h.iloc[0]['close']}"
        assert result_1h.iloc[1]['close'] == 118.0, f"Expected close for second hour to be 118.0, got {result_1h.iloc[1]['close']}"
        # Check that open values are the first of each hour's 30m values
        assert result_1h.iloc[0]['open'] == 100.0, f"Expected open for first hour to be 100.0, got {result_1h.iloc[0]['open']}"
        assert result_1h.iloc[1]['open'] == 112.0, f"Expected open for second hour to be 112.0, got {result_1h.iloc[1]['open']}"
        # Check that volume values are the sum of each hour's 30m values
        assert result_1h.iloc[0]['volume'] == 25.0, f"Expected volume for first hour to be 25.0, got {result_1h.iloc[0]['volume']}"
        assert result_1h.iloc[1]['volume'] == 25.0, f"Expected volume for second hour to be 25.0, got {result_1h.iloc[1]['volume']}"
        print("✅ All verification checks passed!")
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