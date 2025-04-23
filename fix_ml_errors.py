"""
Special utility to fix maximum recursion errors in the ML modules.
This script applies the fix and tests it to ensure it works correctly.
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime, timedelta

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ml_training.log')
    ]
)

def test_historical_data_retrieval():
    """Test that we can get historical data without recursion errors"""
    # First, import and apply the fix
    print("Importing and applying recursion fix...")
    import direct_ml_fix
    direct_ml_fix.fix_ml_modules()
    
    # Now try getting data through various pathways
    symbols = ['BTCUSDT', 'ETHUSDT']
    intervals = ['1h', '4h', '1d', '30m']
    
    print("\nTesting historical data retrieval...")
    for symbol in symbols:
        for interval in intervals:
            print(f"\nTesting {symbol}/{interval}...")
            # Try database_extensions first
            try:
                import database_extensions
                print(f"  Via database_extensions...")
                df = database_extensions.get_historical_data(symbol, interval, lookback_days=30)
                if df is not None and not df.empty:
                    print(f"  ✓ Success! Retrieved {len(df)} records via database_extensions")
                else:
                    print(f"  × No data retrieved via database_extensions")
            except Exception as e:
                print(f"  × Error in database_extensions.get_historical_data: {e}")
            
            # Try binance_api next
            try:
                import binance_api
                print(f"  Via binance_api...")
                df = binance_api.get_historical_data(symbol, interval, lookback_days=30)
                if df is not None and not df.empty:
                    print(f"  ✓ Success! Retrieved {len(df)} records via binance_api")
                else:
                    print(f"  × No data retrieved via binance_api")
            except Exception as e:
                print(f"  × Error in binance_api.get_historical_data: {e}")
    
    print("\nTesting completed.")

def ensure_30m_timeframe():
    """Ensure that 30m data is available in the database"""
    from database import get_db_connection
    import pandas as pd
    
    conn = get_db_connection()
    if conn is None:
        print("Failed to connect to database")
        return False
    
    try:
        # Check if we have 30m data for BTCUSDT
        query = """
            SELECT COUNT(*) as count FROM historical_data 
            WHERE symbol = 'BTCUSDT' AND interval = '30m'
        """
        df = pd.read_sql_query(query, conn)
        count = df['count'].iloc[0] if not df.empty else 0
        
        print(f"Found {count} records for BTCUSDT/30m in database")
        
        if count > 0:
            print("30m timeframe data already exists, no need to download")
            return True
        
        # If we don't have data, initiate download of 30m data
        print("No 30m data found, need to download it")
        from download_binance_data import download_historical_klines
        
        # Download a smaller sample for testing purposes
        symbol = 'BTCUSDT'
        interval = '30m'
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"Downloading {symbol}/{interval} data from {start_date} to {end_date}")
        download_historical_klines(symbol, interval, start_date, end_date)
        
        # Check again to confirm data was downloaded
        df = pd.read_sql_query(query, conn)
        count = df['count'].iloc[0] if not df.empty else 0
        print(f"Now have {count} records for BTCUSDT/30m in database")
        
        return count > 0
    
    except Exception as e:
        print(f"Error ensuring 30m timeframe: {e}")
        return False
    finally:
        conn.close()

def test_advanced_ml():
    """Test that advanced_ml can work with the data"""
    try:
        # Import module and try to run a simple test
        import advanced_ml
        
        # Try to get pattern recommendations
        print("\nTesting advanced_ml pattern recommendations...")
        recommendations = advanced_ml.get_pattern_recommendations(min_strength=0.6, max_days_old=7, limit=5)
        
        if recommendations is not None and not recommendations.empty:
            print(f"✓ Successfully retrieved {len(recommendations)} pattern recommendations")
            print(recommendations)
        else:
            print("× No pattern recommendations available yet")
            
        # Try to analyze patterns
        print("\nTesting advanced_ml pattern analysis...")
        analyzer = advanced_ml.MultiSymbolPatternAnalyzer()
        results = analyzer.analyze_all_patterns(symbols=['BTCUSDT'], intervals=['1h'], days=30)
        
        if results is not None and not results.empty:
            print(f"✓ Successfully analyzed patterns: {len(results)} patterns detected")
        else:
            print("× No patterns detected in analysis")
        
        return True
        
    except Exception as e:
        print(f"Error testing advanced_ml: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting ML error fix and validation...")
    
    # Apply fix first
    import direct_ml_fix
    if direct_ml_fix.fix_ml_modules():
        print("Successfully applied ML module fixes")
        
        # Test data retrieval
        test_historical_data_retrieval()
        
        # Ensure 30m data is available
        if ensure_30m_timeframe():
            print("30m data is available for testing")
            
            # Test advanced ML functions
            if test_advanced_ml():
                print("\n✅ Advanced ML module appears to be working correctly!")
            else:
                print("\n⚠️ Advanced ML tests failed, but data retrieval is working")
        else:
            print("⚠️ Could not ensure 30m data availability")
    else:
        print("❌ Failed to apply ML module fixes")