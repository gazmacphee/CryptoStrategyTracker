import pandas as pd
from datetime import datetime, timedelta
from database import get_db_connection, get_indicators, get_historical_data

def test_indicators():
    """Test if indicators are being retrieved correctly"""
    # Date range for the last 30 days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    # Get indicators for BTCUSDT on 1h interval
    symbol = "BTCUSDT"
    interval = "1h"
    
    print(f"Testing indicator retrieval for {symbol} ({interval}) from {start_time} to {end_time}")
    
    # Get indicators
    indicators_df = get_indicators(symbol, interval, start_time, end_time)
    
    print(f"Retrieved {len(indicators_df)} indicator records")
    
    if not indicators_df.empty:
        print("First 3 indicator rows:")
        print(indicators_df.head(3))
        
        # Check if we have buy/sell signals
        if 'buy_signal' in indicators_df.columns:
            buy_signals = indicators_df['buy_signal'].sum()
            sell_signals = indicators_df['sell_signal'].sum()
            print(f"Total buy signals: {buy_signals}")
            print(f"Total sell signals: {sell_signals}")
    else:
        print("No indicators found")
    
    # Get historical data
    prices_df = get_historical_data(symbol, interval, start_time, end_time)
    
    print(f"Retrieved {len(prices_df)} price records")
    
    if not prices_df.empty:
        print("First 3 price rows:")
        print(prices_df.head(3))
    else:
        print("No price data found")
    
    # Check how many prices have matching indicators
    if not indicators_df.empty and not prices_df.empty:
        matching_times = set(indicators_df['timestamp']) & set(prices_df['timestamp'])
        print(f"{len(matching_times)} timestamps have both price and indicator data")

if __name__ == "__main__":
    test_indicators()