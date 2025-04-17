import pandas as pd
from datetime import datetime, timedelta
import time

# Import local modules
from database import create_tables, get_db_connection, get_historical_data, save_indicators, save_historical_data
from binance_api import get_klines_data
from indicators import add_bollinger_bands, add_rsi, add_macd, add_ema
from strategy import evaluate_buy_sell_signals

def quick_backfill():
    """
    A simpler backfill process that focuses just on getting indicators into the database
    """
    # Create tables if they don't exist
    create_tables()
    
    # Focus on just Bitcoin and hourly data which is most commonly used
    symbol = "BTCUSDT"
    interval = "1h"
    lookback_days = 3  # Just get a few days of data to really speed things up
    
    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)
    
    print(f"Processing {symbol} on {interval} timeframe...")
    
    # Convert datetime to milliseconds timestamp for API
    start_timestamp = int(start_time.timestamp() * 1000)
    end_timestamp = int(end_time.timestamp() * 1000)
    
    # Get klines data
    klines_data = get_klines_data(symbol, interval, start_timestamp, end_timestamp)
    
    if isinstance(klines_data, pd.DataFrame):
        df = klines_data
    elif isinstance(klines_data, list) and klines_data:
        # Create DataFrame from klines data list
        df = pd.DataFrame(klines_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
    else:
        print(f"No data available for {symbol} on {interval} timeframe")
        return
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Convert numeric columns to proper types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    if df.empty:
        print(f"No data available for {symbol} on {interval} timeframe")
        return
    
    # Save the historical price data
    print("Saving historical price data...")
    save_historical_data(df, symbol, interval)
    
    # Calculate all technical indicators
    print("Calculating indicators...")
    df = add_bollinger_bands(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_ema(df, 9)
    df = add_ema(df, 21)
    
    # Add simple buy/sell signals based on RSI (oversold/overbought) without running full strategy
    print("Adding basic signals...")
    df['buy_signal'] = (df['rsi'] < 30)  # Simple RSI oversold condition
    df['sell_signal'] = (df['rsi'] > 70)  # Simple RSI overbought condition
    
    # Save indicators and simple signals to database
    print("Saving indicators to database...")
    save_indicators(df, symbol, interval)
    
    print(f"Quick backfill complete for {symbol} on {interval} timeframe.")

if __name__ == "__main__":
    quick_backfill()