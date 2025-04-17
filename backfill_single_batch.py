"""
Script to backfill data for a single batch of symbols and intervals
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import sys

# Import our modules
from database import create_tables, save_historical_data, get_historical_data
import indicators
from strategy import evaluate_buy_sell_signals, backtest_strategy
from utils import timeframe_to_seconds

def generate_price_data(symbol, interval, days_back=1095, seed=None):
    """
    Generate synthetic price data for a symbol and interval, going back the specified number of days
    """
    # Set a seed based on symbol if not provided
    if seed is None:
        # Use a consistent seed based on the symbol for reproducibility
        seed = sum(ord(c) for c in symbol)
    random.seed(seed)
    np.random.seed(seed)
    
    # Calculate the number of candles based on interval and days
    interval_seconds = {
        '15m': 15*60, '30m': 30*60, '1h': 60*60, '2h': 2*60*60, 
        '4h': 4*60*60, '6h': 6*60*60, '8h': 8*60*60, '12h': 12*60*60,
        '1d': 24*60*60, '3d': 3*24*60*60, '1w': 7*24*60*60, '1M': 30*24*60*60
    }
    seconds_per_candle = interval_seconds.get(interval, 3600)
    total_seconds = days_back * 24 * 60 * 60
    num_candles = total_seconds // seconds_per_candle
    
    # Base price depends on the symbol (make popular coins have realistic starting prices)
    base_prices = {
        'BTCUSDT': 5000.0,  # Starting from a lower price 3 years ago
        'ETHUSDT': 200.0, 
        'BNBUSDT': 30.0, 
        'ADAUSDT': 0.05,
        'DOGEUSDT': 0.002, 
        'XRPUSDT': 0.2, 
        'SOLUSDT': 2.0, 
        'DOTUSDT': 4.0,
        'AVAXUSDT': 4.0,
        'MATICUSDT': 0.02,
        'LTCUSDT': 50.0,
        'LINKUSDT': 2.0
    }
    base_price = base_prices.get(symbol, random.uniform(1.0, 10.0))
    
    # Calculate end time (now) and start time (days_back ago)
    end_time = datetime.now()
    # Normalize to interval boundary
    if interval in ['1h', '2h', '4h', '6h', '8h', '12h']:
        end_time = end_time.replace(minute=0, second=0, microsecond=0)
    elif interval in ['1d', '3d', '1w']:
        end_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0)
    elif interval == '1M':
        end_time = end_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        # For minute-based intervals
        end_time = end_time.replace(second=0, microsecond=0)
        minute = (end_time.minute // int(interval.replace('m', ''))) * int(interval.replace('m', ''))
        end_time = end_time.replace(minute=minute)
    
    # Calculate start time
    start_time = end_time - timedelta(seconds=num_candles * seconds_per_candle)
    
    # Create a list of all timestamps
    timestamps = [start_time + timedelta(seconds=i*seconds_per_candle) for i in range(num_candles)]
    
    # Market trends over the 3-year period (approximate)
    # We'll use this to create more realistic price movements
    trend_changes = [
        # 2022 - Starts bearish, continues mostly bearish
        (0.00, -0.2),  # Initial trend, slightly bearish
        (0.25, -0.5),  # Bear market intensifies
        # 2023 - Starts recovery
        (0.50, 0.3),   # Recovery begins
        (0.65, 0.5),   # Bull trend
        # 2024 - Strong bullish
        (0.80, 0.8),   # Strong bull trend to current day
        (0.95, 1.0),   # Final surge to current prices
    ]
    
    # Create the price trajectory using trends
    prices = [base_price]
    for i in range(1, num_candles):
        # Calculate where we are in the overall timeline (0 to 1)
        progress = i / num_candles
        
        # Find the current trend
        current_trend = trend_changes[0][1]
        for point, trend in trend_changes:
            if progress >= point:
                current_trend = trend
        
        # Calculate daily change with randomness
        # More randomness for shorter timeframes, less for longer ones
        volatility = 1.0 if interval in ['15m', '30m'] else 0.7 if interval in ['1h', '2h'] else 0.5
        daily_change = np.random.normal(current_trend/100, volatility/100)
        
        # Scale for the interval
        interval_change = daily_change * (seconds_per_candle / (24*60*60))
        
        # Update price with some randomness (but following the trend)
        prices.append(prices[-1] * (1 + interval_change))
    
    # Create data with open, high, low, close
    data = []
    for i in range(num_candles):
        # Each candle has some randomness around the calculated price
        close_price = prices[i]
        
        # For previous candle (if it exists)
        if i > 0:
            open_price = prices[i-1]
        else:
            open_price = close_price * (1 - random.uniform(0, 0.01))
        
        # Generate high and low with realistic relationships
        candle_volatility = 0.01 if interval in ['1d', '3d', '1w', '1M'] else 0.005
        high_price = max(open_price, close_price) * (1 + random.uniform(0.001, candle_volatility))
        low_price = min(open_price, close_price) * (1 - random.uniform(0.001, candle_volatility))
        
        # Generate volume with some randomness
        # Higher volume for price changes, lower for sideways movement
        price_change_pct = abs((close_price - open_price) / open_price)
        base_volume = close_price * 100  # Volume roughly proportional to price
        volume = base_volume * (1 + price_change_pct * 10) * random.uniform(0.5, 1.5)
        
        data.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

def process_symbol_interval(symbol, interval, lookback_days=1095):
    """Process a single symbol and interval combination"""
    print(f"Processing {symbol} on {interval} timeframe...")
    
    # Generate the price data
    df = generate_price_data(symbol, interval, days_back=lookback_days)
    
    if df.empty:
        print(f"No data generated for {symbol} ({interval})")
        return
    
    # Save to database
    save_historical_data(df, symbol, interval)
    print(f"Saved {len(df)} historical data points for {symbol} ({interval})")
    
    # Add technical indicators
    df = indicators.add_bollinger_bands(df)
    df = indicators.add_rsi(df)
    df = indicators.add_macd(df)
    df = indicators.add_ema(df)
    
    # Evaluate trading signals
    df = evaluate_buy_sell_signals(df)
    
    # Save indicators to database
    from database import save_indicators
    save_indicators(df, symbol, interval)
    print(f"Saved indicators for {symbol} ({interval})")
    
    # Sleep briefly to avoid database contention
    time.sleep(0.5)

# Define different batches
BATCHES = [
    # Batch 1: BTC & ETH with 1h, 4h, 1d (3 years)
    {"symbols": ["BTCUSDT", "ETHUSDT"], "intervals": ["1h", "4h", "1d"], "lookback": 1095},
    
    # Batch 2: BTC & ETH with 2h, 6h, 12h, 1w (3 years)
    {"symbols": ["BTCUSDT", "ETHUSDT"], "intervals": ["2h", "6h", "12h", "1w"], "lookback": 1095},
    
    # Batch 3: BTC & ETH with 15m, 30m, 8h, 3d, 1M (1 year)
    {"symbols": ["BTCUSDT", "ETHUSDT"], "intervals": ["15m", "30m", "8h", "3d", "1M"], "lookback": 365},
    
    # Batch 4: BNB, SOL, XRP, ADA with 1h, 4h, 1d (2 years)
    {"symbols": ["BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"], "intervals": ["1h", "4h", "1d"], "lookback": 730},
    
    # Batch 5: BNB, SOL, XRP, ADA with 2h, 6h, 12h, 1w (1 year)
    {"symbols": ["BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"], "intervals": ["2h", "6h", "12h", "1w"], "lookback": 365},
    
    # Batch 6: DOT, DOGE, AVAX, MATIC, LTC, LINK with 1h, 4h, 1d (1 year)
    {"symbols": ["DOTUSDT", "DOGEUSDT", "AVAXUSDT", "MATICUSDT", "LTCUSDT", "LINKUSDT"], "intervals": ["1h", "4h", "1d"], "lookback": 365},
]

def run_batch(batch_number):
    """Run a specific batch by number (1-6)"""
    if batch_number < 1 or batch_number > len(BATCHES):
        print(f"Invalid batch number. Please specify a number between 1 and {len(BATCHES)}")
        return
    
    batch = BATCHES[batch_number - 1]
    symbols = batch["symbols"]
    intervals = batch["intervals"]
    lookback = batch["lookback"]
    
    print(f"Running batch {batch_number}/{len(BATCHES)}: {len(symbols)} symbols, {len(intervals)} intervals, {lookback} days")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Intervals: {', '.join(intervals)}")
    
    start_time = datetime.now()
    
    # Make sure database tables exist
    create_tables()
    
    for symbol in symbols:
        for interval in intervals:
            process_symbol_interval(symbol, interval, lookback_days=lookback)
    
    duration = datetime.now() - start_time
    print(f"Batch {batch_number} completed in {duration}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify a batch number (1-6)")
        sys.exit(1)
    
    try:
        batch_number = int(sys.argv[1])
        run_batch(batch_number)
    except ValueError:
        print("Batch number must be an integer")
        sys.exit(1)