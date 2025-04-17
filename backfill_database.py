import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

# Import local modules
from database import create_tables, get_db_connection, get_historical_data, save_indicators, save_trade
from binance_api import get_available_symbols, get_klines_data
from indicators import add_bollinger_bands, add_rsi, add_macd, add_ema
from strategy import evaluate_buy_sell_signals, backtest_strategy
from utils import get_timeframe_options, timeframe_to_interval

def process_symbol_timeframe(symbol, interval, lookback_days=365):
    """
    Process a single symbol and timeframe, calculating and storing indicators and trades
    
    Args:
        symbol: The trading pair symbol (e.g., 'BTCUSDT')
        interval: The timeframe interval (e.g., '4h')
        lookback_days: Number of days to look back
    """
    print(f"Processing {symbol} on {interval} timeframe...")
    
    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)
    
    # First, check if we have data in the database
    df = get_historical_data(symbol, interval, start_time, end_time)
    
    # If not enough data in database, fetch it from the API
    if len(df) < 100:
        print(f"Not enough data in database, fetching from API...")
        # Convert datetime to milliseconds timestamp for API
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        # Get klines data
        klines_data = get_klines_data(symbol, interval, start_timestamp, end_timestamp)
        
        # Check if klines_data is a DataFrame or list
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
    
    # Calculate all technical indicators
    df = add_bollinger_bands(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_ema(df, 9)
    df = add_ema(df, 21)
    df = add_ema(df, 50)
    df = add_ema(df, 200)
    
    # Create parameter combinations for strategies to test
    parameter_ranges = {
        'bb_threshold': [0.1, 0.2, 0.3, 0.4, 0.5],
        'rsi_oversold': [20, 25, 30, 35],
        'rsi_overbought': [65, 70, 75, 80],
        'use_macd_crossover': [True, False]
    }
    
    # Process different strategy combinations
    best_return = -float('inf')
    best_strategy = None
    best_metrics = None
    best_backtest_df = None
    
    # Base parameters
    base_params = {
        'bb_threshold': 0.2,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'use_macd_crossover': True,
        'use_bb': True,
        'use_rsi': True,
        'use_macd': True
    }
    
    # Add enabled parameters to combinations
    combinations = [{}]
    
    def add_parameter_combinations(combinations, param_name, values):
        if not combinations:
            return [{param_name: value} for value in values]
        
        new_combinations = []
        for combo in combinations:
            for value in values:
                new_combo = combo.copy()
                new_combo[param_name] = value
                new_combinations.append(new_combo)
        return new_combinations
    
    # Build combinations based on parameter ranges
    combinations = add_parameter_combinations(combinations, 'bb_threshold', parameter_ranges['bb_threshold'])
    combinations = add_parameter_combinations(combinations, 'rsi_oversold', parameter_ranges['rsi_oversold'])
    combinations = add_parameter_combinations(combinations, 'rsi_overbought', parameter_ranges['rsi_overbought'])
    combinations = add_parameter_combinations(combinations, 'use_macd_crossover', parameter_ranges['use_macd_crossover'])
    
    # Fill in defaults for any parameter that wasn't added
    for combo in combinations:
        for param, default_value in base_params.items():
            if param not in combo:
                combo[param] = default_value
    
    print(f"Testing {len(combinations)} parameter combinations")
    
    # Test each parameter combination
    for params in combinations:
        # Extract parameters
        bb_threshold = float(params.get('bb_threshold', 0.2))
        rsi_oversold = int(params.get('rsi_oversold', 30))
        rsi_overbought = int(params.get('rsi_overbought', 70))
        use_macd_crossover = bool(params.get('use_macd_crossover', True))
        use_bb = bool(params.get('use_bb', True))
        use_rsi = bool(params.get('use_rsi', True))
        use_macd = bool(params.get('use_macd', True))
        
        # Generate buy/sell signals with these parameters
        signals_df = evaluate_buy_sell_signals(
            df.copy(), 
            bb_threshold=bb_threshold,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            use_macd_crossover=use_macd_crossover,
            use_bb=use_bb,
            use_rsi=use_rsi,
            use_macd=use_macd
        )
        
        # Implement the trading rule: Don't buy again after a buy signal until we have a profitable sell
        # Create a new DataFrame for the rule-based signals
        rule_df = signals_df.copy()
        rule_df['rule_buy_signal'] = False
        rule_df['rule_sell_signal'] = False
        
        # State variables for tracking positions
        in_position = False
        entry_price = 0
        
        # Apply the rule to each row
        for i in range(len(rule_df)):
            current_price = rule_df.iloc[i]['close']
            
            # Check for buy signal when not in position
            if rule_df.iloc[i]['buy_signal'] and not in_position:
                rule_df.iloc[i, rule_df.columns.get_loc('rule_buy_signal')] = True
                in_position = True
                entry_price = current_price
            
            # Check for sell signal when in position
            elif rule_df.iloc[i]['sell_signal'] and in_position:
                # Only allow sell if it's profitable
                if current_price > entry_price:
                    rule_df.iloc[i, rule_df.columns.get_loc('rule_sell_signal')] = True
                    in_position = False
                    entry_price = 0
        
        # Use the rule-based signals for backtesting
        rule_df['buy_signal'] = rule_df['rule_buy_signal']
        rule_df['sell_signal'] = rule_df['rule_sell_signal']
        
        # Run backtest with the rule-based signals
        backtest_results = backtest_strategy(rule_df)
        
        if backtest_results:
            total_return = backtest_results['metrics']['total_return_pct']
            
            # Track best strategy
            if total_return > best_return:
                best_return = total_return
                best_strategy = params.copy()
                best_metrics = backtest_results['metrics']
                best_backtest_df = backtest_results['backtest_df']
                best_trades = backtest_results['trades']
    
    # Save the best strategy's indicators and signals to the database
    if best_strategy is not None and best_backtest_df is not None and best_metrics is not None:
        print(f"Best strategy found: {best_strategy}")
        print(f"Total return: {best_metrics['total_return_pct']:.2f}%")
        
        # Use the best strategy parameters to generate final signals
        final_df = evaluate_buy_sell_signals(
            df.copy(), 
            bb_threshold=best_strategy['bb_threshold'],
            rsi_oversold=best_strategy['rsi_oversold'],
            rsi_overbought=best_strategy['rsi_overbought'],
            use_macd_crossover=best_strategy['use_macd_crossover'],
            use_bb=best_strategy['use_bb'],
            use_rsi=best_strategy['use_rsi'],
            use_macd=best_strategy['use_macd']
        )
        
        # Apply the trading rule again to final signals
        rule_df = final_df.copy()
        rule_df['rule_buy_signal'] = False
        rule_df['rule_sell_signal'] = False
        
        in_position = False
        entry_price = 0
        
        for i in range(len(rule_df)):
            current_price = rule_df.iloc[i]['close']
            
            if rule_df.iloc[i]['buy_signal'] and not in_position:
                rule_df.iloc[i, rule_df.columns.get_loc('rule_buy_signal')] = True
                in_position = True
                entry_price = current_price
            
            elif rule_df.iloc[i]['sell_signal'] and in_position:
                if current_price > entry_price:
                    rule_df.iloc[i, rule_df.columns.get_loc('rule_sell_signal')] = True
                    in_position = False
                    entry_price = 0
        
        # Use the rule-based signals for the final DataFrame
        final_df['buy_signal'] = rule_df['rule_buy_signal']
        final_df['sell_signal'] = rule_df['rule_sell_signal']
        
        # Save indicators to database
        print(f"Saving indicators to database...")
        save_indicators(final_df, symbol, interval)
        
        # Save trades to database
        print(f"Saving trades to database...")
        # Process buy and sell trades
        active_trade_id = None
        entry_price = 0
        entry_time = None
        
        for _, row in final_df.iterrows():
            if row['buy_signal']:
                # Create a buy trade
                buy_trade = {
                    'symbol': symbol,
                    'interval': interval,
                    'type': 'BUY',
                    'timestamp': row['timestamp'],
                    'price': row['close'],
                    'coins': 1.0,  # For simplicity
                    'strategy_params': best_strategy
                }
                
                # Save the buy trade and get the ID
                trade_id = save_trade(buy_trade)
                active_trade_id = trade_id
                entry_price = row['close']
                entry_time = row['timestamp']
                
            elif row['sell_signal'] and active_trade_id is not None:
                # Calculate profit
                profit = row['close'] - entry_price
                profit_pct = (profit / entry_price) * 100
                holding_time = (row['timestamp'] - entry_time).total_seconds() / 3600  # hours
                
                # Create a sell trade
                sell_trade = {
                    'symbol': symbol,
                    'interval': interval,
                    'type': 'SELL',
                    'timestamp': row['timestamp'],
                    'price': row['close'],
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'holding_time': holding_time,
                    'trade_id': active_trade_id
                }
                
                # Save the sell trade
                save_trade(sell_trade)
                active_trade_id = None
    
    print(f"Completed processing {symbol} on {interval} timeframe.")

def backfill_database():
    """
    Backfill the database with historical data, indicators, and trading signals for all symbols and timeframes
    """
    # Create database tables if they don't exist
    create_tables()
    
    # Get available symbols
    symbols = get_available_symbols()
    
    # Use just the top 5 symbols for testing
    top_symbols = symbols[:5]
    
    # Get timeframe options
    timeframes = list(get_timeframe_options().keys())
    
    # Process each symbol-timeframe combination
    for symbol in top_symbols:
        for timeframe in timeframes:
            interval = timeframe_to_interval(timeframe)
            process_symbol_timeframe(symbol, interval)
            time.sleep(1)  # Avoid API rate limits

if __name__ == "__main__":
    backfill_database()