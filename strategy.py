import pandas as pd
import numpy as np

def evaluate_buy_sell_signals(df, bb_threshold=0.25, rsi_oversold=35, rsi_overbought=65, use_macd_crossover=True,
                         use_bb=True, use_rsi=True, use_macd=True, bb_window=20, rsi_window=14,
                         macd_fast=12, macd_slow=26, macd_signal=9):
    """Evaluate buy/sell signals based on technical indicators
    
    Args:
        df: DataFrame with OHLCV data and indicators
        bb_threshold: Bollinger Bands threshold (0.0-0.5) - more sensitive at 0.25
        rsi_oversold: RSI level considered oversold - more sensitive at 35 (instead of 30)
        rsi_overbought: RSI level considered overbought - more sensitive at 65 (instead of 70)
        use_macd_crossover: Whether to use MACD crossover signals
        use_bb: Whether to use Bollinger Bands in the strategy
        use_rsi: Whether to use RSI in the strategy
        use_macd: Whether to use MACD in the strategy
        bb_window: Window size for Bollinger Bands
        rsi_window: Window size for RSI
        macd_fast: Fast period for MACD
        macd_slow: Slow period for MACD
        macd_signal: Signal period for MACD
    """
    if df.empty:
        return df
    
    # Initialize signal columns
    df['buy_signal'] = False
    df['sell_signal'] = False
    df['position'] = 0  # Track if we're in a position (0 = no position, 1 = in position)
    
    # Check if needed indicators exist
    has_bb = all(col in df.columns for col in ['bb_lower', 'bb_middle', 'bb_upper', 'bb_percent'])
    has_rsi = 'rsi' in df.columns
    has_macd = all(col in df.columns for col in ['macd', 'macd_signal', 'macd_histogram'])
    has_ema = all(col in df.columns for col in ['ema_9', 'ema_21'])
    
    # Create a copy to avoid SettingWithCopyWarning
    result_df = df.copy()
    
    # Keep track of position state (0 = no position, 1 = in position after buy)
    in_position = False
    
    # Define conditions for buy signals
    for i in range(2, len(result_df)):
        # Skip first few rows to have enough data for indicators
        buy_signals = []
        sell_signals = []
        active_indicators = 0
        min_signals = 1  # Default to 1 since we might only have 1 strategy active
        
        # Bollinger Bands strategy - "buy low, sell high"
        if has_bb and use_bb:
            active_indicators += 1
            # Buy signal: Price near lower Bollinger Band (price is low)
            if result_df.iloc[i]['bb_percent'] < bb_threshold and result_df.iloc[i-1]['bb_percent'] >= bb_threshold:
                buy_signals.append(True)
                
            # Sell signal: Price near upper Bollinger Band (price is high)
            if result_df.iloc[i]['bb_percent'] > (1 - bb_threshold) and result_df.iloc[i-1]['bb_percent'] <= (1 - bb_threshold):
                sell_signals.append(True)
        
        # RSI strategy
        if has_rsi and use_rsi:
            active_indicators += 1
            # Buy signal: RSI drops to oversold level (low price) and then starts moving up
            if result_df.iloc[i]['rsi'] > rsi_oversold and result_df.iloc[i-1]['rsi'] <= rsi_oversold:
                buy_signals.append(True)
                
            # Sell signal: RSI reaches overbought level (high price) and then starts moving down
            if result_df.iloc[i]['rsi'] < rsi_overbought and result_df.iloc[i-1]['rsi'] >= rsi_overbought:
                sell_signals.append(True)
        
        # MACD strategy
        if has_macd and use_macd:
            active_indicators += 1
            if use_macd_crossover:
                # Buy signal: MACD crosses above signal line when MACD is below zero (oversold, price is low)
                if (result_df.iloc[i]['macd'] > result_df.iloc[i]['macd_signal'] and 
                    result_df.iloc[i-1]['macd'] <= result_df.iloc[i-1]['macd_signal'] and
                    result_df.iloc[i]['macd'] < 0):
                    buy_signals.append(True)
                    
                # Sell signal: MACD crosses below signal line when MACD is above zero (overbought, price is high)
                if (result_df.iloc[i]['macd'] < result_df.iloc[i]['macd_signal'] and 
                    result_df.iloc[i-1]['macd'] >= result_df.iloc[i-1]['macd_signal'] and
                    result_df.iloc[i]['macd'] > 0):
                    sell_signals.append(True)
            else:
                # Alternative MACD strategy: Use histogram for divergence
                # Buy signal: Histogram turns positive in oversold territory (MACD below zero)
                if (result_df.iloc[i]['macd_histogram'] > 0 and 
                    result_df.iloc[i-1]['macd_histogram'] <= 0 and
                    result_df.iloc[i]['macd'] < 0):
                    buy_signals.append(True)
                
                # Sell signal: Histogram turns negative in overbought territory (MACD above zero)
                if (result_df.iloc[i]['macd_histogram'] < 0 and 
                    result_df.iloc[i-1]['macd_histogram'] >= 0 and
                    result_df.iloc[i]['macd'] > 0):
                    sell_signals.append(True)
        
        # EMA crossover strategy
        if has_ema:
            # Check recent price trend for context (comparing to 50-period EMA if available)
            is_below_trend = False
            is_above_trend = False
            if 'ema_50' in result_df.columns:
                # Below trend means price is relatively low (good for buying)
                is_below_trend = result_df.iloc[i]['close'] < result_df.iloc[i]['ema_50']
                # Above trend means price is relatively high (good for selling)
                is_above_trend = result_df.iloc[i]['close'] > result_df.iloc[i]['ema_50']
            
            # Buy signal: Short-term EMA crosses above long-term EMA when price is below longer trend (buy low)
            if (result_df.iloc[i]['ema_9'] > result_df.iloc[i]['ema_21'] and 
                result_df.iloc[i-1]['ema_9'] <= result_df.iloc[i-1]['ema_21'] and
                (is_below_trend or 'ema_50' not in result_df.columns)):
                buy_signals.append(True)
                
            # Sell signal: Short-term EMA crosses below long-term EMA when price is above longer trend (sell high)
            if (result_df.iloc[i]['ema_9'] < result_df.iloc[i]['ema_21'] and 
                result_df.iloc[i-1]['ema_9'] >= result_df.iloc[i-1]['ema_21'] and
                (is_above_trend or 'ema_50' not in result_df.columns)):
                sell_signals.append(True)
        
        # If we have multiple active indicators, only require one signal for confirmation
        # This makes the strategy more sensitive and likely to generate signals
        min_signals = 1
        
        # Set final signals based on indicator combinations and position state
        potential_buy = len(buy_signals) >= min_signals
        potential_sell = len(sell_signals) >= min_signals
        
        # Only issue a buy if we're not already in a position
        if potential_buy and not in_position:
            result_df.iloc[i, result_df.columns.get_loc('buy_signal')] = True
            result_df.iloc[i, result_df.columns.get_loc('position')] = 1
            in_position = True
            
        # Only issue a sell if we're in a position from a previous buy
        elif potential_sell and in_position:
            result_df.iloc[i, result_df.columns.get_loc('sell_signal')] = True
            result_df.iloc[i, result_df.columns.get_loc('position')] = 0
            in_position = False
        else:
            # Maintain position state in the dataframe
            result_df.iloc[i, result_df.columns.get_loc('position')] = 1 if in_position else 0
    
    return result_df

def backtest_strategy(df, initial_capital=1000.0, position_size=1.0):
    """Backtest trading strategy using buy/sell signals"""
    if df.empty or 'buy_signal' not in df.columns or 'sell_signal' not in df.columns:
        return None
    
    # Create a copy of the DataFrame
    backtest_df = df.copy()
    
    # Add columns for tracking portfolio value
    # Don't overwrite position if it already exists
    if 'position' not in backtest_df.columns:
        backtest_df['position'] = 0
    backtest_df['cash'] = initial_capital
    backtest_df['holdings'] = 0
    backtest_df['portfolio_value'] = initial_capital
    
    # Track trades
    trades = []
    
    # Initial state - check if already in a position
    in_position = False
    entry_price = 0
    entry_time = None
    
    # Initialize position state from the dataframe if available
    if 'position' in backtest_df.columns and len(backtest_df) > 0:
        in_position = backtest_df.iloc[0]['position'] > 0
    
    for i in range(1, len(backtest_df)):
        # Default: carry over previous values
        backtest_df.iloc[i, backtest_df.columns.get_loc('cash')] = backtest_df.iloc[i-1]['cash']
        backtest_df.iloc[i, backtest_df.columns.get_loc('holdings')] = backtest_df.iloc[i-1]['holdings']
        backtest_df.iloc[i, backtest_df.columns.get_loc('position')] = backtest_df.iloc[i-1]['position']
        
        # Get current close price
        current_price = backtest_df.iloc[i]['close']
        current_time = backtest_df.iloc[i]['timestamp']
        
        # Handle buy signal
        if backtest_df.iloc[i]['buy_signal'] and not in_position:
            # Calculate how many coins to buy
            cash_to_use = backtest_df.iloc[i-1]['cash'] * position_size
            coins_to_buy = cash_to_use / current_price
            
            # Update position and cash
            # Ensure dataframe columns have the right dtype before assignment
            if backtest_df['position'].dtype != float:
                backtest_df['position'] = backtest_df['position'].astype(float)
            if backtest_df['cash'].dtype != float:
                backtest_df['cash'] = backtest_df['cash'].astype(float)
            if backtest_df['holdings'].dtype != float:
                backtest_df['holdings'] = backtest_df['holdings'].astype(float)
                
            # Now assign the values
            backtest_df.iloc[i, backtest_df.columns.get_loc('position')] = float(coins_to_buy)
            backtest_df.iloc[i, backtest_df.columns.get_loc('cash')] = float(backtest_df.iloc[i-1]['cash'] - cash_to_use)
            backtest_df.iloc[i, backtest_df.columns.get_loc('holdings')] = float(coins_to_buy * current_price)
            
            # Record trade
            in_position = True
            entry_price = current_price
            entry_time = current_time
            
            # Get indicators at buy time for analysis
            indicators = {}
            for col in backtest_df.columns:
                if col in ['rsi', 'macd', 'bb_percent', 'stoch_k', 'stoch_d', 'adx']:
                    if col in backtest_df.columns:
                        indicators[col] = backtest_df.iloc[i].get(col, None)
            
            trades.append({
                'type': 'BUY',
                'price': current_price,
                'coins': coins_to_buy,
                'timestamp': current_time,
                'indicators': indicators
            })
        
        # Handle sell signal
        elif backtest_df.iloc[i]['sell_signal'] and in_position:
            # Calculate proceeds from selling
            coins_to_sell = backtest_df.iloc[i-1]['position']
            cash_from_sale = coins_to_sell * current_price
            
            # Update position and cash
            backtest_df.iloc[i, backtest_df.columns.get_loc('position')] = 0
            backtest_df.iloc[i, backtest_df.columns.get_loc('cash')] = backtest_df.iloc[i-1]['cash'] + cash_from_sale
            backtest_df.iloc[i, backtest_df.columns.get_loc('holdings')] = 0
            
            # Calculate profit and percentage
            profit = current_price - entry_price
            profit_pct = (profit / entry_price) * 100
            holding_time = (current_time - entry_time).total_seconds() / 3600  # hours
            
            # Get indicators at sell time for analysis
            indicators = {}
            for col in backtest_df.columns:
                if col in ['rsi', 'macd', 'bb_percent', 'stoch_k', 'stoch_d', 'adx']:
                    if col in backtest_df.columns:
                        indicators[col] = backtest_df.iloc[i].get(col, None)
            
            # Record trade
            in_position = False
            
            trades.append({
                'type': 'SELL',
                'price': current_price,
                'coins': coins_to_sell,
                'timestamp': current_time,
                'profit': profit,
                'profit_pct': profit_pct,
                'holding_time': holding_time,
                'indicators': indicators
            })
        
        # Update holdings value and portfolio value
        if backtest_df.iloc[i]['position'] > 0:
            # Ensure holdings has float dtype
            if backtest_df['holdings'].dtype != float:
                backtest_df['holdings'] = backtest_df['holdings'].astype(float)
            backtest_df.iloc[i, backtest_df.columns.get_loc('holdings')] = float(backtest_df.iloc[i]['position'] * current_price)
        
        # Ensure portfolio_value has float dtype
        if backtest_df['portfolio_value'].dtype != float:
            backtest_df['portfolio_value'] = backtest_df['portfolio_value'].astype(float)
        # Use float for all numeric values to avoid dtype warnings
        backtest_df.iloc[i, backtest_df.columns.get_loc('portfolio_value')] = float(backtest_df.iloc[i]['cash'] + backtest_df.iloc[i]['holdings'])
    
    # Close any open position at the end of the period using the last price
    if in_position:
        last_price = backtest_df.iloc[-1]['close']
        coins_to_sell = backtest_df.iloc[-1]['position']
        profit = last_price - entry_price
        profit_pct = (profit / entry_price) * 100
        
        trades.append({
            'type': 'SELL (EOT)',  # End of Time period
            'price': last_price,
            'coins': coins_to_sell,
            'timestamp': backtest_df.iloc[-1]['timestamp'],
            'profit': profit,
            'profit_pct': profit_pct,
            'holding_time': (backtest_df.iloc[-1]['timestamp'] - entry_time).total_seconds() / 3600  # hours
        })
    
    # Calculate backtest metrics
    initial_value = backtest_df.iloc[0]['portfolio_value']
    final_value = backtest_df.iloc[-1]['portfolio_value']
    total_return = (final_value - initial_value) / initial_value * 100
    
    # Calculate trade metrics
    sell_trades = [t for t in trades if t['type'] in ['SELL', 'SELL (EOT)']]
    num_trades = len(sell_trades)
    winning_trades = len([t for t in sell_trades if t.get('profit_pct', 0) > 0])
    
    if num_trades > 0:
        win_rate = winning_trades / num_trades * 100
        avg_profit = sum(t.get('profit_pct', 0) for t in sell_trades) / num_trades
        avg_holding_time = sum(t.get('holding_time', 0) for t in sell_trades) / num_trades
    else:
        win_rate = 0
        avg_profit = 0
        avg_holding_time = 0
    
    metrics = {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return_pct': total_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_profit_pct': avg_profit,
        'avg_holding_time_hours': avg_holding_time
    }
    
    return {
        'metrics': metrics,
        'trades': trades,
        'backtest_df': backtest_df
    }

def find_optimal_strategy(df, parameter_ranges=None):
    """
    Test multiple strategy parameter combinations to find the most profitable settings
    
    Args:
        df: DataFrame with OHLCV data
        parameter_ranges: Dictionary of parameter ranges to test, e.g.
            {
                'bb_threshold': [0.1, 0.2, 0.3, 0.4],
                'rsi_oversold': [20, 25, 30, 35],
                'rsi_overbought': [65, 70, 75, 80]
            }
    
    Returns:
        Dictionary with optimal parameters and backtest results
    """
    if df.empty:
        return None
    
    # Default parameter ranges if none provided
    if parameter_ranges is None:
        parameter_ranges = {
            'bb_threshold': [0.1, 0.2, 0.3, 0.4, 0.5],
            'rsi_oversold': [20, 25, 30, 35],
            'rsi_overbought': [65, 70, 75, 80],
            'use_macd_crossover': [True, False]  # Test both MACD strategies
        }
    
    # Calculate all indicators first
    base_df = df.copy()
    try:
        # Import indicators once at the top level
        from indicators import add_bollinger_bands, add_rsi, add_macd
        
        if 'bb_lower' not in base_df.columns:
            base_df = add_bollinger_bands(base_df)
        
        if 'rsi' not in base_df.columns:
            base_df = add_rsi(base_df)
        
        if 'macd' not in base_df.columns:
            base_df = add_macd(base_df)
    except Exception as e:
        # Print error but continue with what we have
        print(f"Error calculating indicators: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    # Test all combinations
    results = []
    
    # Generate all combinations of parameters
    param_combinations = []
    if 'bb_threshold' in parameter_ranges:
        for bb in parameter_ranges['bb_threshold']:
            if 'rsi_oversold' in parameter_ranges:
                for rsi_o in parameter_ranges['rsi_oversold']:
                    if 'rsi_overbought' in parameter_ranges:
                        for rsi_b in parameter_ranges['rsi_overbought']:
                            if 'use_macd_crossover' in parameter_ranges:
                                for macd_cross in parameter_ranges['use_macd_crossover']:
                                    param_combinations.append({
                                        'bb_threshold': bb,
                                        'rsi_oversold': rsi_o,
                                        'rsi_overbought': rsi_b,
                                        'use_macd_crossover': macd_cross
                                    })
                            else:
                                param_combinations.append({
                                    'bb_threshold': bb,
                                    'rsi_oversold': rsi_o,
                                    'rsi_overbought': rsi_b
                                })
    
    # If no parameters to test, use defaults
    if not param_combinations:
        param_combinations = [{'bb_threshold': 0.2, 'rsi_oversold': 30, 'rsi_overbought': 70, 'use_macd_crossover': True}]
    
    # Test each parameter combination
    for params in param_combinations:
        # Generate buy/sell signals with these parameters
        signals_df = evaluate_buy_sell_signals(
            base_df, 
            bb_threshold=params.get('bb_threshold', 0.2),
            rsi_oversold=params.get('rsi_oversold', 30),
            rsi_overbought=params.get('rsi_overbought', 70),
            use_macd_crossover=params.get('use_macd_crossover', True)
        )
        
        # Run backtest
        backtest_results = backtest_strategy(signals_df)
        
        if backtest_results and backtest_results['metrics']['num_trades'] > 0:
            results.append({
                'parameters': params,
                'metrics': backtest_results['metrics'],
                'trades': backtest_results['trades']
            })
    
    # Find best strategy by total return
    if results:
        # Sort by total return
        results.sort(key=lambda x: x['metrics']['total_return_pct'], reverse=True)
        return results[0]
    
    return None
