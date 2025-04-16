import pandas as pd
import numpy as np

def evaluate_buy_sell_signals(df, bb_threshold=0.2, rsi_oversold=30, rsi_overbought=70):
    """Evaluate buy/sell signals based on technical indicators"""
    if df.empty:
        return df
    
    # Initialize signal columns
    df['buy_signal'] = False
    df['sell_signal'] = False
    
    # Check if needed indicators exist
    has_bb = all(col in df.columns for col in ['bb_lower', 'bb_middle', 'bb_upper', 'bb_percent'])
    has_rsi = 'rsi' in df.columns
    has_macd = all(col in df.columns for col in ['macd', 'macd_signal', 'macd_histogram'])
    has_ema = all(col in df.columns for col in ['ema_9', 'ema_21'])
    
    # Create a copy to avoid SettingWithCopyWarning
    result_df = df.copy()
    
    # Define conditions for buy signals
    for i in range(2, len(result_df)):
        # Skip first few rows to have enough data for indicators
        buy_signals = []
        sell_signals = []
        
        # Bollinger Bands strategy - price crosses below lower band
        if has_bb:
            # Buy signal: Price near lower Bollinger Band
            if result_df.iloc[i]['bb_percent'] < bb_threshold and result_df.iloc[i-1]['bb_percent'] >= bb_threshold:
                buy_signals.append(True)
                
            # Sell signal: Price near upper Bollinger Band
            if result_df.iloc[i]['bb_percent'] > (1 - bb_threshold) and result_df.iloc[i-1]['bb_percent'] <= (1 - bb_threshold):
                sell_signals.append(True)
        
        # RSI strategy
        if has_rsi:
            # Buy signal: RSI crosses above oversold level
            if result_df.iloc[i]['rsi'] > rsi_oversold and result_df.iloc[i-1]['rsi'] <= rsi_oversold:
                buy_signals.append(True)
                
            # Sell signal: RSI crosses below overbought level
            if result_df.iloc[i]['rsi'] < rsi_overbought and result_df.iloc[i-1]['rsi'] >= rsi_overbought:
                sell_signals.append(True)
        
        # MACD strategy
        if has_macd:
            # Buy signal: MACD crosses above signal line
            if (result_df.iloc[i]['macd'] > result_df.iloc[i]['macd_signal'] and 
                result_df.iloc[i-1]['macd'] <= result_df.iloc[i-1]['macd_signal']):
                buy_signals.append(True)
                
            # Sell signal: MACD crosses below signal line
            if (result_df.iloc[i]['macd'] < result_df.iloc[i]['macd_signal'] and 
                result_df.iloc[i-1]['macd'] >= result_df.iloc[i-1]['macd_signal']):
                sell_signals.append(True)
        
        # EMA crossover strategy
        if has_ema:
            # Buy signal: Short-term EMA crosses above long-term EMA
            if (result_df.iloc[i]['ema_9'] > result_df.iloc[i]['ema_21'] and 
                result_df.iloc[i-1]['ema_9'] <= result_df.iloc[i-1]['ema_21']):
                buy_signals.append(True)
                
            # Sell signal: Short-term EMA crosses below long-term EMA
            if (result_df.iloc[i]['ema_9'] < result_df.iloc[i]['ema_21'] and 
                result_df.iloc[i-1]['ema_9'] >= result_df.iloc[i-1]['ema_21']):
                sell_signals.append(True)
        
        # Set final signals based on indicator combinations
        if len(buy_signals) >= 2:  # At least 2 indicators confirm buy
            result_df.iloc[i, result_df.columns.get_loc('buy_signal')] = True
            
        if len(sell_signals) >= 2:  # At least 2 indicators confirm sell
            result_df.iloc[i, result_df.columns.get_loc('sell_signal')] = True
    
    return result_df

def backtest_strategy(df, initial_capital=1000.0, position_size=1.0):
    """Backtest trading strategy using buy/sell signals"""
    if df.empty or 'buy_signal' not in df.columns or 'sell_signal' not in df.columns:
        return None
    
    # Create a copy of the DataFrame
    backtest_df = df.copy()
    
    # Add columns for tracking portfolio value
    backtest_df['position'] = 0
    backtest_df['cash'] = initial_capital
    backtest_df['holdings'] = 0
    backtest_df['portfolio_value'] = initial_capital
    
    # Track trades
    trades = []
    
    # Initial state
    in_position = False
    entry_price = 0
    
    for i in range(1, len(backtest_df)):
        # Default: carry over previous values
        backtest_df.iloc[i, backtest_df.columns.get_loc('cash')] = backtest_df.iloc[i-1]['cash']
        backtest_df.iloc[i, backtest_df.columns.get_loc('holdings')] = backtest_df.iloc[i-1]['holdings']
        backtest_df.iloc[i, backtest_df.columns.get_loc('position')] = backtest_df.iloc[i-1]['position']
        
        # Get current close price
        current_price = backtest_df.iloc[i]['close']
        
        # Handle buy signal
        if backtest_df.iloc[i]['buy_signal'] and not in_position:
            # Calculate how many coins to buy
            cash_to_use = backtest_df.iloc[i-1]['cash'] * position_size
            coins_to_buy = cash_to_use / current_price
            
            # Update position and cash
            backtest_df.iloc[i, backtest_df.columns.get_loc('position')] = coins_to_buy
            backtest_df.iloc[i, backtest_df.columns.get_loc('cash')] = backtest_df.iloc[i-1]['cash'] - cash_to_use
            backtest_df.iloc[i, backtest_df.columns.get_loc('holdings')] = coins_to_buy * current_price
            
            # Record trade
            in_position = True
            entry_price = current_price
            
            trades.append({
                'type': 'BUY',
                'price': current_price,
                'coins': coins_to_buy,
                'timestamp': backtest_df.iloc[i]['timestamp']
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
            
            # Record trade
            in_position = False
            
            trades.append({
                'type': 'SELL',
                'price': current_price,
                'coins': coins_to_sell,
                'timestamp': backtest_df.iloc[i]['timestamp'],
                'profit_pct': (current_price - entry_price) / entry_price * 100
            })
        
        # Update holdings value and portfolio value
        if backtest_df.iloc[i]['position'] > 0:
            backtest_df.iloc[i, backtest_df.columns.get_loc('holdings')] = backtest_df.iloc[i]['position'] * current_price
        
        backtest_df.iloc[i, backtest_df.columns.get_loc('portfolio_value')] = backtest_df.iloc[i]['cash'] + backtest_df.iloc[i]['holdings']
    
    # Calculate backtest metrics
    initial_value = backtest_df.iloc[0]['portfolio_value']
    final_value = backtest_df.iloc[-1]['portfolio_value']
    total_return = (final_value - initial_value) / initial_value * 100
    
    # Calculate trade metrics
    num_trades = len([t for t in trades if t['type'] == 'SELL'])
    winning_trades = len([t for t in trades if t['type'] == 'SELL' and t.get('profit_pct', 0) > 0])
    if num_trades > 0:
        win_rate = winning_trades / num_trades * 100
    else:
        win_rate = 0
    
    metrics = {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return_pct': total_return,
        'num_trades': num_trades,
        'win_rate': win_rate
    }
    
    return {
        'metrics': metrics,
        'trades': trades,
        'backtest_df': backtest_df
    }
