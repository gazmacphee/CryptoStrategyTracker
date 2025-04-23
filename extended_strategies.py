"""
Extended trading strategies module with various technical analysis approaches.
This module provides multiple strategy implementations to generate more signals
for machine learning models to analyze.
"""

import pandas as pd
import numpy as np

def dual_ma_strategy(df, fast_ma=20, slow_ma=50, use_ema=True):
    """
    Implements a dual moving average crossover strategy.
    
    Args:
        df: DataFrame with OHLCV data and indicators
        fast_ma: Fast moving average period
        slow_ma: Slow moving average period
        use_ema: Whether to use EMA (True) or SMA (False)
    
    Returns:
        DataFrame with added buy/sell signals
    """
    if df.empty:
        return df
        
    result_df = df.copy()
    
    # Calculate moving averages if they don't exist
    if use_ema:
        # Use EMA - Exponential Moving Average
        fast_col = f'ema_{fast_ma}'
        slow_col = f'ema_{slow_ma}'
        
        if fast_col not in result_df.columns:
            result_df[fast_col] = result_df['close'].ewm(span=fast_ma, adjust=False).mean()
        
        if slow_col not in result_df.columns:
            result_df[slow_col] = result_df['close'].ewm(span=slow_ma, adjust=False).mean()
    else:
        # Use SMA - Simple Moving Average
        fast_col = f'sma_{fast_ma}'
        slow_col = f'sma_{slow_ma}'
        
        if fast_col not in result_df.columns:
            result_df[fast_col] = result_df['close'].rolling(window=fast_ma).mean()
        
        if slow_col not in result_df.columns:
            result_df[slow_col] = result_df['close'].rolling(window=slow_ma).mean()
    
    # Initialize signal columns if they don't exist
    if 'buy_signal' not in result_df.columns:
        result_df['buy_signal'] = False
    
    if 'sell_signal' not in result_df.columns:
        result_df['sell_signal'] = False
    
    # Initialize position column if it doesn't exist
    if 'position' not in result_df.columns:
        result_df['position'] = 0
    
    # Track if we're in a position (0 = no position, 1 = in position)
    in_position = False
    
    # Generate signals based on moving average crossovers
    for i in range(1, len(result_df)):
        # Buy when fast MA crosses above slow MA (bullish crossover)
        if (result_df.iloc[i-1][fast_col] <= result_df.iloc[i-1][slow_col] and 
            result_df.iloc[i][fast_col] > result_df.iloc[i][slow_col] and 
            not in_position):
            result_df.iloc[i, result_df.columns.get_loc('buy_signal')] = True
            result_df.iloc[i, result_df.columns.get_loc('position')] = 1
            in_position = True
        
        # Sell when fast MA crosses below slow MA (bearish crossover)
        elif (result_df.iloc[i-1][fast_col] >= result_df.iloc[i-1][slow_col] and 
              result_df.iloc[i][fast_col] < result_df.iloc[i][slow_col] and 
              in_position):
            result_df.iloc[i, result_df.columns.get_loc('sell_signal')] = True
            result_df.iloc[i, result_df.columns.get_loc('position')] = 0
            in_position = False
        else:
            # Maintain position state in the dataframe
            result_df.iloc[i, result_df.columns.get_loc('position')] = 1 if in_position else 0
    
    # Add strategy column to identify which strategy generated the signals
    result_df['strategy_used'] = f"dual_ma_{fast_ma}_{slow_ma}_{'ema' if use_ema else 'sma'}"
    return result_df

def stochastic_rsi_strategy(df, stoch_oversold=20, stoch_overbought=80, 
                           rsi_oversold=30, rsi_overbought=70,
                           position_filter=True):
    """
    Implements a strategy using both Stochastic Oscillator and RSI for confirmation.
    
    Args:
        df: DataFrame with OHLCV data and indicators
        stoch_oversold: Stochastic oversold threshold (0-100)
        stoch_overbought: Stochastic overbought threshold (0-100)
        rsi_oversold: RSI oversold threshold (0-100)
        rsi_overbought: RSI overbought threshold (0-100)
        position_filter: Whether to use the 50-day EMA for position filtering
    
    Returns:
        DataFrame with added buy/sell signals
    """
    if df.empty:
        return df
        
    result_df = df.copy()
    
    # Check if required indicators exist
    has_stoch = all(col in result_df.columns for col in ['stoch_k', 'stoch_d'])
    has_rsi = 'rsi' in result_df.columns
    
    # Initialize position filter column
    pos_filter_col = 'ema_50'
    
    # Add position filter using 50-day EMA if requested
    if position_filter:
        if pos_filter_col not in result_df.columns:
            result_df[pos_filter_col] = result_df['close'].ewm(span=50, adjust=False).mean()
    
    # Initialize signal columns if they don't exist
    if 'buy_signal' not in result_df.columns:
        result_df['buy_signal'] = False
    
    if 'sell_signal' not in result_df.columns:
        result_df['sell_signal'] = False
    
    # Initialize position column if it doesn't exist
    if 'position' not in result_df.columns:
        result_df['position'] = 0
    
    # Skip if required indicators are missing
    if not has_stoch or not has_rsi:
        return result_df
    
    # Track if we're in a position
    in_position = False
    
    # Generate signals based on Stochastic and RSI
    for i in range(1, len(result_df)):
        # Buy conditions: 
        # 1. Stochastic K line crosses above D line from below the oversold level
        # 2. RSI is below the oversold level and starting to rise
        # 3. (Optional) Price is above 50 EMA for trend confirmation
        stoch_buy_signal = (result_df.iloc[i-1]['stoch_k'] <= result_df.iloc[i-1]['stoch_d'] and
                          result_df.iloc[i]['stoch_k'] > result_df.iloc[i]['stoch_d'] and
                          result_df.iloc[i-1]['stoch_k'] < stoch_oversold)
        
        rsi_buy_signal = (result_df.iloc[i-1]['rsi'] <= rsi_oversold and
                         result_df.iloc[i]['rsi'] > result_df.iloc[i-1]['rsi'])
        
        trend_filter = True
        if position_filter:
            trend_filter = result_df.iloc[i]['close'] > result_df.iloc[i][pos_filter_col]
        
        if stoch_buy_signal and rsi_buy_signal and trend_filter and not in_position:
            result_df.iloc[i, result_df.columns.get_loc('buy_signal')] = True
            result_df.iloc[i, result_df.columns.get_loc('position')] = 1
            in_position = True
        
        # Sell conditions:
        # 1. Stochastic K line crosses below D line from above the overbought level
        # 2. RSI is above the overbought level and starting to fall
        # 3. (Optional) Price is below 50 EMA for trend confirmation
        stoch_sell_signal = (result_df.iloc[i-1]['stoch_k'] >= result_df.iloc[i-1]['stoch_d'] and
                           result_df.iloc[i]['stoch_k'] < result_df.iloc[i]['stoch_d'] and
                           result_df.iloc[i-1]['stoch_k'] > stoch_overbought)
        
        rsi_sell_signal = (result_df.iloc[i-1]['rsi'] >= rsi_overbought and
                          result_df.iloc[i]['rsi'] < result_df.iloc[i-1]['rsi'])
        
        counter_trend_filter = True
        if position_filter:
            counter_trend_filter = result_df.iloc[i]['close'] < result_df.iloc[i][pos_filter_col]
        
        if stoch_sell_signal and rsi_sell_signal and counter_trend_filter and in_position:
            result_df.iloc[i, result_df.columns.get_loc('sell_signal')] = True
            result_df.iloc[i, result_df.columns.get_loc('position')] = 0
            in_position = False
        else:
            # Maintain position state in the dataframe
            result_df.iloc[i, result_df.columns.get_loc('position')] = 1 if in_position else 0
    
    # Add strategy column to identify which strategy generated the signals
    result_df['strategy_used'] = f"stoch_rsi_{stoch_oversold}_{stoch_overbought}_{rsi_oversold}_{rsi_overbought}"
    return result_df

def breakout_strategy(df, lookback_periods=20, confirmation_periods=2, atr_multiplier=2.0):
    """
    Implements a volatility breakout strategy based on price breaking out of ranges.
    
    Args:
        df: DataFrame with OHLCV data and indicators
        lookback_periods: Number of periods to look back for establishing the range
        confirmation_periods: Number of periods to confirm a breakout
        atr_multiplier: Multiplier applied to ATR for setting stop loss
    
    Returns:
        DataFrame with added buy/sell signals
    """
    if df.empty:
        return df
        
    result_df = df.copy()
    
    # Check if ATR exists, if not calculate it
    if 'atr' not in result_df.columns:
        # We need high, low, close columns for ATR calculation
        if not all(col in result_df.columns for col in ['high', 'low', 'close']):
            return result_df
            
        tr1 = result_df['high'] - result_df['low']
        tr2 = abs(result_df['high'] - result_df['close'].shift(1))
        tr3 = abs(result_df['low'] - result_df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        result_df['atr'] = tr.rolling(window=14).mean()
    
    # Calculate rolling high and low
    result_df['highest_high'] = result_df['high'].rolling(window=lookback_periods).max()
    result_df['lowest_low'] = result_df['low'].rolling(window=lookback_periods).min()
    
    # Initialize signal columns if they don't exist
    if 'buy_signal' not in result_df.columns:
        result_df['buy_signal'] = False
    
    if 'sell_signal' not in result_df.columns:
        result_df['sell_signal'] = False
    
    # Initialize position column if it doesn't exist
    if 'position' not in result_df.columns:
        result_df['position'] = 0
    
    # Track if we're in a position and stop loss level
    in_position = False
    stop_loss = 0.0
    confirmation_count = 0
    
    # Generate signals based on breakouts
    for i in range(lookback_periods + 1, len(result_df)):
        current_price = result_df.iloc[i]['close']
        
        # Buy signal: Price breaks above the highest high of the lookback period
        breakout_up = current_price > result_df.iloc[i-1]['highest_high']
        
        # Sell signal: Price breaks below the lowest low of the lookback period
        breakout_down = current_price < result_df.iloc[i-1]['lowest_low']
        
        # Track confirmation periods for breakouts
        if breakout_up and not in_position:
            confirmation_count += 1
            if confirmation_count >= confirmation_periods:
                result_df.iloc[i, result_df.columns.get_loc('buy_signal')] = True
                result_df.iloc[i, result_df.columns.get_loc('position')] = 1
                in_position = True
                # Set stop loss below current price using ATR
                stop_loss = current_price - (result_df.iloc[i]['atr'] * atr_multiplier)
                confirmation_count = 0
        elif not breakout_up:
            confirmation_count = 0
        
        # Check for stop loss or breakout down
        if in_position:
            # Exit position if price hits stop loss or breaks down
            if current_price <= stop_loss or breakout_down:
                result_df.iloc[i, result_df.columns.get_loc('sell_signal')] = True
                result_df.iloc[i, result_df.columns.get_loc('position')] = 0
                in_position = False
                confirmation_count = 0
            else:
                # Maintain position state
                result_df.iloc[i, result_df.columns.get_loc('position')] = 1
        else:
            # Not in position
            result_df.iloc[i, result_df.columns.get_loc('position')] = 0
    
    # Add strategy column to identify which strategy generated the signals
    result_df['strategy_used'] = f"breakout_{lookback_periods}_{confirmation_periods}_{atr_multiplier}"
    return result_df

def adx_strategy(df, adx_threshold=25, dmi_window=14, use_ema_filter=True, ema_window=50):
    """
    Implements a strategy based on ADX (Average Directional Index) for trend strength
    and DMI (Directional Movement Index) for trend direction.
    
    Args:
        df: DataFrame with OHLCV data and indicators
        adx_threshold: Threshold for ADX to indicate a strong trend (typically 25+)
        dmi_window: Window size for DMI/ADX calculation
        use_ema_filter: Whether to use EMA as an additional filter
        ema_window: Window size for EMA if used as filter
    
    Returns:
        DataFrame with added buy/sell signals
    """
    if df.empty:
        return df
        
    result_df = df.copy()
    
    # Initialize EMA column name
    ema_col = f'ema_{ema_window}'
    
    # Check if ADX exists, calculate if missing
    if 'adx' not in result_df.columns or 'pdi' not in result_df.columns or 'ndi' not in result_df.columns:
        # We need high, low, close columns for ADX calculation
        if not all(col in result_df.columns for col in ['high', 'low', 'close']):
            return result_df
            
        # Calculate DMI components
        # True Range
        result_df['tr'] = np.maximum(
            np.maximum(
                result_df['high'] - result_df['low'],
                abs(result_df['high'] - result_df['close'].shift(1))
            ),
            abs(result_df['low'] - result_df['close'].shift(1))
        )
        
        # Directional Movement
        result_df['up_move'] = result_df['high'] - result_df['high'].shift(1)
        result_df['down_move'] = result_df['low'].shift(1) - result_df['low']
        
        # Positive Directional Movement (+DM)
        result_df['+dm'] = np.where(
            (result_df['up_move'] > result_df['down_move']) & (result_df['up_move'] > 0),
            result_df['up_move'],
            0
        )
        
        # Negative Directional Movement (-DM)
        result_df['-dm'] = np.where(
            (result_df['down_move'] > result_df['up_move']) & (result_df['down_move'] > 0),
            result_df['down_move'],
            0
        )
        
        # Smoothed averages of TR, +DM, -DM
        result_df['tr_' + str(dmi_window)] = result_df['tr'].rolling(window=dmi_window).sum()
        result_df['+dm_' + str(dmi_window)] = result_df['+dm'].rolling(window=dmi_window).sum()
        result_df['-dm_' + str(dmi_window)] = result_df['-dm'].rolling(window=dmi_window).sum()
        
        # Directional Indicators
        result_df['pdi'] = 100 * result_df['+dm_' + str(dmi_window)] / result_df['tr_' + str(dmi_window)]
        result_df['ndi'] = 100 * result_df['-dm_' + str(dmi_window)] / result_df['tr_' + str(dmi_window)]
        
        # Directional Index
        result_df['dx'] = 100 * abs(result_df['pdi'] - result_df['ndi']) / (result_df['pdi'] + result_df['ndi'])
        
        # Average Directional Index
        result_df['adx'] = result_df['dx'].rolling(window=dmi_window).mean()
    
    # Add EMA filter if requested
    if use_ema_filter:
        if ema_col not in result_df.columns:
            result_df[ema_col] = result_df['close'].ewm(span=ema_window, adjust=False).mean()
    
    # Initialize signal columns if they don't exist
    if 'buy_signal' not in result_df.columns:
        result_df['buy_signal'] = False
    
    if 'sell_signal' not in result_df.columns:
        result_df['sell_signal'] = False
    
    # Initialize position column if it doesn't exist
    if 'position' not in result_df.columns:
        result_df['position'] = 0
    
    # Track if we're in a position
    in_position = False
    
    # Generate signals based on ADX and DMI
    for i in range(1, len(result_df)):
        # Skip rows with NaN values
        if result_df.iloc[i]['adx'] != result_df.iloc[i]['adx']:  # Check for NaN
            continue
            
        adx_strong_trend = result_df.iloc[i]['adx'] > adx_threshold
        
        # Buy signal: ADX indicates strong trend, +DI crosses above -DI (bullish)
        pdi_crossover = (result_df.iloc[i-1]['pdi'] <= result_df.iloc[i-1]['ndi'] and
                        result_df.iloc[i]['pdi'] > result_df.iloc[i]['ndi'])
        
        # Add EMA filter if requested
        ema_filter_buy = True
        if use_ema_filter:
            ema_filter_buy = result_df.iloc[i]['close'] > result_df.iloc[i][ema_col]
        
        if adx_strong_trend and pdi_crossover and ema_filter_buy and not in_position:
            result_df.iloc[i, result_df.columns.get_loc('buy_signal')] = True
            result_df.iloc[i, result_df.columns.get_loc('position')] = 1
            in_position = True
        
        # Sell signal: ADX indicates strong trend, -DI crosses above +DI (bearish)
        ndi_crossover = (result_df.iloc[i-1]['ndi'] <= result_df.iloc[i-1]['pdi'] and
                        result_df.iloc[i]['ndi'] > result_df.iloc[i]['pdi'])
        
        # Add EMA filter if requested
        ema_filter_sell = True
        if use_ema_filter:
            ema_filter_sell = result_df.iloc[i]['close'] < result_df.iloc[i][ema_col]
        
        if adx_strong_trend and ndi_crossover and ema_filter_sell and in_position:
            result_df.iloc[i, result_df.columns.get_loc('sell_signal')] = True
            result_df.iloc[i, result_df.columns.get_loc('position')] = 0
            in_position = False
        else:
            # Maintain position state in the dataframe
            result_df.iloc[i, result_df.columns.get_loc('position')] = 1 if in_position else 0
    
    # Add strategy column to identify which strategy generated the signals
    result_df['strategy_used'] = f"adx_{adx_threshold}_{dmi_window}"
    return result_df

def support_resistance_strategy(df, lookback=50, threshold_pct=1.0, confirmation_periods=2):
    """
    Implements a strategy based on support and resistance levels.
    
    Args:
        df: DataFrame with OHLCV data and indicators
        lookback: Number of periods to look back for establishing support/resistance
        threshold_pct: Percentage threshold for considering a level broken (0-100)
        confirmation_periods: Number of periods to confirm a breakout
    
    Returns:
        DataFrame with added buy/sell signals
    """
    if df.empty or len(df) < lookback:
        return df
        
    result_df = df.copy()
    
    # Initialize signal columns if they don't exist
    if 'buy_signal' not in result_df.columns:
        result_df['buy_signal'] = False
    
    if 'sell_signal' not in result_df.columns:
        result_df['sell_signal'] = False
    
    # Initialize position column if it doesn't exist
    if 'position' not in result_df.columns:
        result_df['position'] = 0
    
    # Add columns for support and resistance levels
    result_df['support_level'] = 0.0
    result_df['resistance_level'] = 0.0
    
    # Identify local minima and maxima for support and resistance
    for i in range(lookback, len(result_df)):
        # Window for detecting local min/max
        window = result_df.iloc[i-lookback:i]
        
        # Find significant local minima (support levels)
        local_min = window['low'].min()
        result_df.iloc[i, result_df.columns.get_loc('support_level')] = local_min
        
        # Find significant local maxima (resistance levels)
        local_max = window['high'].max()
        result_df.iloc[i, result_df.columns.get_loc('resistance_level')] = local_max
    
    # Track if we're in a position
    in_position = False
    confirmation_count_buy = 0
    confirmation_count_sell = 0
    
    # Generate signals based on support and resistance levels
    for i in range(lookback + 1, len(result_df)):
        current_price = result_df.iloc[i]['close']
        resistance = result_df.iloc[i]['resistance_level']
        support = result_df.iloc[i]['support_level']
        
        # Threshold for breakout/breakdown (e.g., 1% above resistance or below support)
        threshold_resistance = resistance * (1 + threshold_pct/100)
        threshold_support = support * (1 - threshold_pct/100)
        
        # Breakout above resistance (buy signal)
        if current_price > threshold_resistance and not in_position:
            confirmation_count_buy += 1
            confirmation_count_sell = 0
            
            if confirmation_count_buy >= confirmation_periods:
                result_df.iloc[i, result_df.columns.get_loc('buy_signal')] = True
                result_df.iloc[i, result_df.columns.get_loc('position')] = 1
                in_position = True
                confirmation_count_buy = 0
        # Reset confirmation if price falls back below resistance
        elif current_price < resistance and not in_position:
            confirmation_count_buy = 0
        
        # Breakdown below support (sell signal)
        if current_price < threshold_support and in_position:
            confirmation_count_sell += 1
            confirmation_count_buy = 0
            
            if confirmation_count_sell >= confirmation_periods:
                result_df.iloc[i, result_df.columns.get_loc('sell_signal')] = True
                result_df.iloc[i, result_df.columns.get_loc('position')] = 0
                in_position = False
                confirmation_count_sell = 0
        # Reset confirmation if price rises back above support
        elif current_price > support and in_position:
            confirmation_count_sell = 0
            
        # Maintain position state in the dataframe
        result_df.iloc[i, result_df.columns.get_loc('position')] = 1 if in_position else 0
    
    # Add strategy column to identify which strategy generated the signals
    result_df['strategy_used'] = f"sr_{lookback}_{threshold_pct}_{confirmation_periods}"
    return result_df

def apply_all_extended_strategies(df):
    """
    Apply all extended strategies to the dataframe and merge the results.
    
    Args:
        df: DataFrame with OHLCV data and indicators
        
    Returns:
        Dictionary of DataFrames, each with a different strategy applied
    """
    if df.empty:
        return {'original': df}
    
    strategies = {}
    
    # Apply each strategy to a copy of the original dataframe
    strategies['dual_ma_fast'] = dual_ma_strategy(df.copy(), fast_ma=9, slow_ma=21, use_ema=True)
    strategies['dual_ma_medium'] = dual_ma_strategy(df.copy(), fast_ma=20, slow_ma=50, use_ema=True)
    strategies['dual_ma_slow'] = dual_ma_strategy(df.copy(), fast_ma=50, slow_ma=200, use_ema=True)
    
    strategies['stoch_rsi'] = stochastic_rsi_strategy(df.copy(), stoch_oversold=20, stoch_overbought=80,
                                                   rsi_oversold=30, rsi_overbought=70, position_filter=True)
    
    strategies['breakout'] = breakout_strategy(df.copy(), lookback_periods=20, confirmation_periods=2, atr_multiplier=2.0)
    
    strategies['adx_trend'] = adx_strategy(df.copy(), adx_threshold=25, dmi_window=14, use_ema_filter=True)
    
    strategies['support_resistance'] = support_resistance_strategy(df.copy(), lookback=50, threshold_pct=1.0, confirmation_periods=2)
    
    return strategies