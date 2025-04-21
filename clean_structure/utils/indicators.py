import numpy as np
import pandas as pd

def add_bollinger_bands(df, window=20, window_dev=2):
    """Add Bollinger Bands indicator to DataFrame"""
    if df.empty:
        return df
    
    # Calculate rolling mean and standard deviation
    df['bb_middle'] = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    
    # Calculate upper and lower bands
    df['bb_upper'] = df['bb_middle'] + (rolling_std * window_dev)
    df['bb_lower'] = df['bb_middle'] - (rolling_std * window_dev)
    
    # Calculate %B (relative position within Bollinger Bands)
    df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    return df

def add_rsi(df, window=14):
    """Add Relative Strength Index indicator to DataFrame"""
    if df.empty:
        return df
    
    # Calculate price changes
    delta = df['close'].diff()
    
    # Create gain (positive) and loss (negative) series
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Calculate average gain and average loss
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def add_macd(df, fast=12, slow=26, signal=9):
    """Add MACD indicator to DataFrame"""
    if df.empty:
        return df
    
    # Calculate EMA values
    df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    
    # Calculate MACD line
    df['macd'] = df['ema_fast'] - df['ema_slow']
    
    # Calculate MACD signal line
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    
    # Calculate MACD histogram
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Clean up temporary columns
    df.drop(['ema_fast', 'ema_slow'], axis=1, inplace=True, errors='ignore')
    
    return df

def add_ema(df, window=9):
    """Add Exponential Moving Average to DataFrame"""
    if df.empty:
        return df
    
    # Calculate EMA
    df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
    
    return df

def add_atr(df, window=14):
    """Add Average True Range indicator to DataFrame"""
    if df.empty:
        return df
    
    # Calculate True Range
    df['tr0'] = abs(df['high'] - df['low'])
    df['tr1'] = abs(df['high'] - df['close'].shift())
    df['tr2'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    
    # Calculate ATR
    df['atr'] = df['tr'].rolling(window=window).mean()
    
    # Clean up temporary columns
    df.drop(['tr0', 'tr1', 'tr2', 'tr'], axis=1, inplace=True, errors='ignore')
    
    return df

def add_stochastic(df, k_period=14, d_period=3, smooth_k=3):
    """Add Stochastic Oscillator to DataFrame"""
    if df.empty:
        return df
    
    # Calculate %K (fast stochastic)
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    # Handle division by zero
    denom = high_max - low_min
    denom = denom.replace(0, np.nan)
    
    df['stoch_k_raw'] = 100 * ((df['close'] - low_min) / denom)
    
    # Apply smoothing to %K if needed
    if smooth_k > 1:
        df['stoch_k'] = df['stoch_k_raw'].rolling(window=smooth_k).mean()
    else:
        df['stoch_k'] = df['stoch_k_raw']
    
    # Calculate %D (slow stochastic) - moving average of %K
    df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
    
    # Clean up temporary columns
    df.drop(['stoch_k_raw'], axis=1, inplace=True, errors='ignore')
    
    return df

def add_adx(df, window=14):
    """Add Average Directional Index to DataFrame"""
    if df.empty:
        return df
    
    # Calculate True Range
    df['tr0'] = abs(df['high'] - df['low'])
    df['tr1'] = abs(df['high'] - df['close'].shift())
    df['tr2'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    
    # Calculate Directional Movement
    df['up_move'] = df['high'].diff()
    df['down_move'] = df['low'].diff(-1).abs()
    
    # Calculate Positive and Negative Directional Movement
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    # Smooth the True Range and Directional Movement
    df['tr_smooth'] = df['tr'].rolling(window=window).mean()
    df['plus_dm_smooth'] = df['plus_dm'].rolling(window=window).mean()
    df['minus_dm_smooth'] = df['minus_dm'].rolling(window=window).mean()
    
    # Calculate Directional Indicators
    df['di_plus'] = 100 * (df['plus_dm_smooth'] / df['tr_smooth'])
    df['di_minus'] = 100 * (df['minus_dm_smooth'] / df['tr_smooth'])
    
    # Calculate Directional Index
    df['dx'] = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
    
    # Calculate Average Directional Index
    df['adx'] = df['dx'].rolling(window=window).mean()
    
    # Clean up temporary columns
    df.drop(['tr0', 'tr1', 'tr2', 'tr', 'up_move', 'down_move', 'plus_dm', 'minus_dm', 
             'tr_smooth', 'plus_dm_smooth', 'minus_dm_smooth', 'dx'], 
             axis=1, inplace=True, errors='ignore')
    
    return df

def add_volume_profile(df, bins=10):
    """Add Volume Profile analysis"""
    if df.empty or len(df) < bins:
        return df
    
    # Calculate price range
    price_min = df['low'].min()
    price_max = df['high'].max()
    price_bin_size = (price_max - price_min) / bins
    
    # Create price bins
    price_bins = [price_min + i * price_bin_size for i in range(bins + 1)]
    
    # Initialize volume profile
    volume_profile = np.zeros(bins)
    
    # Calculate volume for each bin
    for i in range(len(df)):
        price = df['close'].iloc[i]
        volume = df['volume'].iloc[i]
        
        # Find bin index
        bin_idx = min(int((price - price_min) / price_bin_size), bins - 1)
        volume_profile[bin_idx] += volume
    
    # Add to dataframe
    df['volume_profile_bin'] = pd.cut(df['close'], bins=price_bins, labels=False)
    
    # Calculate POC (Point of Control) - price level with highest volume
    poc_bin = np.argmax(volume_profile)
    poc_price = price_min + (poc_bin + 0.5) * price_bin_size
    
    # Add POC price to all rows
    df['volume_poc'] = poc_price
    
    return df
