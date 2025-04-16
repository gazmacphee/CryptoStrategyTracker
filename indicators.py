import numpy as np
import pandas as pd
import pandas_ta as ta

def add_bollinger_bands(df, window=20, window_dev=2):
    """Add Bollinger Bands indicator to DataFrame"""
    if df.empty:
        return df
    
    # Calculate Bollinger Bands
    bb_result = ta.bbands(df['close'], length=window, std=window_dev)
    
    # Add to DataFrame
    df['bb_lower'] = bb_result['BBL_' + str(window) + '_' + str(window_dev) + '.0']
    df['bb_middle'] = bb_result['BBM_' + str(window) + '_' + str(window_dev) + '.0']
    df['bb_upper'] = bb_result['BBU_' + str(window) + '_' + str(window_dev) + '.0']
    
    # Calculate %B (relative position within Bollinger Bands)
    df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    return df

def add_rsi(df, window=14):
    """Add Relative Strength Index indicator to DataFrame"""
    if df.empty:
        return df
    
    # Calculate RSI
    df['rsi'] = ta.rsi(df['close'], length=window)
    
    return df

def add_macd(df, fast=12, slow=26, signal=9):
    """Add MACD indicator to DataFrame"""
    if df.empty:
        return df
    
    # Calculate MACD
    macd_result = ta.macd(df['close'], fast=fast, slow=slow, signal=signal)
    
    # Add to DataFrame
    df['macd'] = macd_result[f'MACD_{fast}_{slow}_{signal}']
    df['macd_signal'] = macd_result[f'MACDs_{fast}_{slow}_{signal}']
    df['macd_histogram'] = macd_result[f'MACDh_{fast}_{slow}_{signal}']
    
    return df

def add_ema(df, window=9):
    """Add Exponential Moving Average to DataFrame"""
    if df.empty:
        return df
    
    # Calculate EMA
    df[f'ema_{window}'] = ta.ema(df['close'], length=window)
    
    return df

def add_atr(df, window=14):
    """Add Average True Range indicator to DataFrame"""
    if df.empty:
        return df
    
    # Calculate ATR
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=window)
    
    return df

def add_stochastic(df, k_period=14, d_period=3, smooth_k=3):
    """Add Stochastic Oscillator to DataFrame"""
    if df.empty:
        return df
    
    # Calculate Stochastic
    stoch_result = ta.stoch(df['high'], df['low'], df['close'], k=k_period, d=d_period, smooth_k=smooth_k)
    
    # Add to DataFrame
    df['stoch_k'] = stoch_result[f'STOCHk_{k_period}_{d_period}_{smooth_k}']
    df['stoch_d'] = stoch_result[f'STOCHd_{k_period}_{d_period}_{smooth_k}']
    
    return df

def add_adx(df, window=14):
    """Add Average Directional Index to DataFrame"""
    if df.empty:
        return df
    
    # Calculate ADX
    adx_result = ta.adx(df['high'], df['low'], df['close'], length=window)
    
    # Add to DataFrame
    df['adx'] = adx_result[f'ADX_{window}']
    df['di_plus'] = adx_result[f'DMP_{window}']
    df['di_minus'] = adx_result[f'DMN_{window}']
    
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
