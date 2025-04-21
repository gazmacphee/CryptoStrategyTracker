"""
Technical Indicators Service

Provides functionality to calculate and store technical indicators
for cryptocurrency price data.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta  # Using pandas-ta for technical indicators
from typing import Dict, Any, List, Optional

from src.config.container import container
from src.config import settings


class IndicatorsService:
    """Service for managing technical indicators"""
    
    def __init__(self):
        """Initialize the indicators service"""
        self.logger = container.get("logger")
        self.indicators_repo = container.get("indicators_repo")
        self.data_service = container.get("data_service")
    
    def calculate_all_indicators(
        self,
        df: pd.DataFrame,
        indicator_config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Calculate all technical indicators for a DataFrame of price data
        
        Args:
            df: DataFrame with OHLCV data
            indicator_config: Optional custom indicator configuration
            
        Returns:
            DataFrame with original data and added indicators
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Use provided config or default from settings
        config = indicator_config or settings.TECHNICAL_INDICATORS
        
        # Calculate each enabled indicator
        if config['Bollinger Bands']['enabled']:
            result_df = self.add_bollinger_bands(
                result_df,
                **config['Bollinger Bands']['params']
            )
        
        if config['RSI']['enabled']:
            result_df = self.add_rsi(
                result_df,
                **config['RSI']['params']
            )
        
        if config['MACD']['enabled']:
            result_df = self.add_macd(
                result_df,
                **config['MACD']['params']
            )
        
        if config['EMA']['enabled']:
            result_df = self.add_ema(
                result_df,
                windows=config['EMA']['params']['windows']
            )
        
        if config['Stochastic']['enabled']:
            result_df = self.add_stochastic(
                result_df,
                **config['Stochastic']['params']
            )
        
        if config['ATR']['enabled']:
            result_df = self.add_atr(
                result_df,
                **config['ATR']['params']
            )
        
        if config['ADX']['enabled']:
            result_df = self.add_adx(
                result_df,
                **config['ADX']['params']
            )
        
        return result_df
    
    def add_bollinger_bands(
        self,
        df: pd.DataFrame,
        window: int = 20,
        window_dev: float = 2.0
    ) -> pd.DataFrame:
        """
        Add Bollinger Bands to DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            window: Window size for moving average
            window_dev: Number of standard deviations
            
        Returns:
            DataFrame with added Bollinger Bands
        """
        if df.empty:
            return df
        
        # Calculate Bollinger Bands
        indicator = ta.bbands(df['close'], length=window, std=window_dev)
        
        # Rename columns to match our conventions
        indicator = indicator.rename(columns={
            'BBL_' + str(window) + '_' + str(window_dev): 'bb_lower',
            'BBM_' + str(window) + '_' + str(window_dev): 'bb_middle',
            'BBU_' + str(window) + '_' + str(window_dev): 'bb_upper'
        })
        
        # Merge with original DataFrame
        result = pd.concat([df, indicator], axis=1)
        
        return result
    
    def add_rsi(
        self,
        df: pd.DataFrame,
        window: int = 14
    ) -> pd.DataFrame:
        """
        Add RSI to DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            window: RSI period
            
        Returns:
            DataFrame with added RSI
        """
        if df.empty:
            return df
        
        # Calculate RSI
        rsi = ta.rsi(df['close'], length=window)
        
        # Rename column
        rsi = rsi.rename('rsi')
        
        # Merge with original DataFrame
        result = pd.concat([df, rsi], axis=1)
        
        return result
    
    def add_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """
        Add MACD to DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            DataFrame with added MACD
        """
        if df.empty:
            return df
        
        # Calculate MACD
        macd_data = ta.macd(df['close'], fast=fast, slow=slow, signal=signal)
        
        # Rename columns
        macd_data = macd_data.rename(columns={
            'MACD_' + str(fast) + '_' + str(slow) + '_' + str(signal): 'macd',
            'MACDs_' + str(fast) + '_' + str(slow) + '_' + str(signal): 'macd_signal',
            'MACDh_' + str(fast) + '_' + str(slow) + '_' + str(signal): 'macd_histogram'
        })
        
        # Merge with original DataFrame
        result = pd.concat([df, macd_data], axis=1)
        
        return result
    
    def add_ema(
        self,
        df: pd.DataFrame,
        windows: List[int] = [9, 21, 50, 200]
    ) -> pd.DataFrame:
        """
        Add multiple EMAs to DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            windows: List of EMA periods
            
        Returns:
            DataFrame with added EMAs
        """
        if df.empty:
            return df
        
        result = df.copy()
        
        # Calculate each EMA
        for window in windows:
            ema = ta.ema(df['close'], length=window)
            ema = ema.rename(f'ema_{window}')
            result = pd.concat([result, ema], axis=1)
        
        return result
    
    def add_stochastic(
        self,
        df: pd.DataFrame,
        k: int = 14,
        d: int = 3,
        smooth_k: int = 3
    ) -> pd.DataFrame:
        """
        Add Stochastic Oscillator to DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            k: %K period
            d: %D period
            smooth_k: %K smoothing
            
        Returns:
            DataFrame with added Stochastic
        """
        if df.empty:
            return df
        
        # Calculate Stochastic
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=k, d=d, smooth_k=smooth_k)
        
        # Rename columns
        stoch = stoch.rename(columns={
            'STOCHk_' + str(k) + '_' + str(d) + '_' + str(smooth_k): 'stoch_k',
            'STOCHd_' + str(k) + '_' + str(d) + '_' + str(smooth_k): 'stoch_d'
        })
        
        # Merge with original DataFrame
        result = pd.concat([df, stoch], axis=1)
        
        return result
    
    def add_atr(
        self,
        df: pd.DataFrame,
        window: int = 14
    ) -> pd.DataFrame:
        """
        Add Average True Range to DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            window: ATR period
            
        Returns:
            DataFrame with added ATR
        """
        if df.empty:
            return df
        
        # Calculate ATR
        atr = ta.atr(df['high'], df['low'], df['close'], length=window)
        
        # Rename column
        atr = atr.rename('atr')
        
        # Merge with original DataFrame
        result = pd.concat([df, atr], axis=1)
        
        return result
    
    def add_adx(
        self,
        df: pd.DataFrame,
        window: int = 14
    ) -> pd.DataFrame:
        """
        Add Average Directional Index to DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            window: ADX period
            
        Returns:
            DataFrame with added ADX
        """
        if df.empty:
            return df
        
        # Calculate ADX
        adx = ta.adx(df['high'], df['low'], df['close'], length=window)
        
        # Keep only the main ADX line
        adx = adx['ADX_' + str(window)].rename('adx')
        
        # Merge with original DataFrame
        result = pd.concat([df, adx], axis=1)
        
        return result
    
    def update_indicators(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[pd.Timestamp] = None,
        end_time: Optional[pd.Timestamp] = None
    ) -> Dict[str, Any]:
        """
        Update technical indicators for a symbol/interval
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            start_time: Start time for update range
            end_time: End time for update range
            
        Returns:
            Dictionary with update results
        """
        # Get historical data
        df = self.data_service.get_klines_data(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time
        )
        
        if df.empty:
            return {'status': 'error', 'message': 'No data available for indicators calculation'}
        
        # We need some extra data before the start time to calculate indicators correctly
        # Get additional data from before the start_time
        if start_time:
            # Calculate how much extra data we need
            # Use the highest window size from our indicators (e.g. EMA 200)
            lookback_periods = 200
            lookback_seconds = lookback_periods * self.data_service._interval_to_seconds(interval)
            lookback_start = start_time - pd.Timedelta(seconds=lookback_seconds)
            
            # Get lookback data
            lookback_df = self.data_service.get_klines_data(
                symbol=symbol,
                interval=interval,
                start_time=lookback_start,
                end_time=start_time
            )
            
            # Combine with main data if we got lookback data
            if not lookback_df.empty:
                df = pd.concat([lookback_df, df], ignore_index=True)
                df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        # Calculate indicators
        with_indicators = self.calculate_all_indicators(df)
        
        # Drop rows with NaN indicator values (typically at the beginning of the dataset)
        with_indicators = with_indicators.dropna()
        
        # Filter to requested time range
        if start_time:
            with_indicators = with_indicators[with_indicators['timestamp'] >= start_time]
        
        if end_time:
            with_indicators = with_indicators[with_indicators['timestamp'] <= end_time]
        
        # Prepare data for storage
        to_save = []
        for _, row in with_indicators.iterrows():
            indicator_row = {
                'symbol': symbol,
                'interval': interval,
                'timestamp': row['timestamp'],
                'bb_upper': row.get('bb_upper'),
                'bb_middle': row.get('bb_middle'),
                'bb_lower': row.get('bb_lower'),
                'rsi': row.get('rsi'),
                'macd': row.get('macd'),
                'macd_signal': row.get('macd_signal'),
                'macd_histogram': row.get('macd_histogram'),
                'ema_9': row.get('ema_9'),
                'ema_21': row.get('ema_21'),
                'ema_50': row.get('ema_50'),
                'ema_200': row.get('ema_200'),
                'stoch_k': row.get('stoch_k'),
                'stoch_d': row.get('stoch_d'),
                'atr': row.get('atr'),
                'adx': row.get('adx')
            }
            to_save.append(indicator_row)
        
        # Save to database
        if to_save:
            try:
                # Delete existing indicators for this time range to avoid duplicates
                if start_time and end_time:
                    # TODO: Implement a delete method in the repository
                    pass
                
                # Save new indicators
                saved_count = self.indicators_repo.save_indicators(to_save)
                
                return {
                    'status': 'success',
                    'indicators_calculated': len(to_save),
                    'indicators_saved': saved_count
                }
            except Exception as e:
                self.logger.error(f"Error saving indicators: {e}")
                return {'status': 'error', 'message': str(e)}
        else:
            return {'status': 'warning', 'message': 'No valid indicators to save'}


# Register in the container
def indicators_service_factory(container):
    return IndicatorsService()

container.register_service("indicators_service", indicators_service_factory)