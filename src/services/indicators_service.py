"""
Technical Indicators Service

Provides functionality to calculate and store technical indicators
for cryptocurrency price data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from src.config.container import container
from src.config import settings


class IndicatorsService:
    """Service for managing technical indicators"""
    
    def __init__(self, logger=None, indicators_repo=None, data_service=None):
        """
        Initialize the indicators service
        
        Args:
            logger: Logger instance
            indicators_repo: Technical indicators repository
            data_service: Data service instance
        """
        # These will be set by the factory method
        self.logger = logger
        self.indicators_repo = indicators_repo
        self.data_service = data_service
    
    def initialize(self, data_service=None):
        """
        Initialize the service with dependencies that might not be available at construction time
        
        Args:
            data_service: Data service instance
        """
        if data_service is not None:
            self.data_service = data_service
    
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
        
        # Ensure all price columns are float to avoid decimal operations
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
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
        
        # Ensure data type is float
        close_float = df['close'].astype(float)
        
        # Calculate moving average and standard deviation
        ma = close_float.rolling(window=window).mean()
        std = close_float.rolling(window=window).std()
        
        # Calculate Bollinger Bands
        bb_lower = ma - (std * window_dev)
        bb_middle = ma
        bb_upper = ma + (std * window_dev)
        
        # Create DataFrame with results
        indicator = pd.DataFrame({
            'bb_lower': bb_lower,
            'bb_middle': bb_middle,
            'bb_upper': bb_upper
        }, index=df.index)
        
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
        
        # Ensure data type is float
        close_float = df['close'].astype(float)
        
        # Calculate RSI using pandas native functions
        delta = close_float.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Create a series with the result
        rsi = pd.Series(rsi, name='rsi', index=df.index)
        
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
        
        # Ensure data type is float
        close_float = df['close'].astype(float)
        
        # Calculate EMAs for MACD
        fast_ema = close_float.ewm(span=fast, adjust=False).mean()
        slow_ema = close_float.ewm(span=slow, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate MACD signal line
        macd_signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Calculate MACD histogram
        macd_histogram = macd_line - macd_signal_line
        
        # Create a DataFrame with the results
        macd_data = pd.DataFrame({
            'macd': macd_line,
            'macd_signal': macd_signal_line,
            'macd_histogram': macd_histogram
        }, index=df.index)
        
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
        
        # Ensure data type is float
        close_float = df['close'].astype(float)
        
        # Calculate each EMA
        for window in windows:
            ema = close_float.ewm(span=window, adjust=False).mean()
            ema = pd.Series(ema, name=f'ema_{window}', index=df.index)
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
        
        # Ensure data types are float
        low_float = df['low'].astype(float)
        high_float = df['high'].astype(float)
        close_float = df['close'].astype(float)
        
        # Calculate %K
        # Formula: %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
        low_min = low_float.rolling(window=k).min()
        high_max = high_float.rolling(window=k).max()
        
        # Avoid division by zero
        denominator = high_max - low_min
        denominator = denominator.replace(0, np.nan)
        
        # Calculate raw %K
        k_raw = 100 * ((close_float - low_min) / denominator)
        
        # Apply smoothing to %K if specified
        if smooth_k > 1:
            stoch_k = k_raw.rolling(window=smooth_k).mean()
        else:
            stoch_k = k_raw
        
        # Calculate %D which is n-period SMA of %K
        stoch_d = stoch_k.rolling(window=d).mean()
        
        # Create a DataFrame with the results
        stoch = pd.DataFrame({
            'stoch_k': stoch_k,
            'stoch_d': stoch_d
        }, index=df.index)
        
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
        
        # Ensure data types are float
        high_float = df['high'].astype(float)
        low_float = df['low'].astype(float)
        close_float = df['close'].astype(float)
        
        # Calculate True Range (TR)
        high_low = high_float - low_float
        high_close_prev = abs(high_float - close_float.shift(1))
        low_close_prev = abs(low_float - close_float.shift(1))
        
        # Create a DataFrame for the TR components
        tr_components = pd.DataFrame({
            'high_low': high_low,
            'high_close_prev': high_close_prev,
            'low_close_prev': low_close_prev
        })
        
        # True Range is the maximum of the three
        tr = tr_components.max(axis=1)
        
        # Average True Range is the simple moving average of the TR
        atr = tr.rolling(window=window).mean()
        
        # Create a series with the result
        atr = pd.Series(atr, name='atr', index=df.index)
        
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
        
        # Ensure data types are float
        high_float = df['high'].astype(float)
        low_float = df['low'].astype(float)
        close_float = df['close'].astype(float)
        
        # ATR calculation required for ADX
        # Calculate True Range (TR)
        high_low = high_float - low_float
        high_close_prev = abs(high_float - close_float.shift(1))
        low_close_prev = abs(low_float - close_float.shift(1))
        
        # Create a DataFrame for TR components
        tr_components = pd.DataFrame({
            'high_low': high_low,
            'high_close_prev': high_close_prev,
            'low_close_prev': low_close_prev
        })
        
        # True Range is the maximum of the three
        tr = tr_components.max(axis=1)
        
        # Calculate +DM and -DM
        up_move = high_float - high_float.shift(1)
        down_move = low_float.shift(1) - low_float
        
        # +DM and -DM conditions
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Convert to series
        pos_dm = pd.Series(pos_dm, index=df.index)
        neg_dm = pd.Series(neg_dm, index=df.index)
        
        # Calculate smoothed +DM, -DM and TR
        smoothed_tr = tr.rolling(window=window).sum()
        smoothed_pos_dm = pos_dm.rolling(window=window).sum()
        smoothed_neg_dm = neg_dm.rolling(window=window).sum()
        
        # Calculate +DI and -DI
        pos_di = 100 * (smoothed_pos_dm / smoothed_tr)
        neg_di = 100 * (smoothed_neg_dm / smoothed_tr)
        
        # Calculate the directional index (DX)
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        
        # Calculate ADX with smoothed DX
        adx = dx.rolling(window=window).mean()
        
        # Create a series with the result
        adx = pd.Series(adx, name='adx', index=df.index)
        
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
        # Check if data_service is available
        if not self.data_service:
            return {'status': 'error', 'message': 'Data service not initialized'}
            
        # Get historical data
        try:
            df = self.data_service.get_klines_data(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time
            )
            
            if df.empty:
                return {'status': 'error', 'message': 'No data available for indicators calculation'}
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting historical data: {e}")
            return {'status': 'error', 'message': f'Failed to retrieve historical data: {str(e)}'}
        
        # We need some extra data before the start time to calculate indicators correctly
        # Get additional data from before the start_time
        if start_time:
            # Calculate how much extra data we need
            # Use the highest window size from our indicators (e.g. EMA 200)
            lookback_periods = 200
            try:
                lookback_seconds = lookback_periods * self.data_service._interval_to_seconds(interval)
                lookback_start = start_time - pd.Timedelta(seconds=lookback_seconds)
                
                # Get lookback data
                lookback_df = self.data_service.get_klines_data(
                    symbol=symbol,
                    interval=interval,
                    start_time=lookback_start,
                    end_time=start_time
                )
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Error getting lookback data: {e}")
                # Continue without lookback data
                lookback_df = pd.DataFrame()
            
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
                if not self.indicators_repo:
                    return {'status': 'error', 'message': 'Indicators repository not initialized'}
                    
                saved_count = self.indicators_repo.save_indicators(to_save)
                
                return {
                    'status': 'success',
                    'indicators_calculated': len(to_save),
                    'indicators_saved': saved_count
                }
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error saving indicators: {e}")
                return {'status': 'error', 'message': str(e)}
        else:
            return {'status': 'warning', 'message': 'No valid indicators to save'}


# Factory function for dependency injection
def indicators_service_factory(container):
    """
    Create and return an IndicatorsService instance with injected dependencies
    
    Args:
        container: The dependency container
        
    Returns:
        Configured IndicatorsService instance
    """
    # Get the required dependencies from the container
    logger = container.get("logger")
    indicators_repo = container.get("indicators_repo")
    
    # Data service might not be available yet during initialization
    # But it will be available by the time the indicators service is used
    data_service = None
    try:
        data_service = container.get("data_service")
    except KeyError:
        # We'll set this later or in the initialize method
        pass
    
    # Create and return service with injected dependencies
    return IndicatorsService(
        logger=logger,
        indicators_repo=indicators_repo,
        data_service=data_service
    )