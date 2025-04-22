"""
Sentiment scraper module for cryptocurrency market sentiment analysis.
This module provides functions to scrape and analyze sentiment data from various sources.
"""
import datetime
import pandas as pd
import requests
import random
import os
from database import save_sentiment_data, get_sentiment_data
import logging
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_sentiment_data(symbol="BTC", days=7):
    """
    Fetch sentiment data for a given symbol from various sources.
    This is a placeholder function until we implement real scraping logic.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., "BTC")
        days: Number of days to look back
    
    Returns:
        DataFrame with sentiment data or None if unavailable
    """
    try:
        # First try to get data from database
        start_time = datetime.datetime.now() - datetime.timedelta(days=days)
        db_data = get_sentiment_data(symbol, start_time=start_time)
        
        if not db_data.empty:
            logger.info(f"Retrieved {len(db_data)} sentiment records from database for {symbol}")
            return db_data
        
        # If no data in database, generate placeholder data
        logger.info(f"No sentiment data found in database for {symbol}, creating placeholder")
        
        # This is where we would implement real web scraping
        # For now we'll use random data with a realistic pattern
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        
        # Get data from database if available, otherwise display informational message
        df = get_sentiment_data(
            symbol=symbol, 
            start_time=start_date.timestamp(), 
            end_time=end_date.timestamp()
        )
        
        if df.empty:
            logger.info(f"No sentiment data available for {symbol}. Data collection in progress.")
            # Return empty dataframe with expected schema
            return pd.DataFrame(columns=['timestamp', 'symbol', 'source', 'sentiment_score', 'volume'])
            
        return df
        
    except Exception as e:
        logger.error(f"Error fetching sentiment data: {str(e)}")
        return pd.DataFrame(columns=['timestamp', 'symbol', 'source', 'sentiment_score', 'volume'])

def analyze_sentiment_correlation(price_df, sentiment_df):
    """
    Analyze correlation between price and sentiment data
    
    Args:
        price_df: DataFrame with price data
        sentiment_df: DataFrame with sentiment data
        
    Returns:
        DataFrame with merged data and correlation statistics
    """
    if price_df.empty or sentiment_df.empty:
        return None
    
    try:
        # Ensure timestamp columns are in the same format
        price_df['date'] = pd.to_datetime(price_df['timestamp'])
        sentiment_df['date'] = pd.to_datetime(sentiment_df['timestamp'])
        
        # Aggregate sentiment by day
        sentiment_daily = sentiment_df.groupby([sentiment_df['date'].dt.date]).agg({
            'sentiment_score': 'mean',
            'volume': 'sum'
        }).reset_index()
        
        # Aggregate price data by day
        price_daily = price_df.groupby([price_df['date'].dt.date]).agg({
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        
        # Merge datasets
        merged = pd.merge(price_daily, sentiment_daily, on='date', suffixes=('_price', '_sentiment'))
        
        # Calculate correlation
        correlation = merged['close'].corr(merged['sentiment_score'])
        
        # Add correlation to the dataframe
        merged['price_sentiment_correlation'] = correlation
        
        return merged
    
    except Exception as e:
        logger.error(f"Error analyzing sentiment correlation: {str(e)}")
        return None

def get_latest_sentiment(symbol="BTC"):
    """
    Get the latest sentiment score for a specific symbol
    
    Args:
        symbol: Cryptocurrency symbol (e.g., "BTC")
        
    Returns:
        Dictionary with latest sentiment data or None if unavailable
    """
    try:
        # Get recent sentiment data
        df = get_sentiment_data(symbol, limit=10)
        
        if df.empty:
            return None
        
        # Get the most recent entry
        latest = df.sort_values('timestamp', ascending=False).iloc[0]
        
        return {
            'symbol': latest['symbol'],
            'score': latest['sentiment_score'],
            'source': latest['source'],
            'timestamp': latest['timestamp'],
            'volume': latest['volume']
        }
    
    except Exception as e:
        logger.error(f"Error getting latest sentiment: {str(e)}")
        return None


def get_combined_sentiment(symbol: str, days_back: int = 7) -> Dict[str, Any]:
    """
    Get combined sentiment data from multiple sources for a specific symbol.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., "BTC")
        days_back: Number of days to look back
        
    Returns:
        Dictionary with combined sentiment metrics
    """
    try:
        logger.info(f"Getting combined sentiment for {symbol}")
        
        # Get sentiment data from database
        start_time = datetime.datetime.now() - datetime.timedelta(days=days_back)
        sentiment_df = get_sentiment_data(symbol, start_time=start_time)
        
        if sentiment_df.empty:
            logger.info(f"No sentiment data found for {symbol}, returning empty result")
            
            # Return placeholder structure with null values
            return {
                'symbol': symbol,
                'overall_score': None,
                'period': f"Last {days_back} days",
                'sources': [],
                'trend': 'neutral',
                'data_available': False,
                'timestamp': datetime.datetime.now().isoformat()
            }
        
        # Calculate overall sentiment metrics
        overall_score = sentiment_df['sentiment_score'].mean()
        
        # Get list of unique sources
        sources = sentiment_df['source'].unique().tolist()
        
        # Calculate sentiment by source
        source_sentiment = []
        for source in sources:
            source_df = sentiment_df[sentiment_df['source'] == source]
            source_score = source_df['sentiment_score'].mean()
            source_volume = source_df['volume'].sum()
            
            source_sentiment.append({
                'source': source,
                'score': source_score,
                'volume': source_volume,
                'weight': len(source_df) / len(sentiment_df)
            })
        
        # Determine trend (positive, negative, neutral)
        if overall_score > 0.6:
            trend = 'positive'
        elif overall_score < 0.4:
            trend = 'negative'
        else:
            trend = 'neutral'
        
        return {
            'symbol': symbol,
            'overall_score': overall_score,
            'period': f"Last {days_back} days",
            'sources': source_sentiment,
            'trend': trend,
            'data_available': True,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting combined sentiment: {str(e)}")
        return {
            'symbol': symbol,
            'overall_score': None,
            'period': f"Last {days_back} days",
            'sources': [],
            'trend': 'neutral',
            'data_available': False,
            'error': str(e),
            'timestamp': datetime.datetime.now().isoformat()
        }


def get_sentiment_summary(symbol: str = None, days_back: int = 7) -> str:
    """
    Get a textual summary of sentiment data for display.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., "BTC")
        days_back: Number of days to look back
        
    Returns:
        Formatted summary text
    """
    try:
        if symbol:
            # Get combined sentiment for specific symbol
            sentiment = get_combined_sentiment(symbol, days_back)
            
            if not sentiment['data_available']:
                return f"No sentiment data available for {symbol} in the last {days_back} days."
            
            # Format the sentiment score as a percentage
            score_pct = round(sentiment['overall_score'] * 100)
            
            # Create a summary
            summary = f"**Sentiment Summary for {symbol}**\n\n"
            summary += f"Overall sentiment: {sentiment['trend'].title()} ({score_pct}%)\n"
            summary += f"Period: {sentiment['period']}\n\n"
            
            if sentiment['sources']:
                summary += "**Sources:**\n"
                for source in sentiment['sources']:
                    source_score_pct = round(source['score'] * 100)
                    summary += f"- {source['source']}: {source_score_pct}% ({source['weight']*100:.1f}% weight)\n"
            
            return summary
        else:
            # Get general market sentiment summary across multiple coins
            top_coins = ["BTC", "ETH", "BNB", "SOL", "XRP"]
            sentiment_data = {}
            
            # Collect sentiment data for top coins
            for coin in top_coins:
                sentiment_data[coin] = get_combined_sentiment(coin, days_back)
            
            # Count coins with available data
            coins_with_data = [c for c in top_coins if sentiment_data[c]['data_available']]
            
            if not coins_with_data:
                return "No sentiment data available for major cryptocurrencies."
            
            # Create a summary
            summary = "**Market Sentiment Overview**\n\n"
            
            for coin in top_coins:
                if sentiment_data[coin]['data_available']:
                    score_pct = round(sentiment_data[coin]['overall_score'] * 100)
                    trend = sentiment_data[coin]['trend'].title()
                    summary += f"- {coin}: {trend} ({score_pct}%)\n"
                else:
                    summary += f"- {coin}: No data available\n"
            
            return summary
            
    except Exception as e:
        logger.error(f"Error generating sentiment summary: {str(e)}")
        return f"Unable to generate sentiment summary due to an error: {str(e)}"