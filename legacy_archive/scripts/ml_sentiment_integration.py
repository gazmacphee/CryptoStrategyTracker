"""
Sentiment Integration Module for ML Predictions

This module integrates sentiment analysis data with technical price predictions to create
combined predictions that factor in both market sentiment and price patterns.

Key functionalities:
1. Extract recent sentiment data from the database
2. Calculate sentiment scores and trends
3. Adjust ML predictions based on sentiment signals
4. Provide confidence adjustments for predictions
"""

import os
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Local imports
from database import get_sentiment_data
from crypto_news import get_crypto_news

class SentimentIntegrator:
    """
    Integrates sentiment analysis with ML price predictions
    to create more comprehensive forecasts
    """
    
    def __init__(self, symbol, interval):
        """
        Initialize the sentiment integrator
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Time interval (e.g., '1h', '4h', '1d')
        """
        self.symbol = symbol
        self.interval = interval
        self.sentiment_data = None
        self.sentiment_summary = None
        
        # Get base currency name (e.g., 'BTC' from 'BTCUSDT')
        self.base_currency = symbol.replace('USDT', '')
        if self.base_currency == symbol:  # Fallback if not USDT pair
            self.base_currency = symbol[:3]  # First 3 chars as best guess
    
    def fetch_sentiment_data(self, days_back=7):
        """
        Fetch and process recent sentiment data
        
        Args:
            days_back: How many days of sentiment data to retrieve
            
        Returns:
            DataFrame with sentiment data
        """
        # Calculate date range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        # Get sentiment data from database
        try:
            sentiment_df = get_sentiment_data(self.symbol, start_time=start_time, end_time=end_time)
            
            if sentiment_df.empty:
                logging.warning(f"No sentiment data found for {self.symbol} in the last {days_back} days")
                
                # Try with just the base currency
                sentiment_df = get_sentiment_data(self.base_currency, start_time=start_time, end_time=end_time)
                
                if sentiment_df.empty:
                    logging.warning(f"No sentiment data found for {self.base_currency} either")
                    return None
            
            # Store sentiment data
            self.sentiment_data = sentiment_df
            
            # Process sentiment data into a summary
            self._process_sentiment_data()
            
            return sentiment_df
            
        except Exception as e:
            logging.error(f"Error fetching sentiment data: {e}")
            return None
    
    def _process_sentiment_data(self):
        """
        Process sentiment data to extract useful metrics
        """
        if self.sentiment_data is None or self.sentiment_data.empty:
            self.sentiment_summary = None
            return
        
        # Create a copy of the sentiment data
        df = self.sentiment_data.copy()
        
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Calculate recent average sentiment (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_df = df[df['timestamp'] >= recent_cutoff]
        
        if not recent_df.empty:
            recent_sentiment = recent_df['sentiment_score'].mean()
            recent_volume = recent_df['volume'].sum()
        else:
            recent_sentiment = df['sentiment_score'].iloc[-1] if not df.empty else 0
            recent_volume = df['volume'].iloc[-1] if not df.empty else 0
        
        # Calculate sentiment trend (change over time)
        if len(df) >= 2:
            # Split data into two halves
            half_point = len(df) // 2
            first_half = df.iloc[:half_point]
            second_half = df.iloc[half_point:]
            
            sentiment_trend = second_half['sentiment_score'].mean() - first_half['sentiment_score'].mean()
        else:
            sentiment_trend = 0
            
        # Determine sources with most extreme sentiment
        if not df.empty:
            df_grouped = df.groupby('source').agg({
                'sentiment_score': ['mean', 'count'],
                'volume': 'sum'
            })
            
            df_grouped.columns = ['mean_sentiment', 'count', 'volume']
            df_grouped = df_grouped.reset_index()
            
            # Filter to sources with at least 3 data points
            valid_sources = df_grouped[df_grouped['count'] >= 3]
            
            if not valid_sources.empty:
                most_positive = valid_sources.loc[valid_sources['mean_sentiment'].idxmax()]
                most_negative = valid_sources.loc[valid_sources['mean_sentiment'].idxmin()]
            else:
                most_positive = most_negative = None
        else:
            most_positive = most_negative = None
        
        # Calculate volume-weighted sentiment
        if not df.empty and df['volume'].sum() > 0:
            volume_weighted_sentiment = (df['sentiment_score'] * df['volume']).sum() / df['volume'].sum()
        else:
            volume_weighted_sentiment = recent_sentiment
        
        # Store sentiment summary
        self.sentiment_summary = {
            'recent_sentiment': float(recent_sentiment),
            'sentiment_trend': float(sentiment_trend),
            'volume_weighted_sentiment': float(volume_weighted_sentiment),
            'recent_volume': float(recent_volume),
            'data_points': len(df),
            'sentiment_by_source': df.groupby('source')['sentiment_score'].mean().to_dict() if not df.empty else {},
            'most_positive_source': {
                'source': most_positive['source'],
                'sentiment': float(most_positive['mean_sentiment']),
                'volume': float(most_positive['volume'])
            } if most_positive is not None else None,
            'most_negative_source': {
                'source': most_negative['source'],
                'sentiment': float(most_negative['mean_sentiment']),
                'volume': float(most_negative['volume'])
            } if most_negative is not None else None,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_news_sentiment(self, max_articles=5):
        """
        Get sentiment from recent news articles
        
        Args:
            max_articles: Maximum number of news articles to fetch
            
        Returns:
            Dictionary with news sentiment summary
        """
        try:
            # Try to get news for this currency
            news_articles = get_crypto_news(self.base_currency, max_articles=max_articles)
            
            if not news_articles:
                logging.warning(f"No news articles found for {self.base_currency}")
                return None
            
            # Calculate average sentiment from articles
            sentiments = [article.get('sentiment', 0) for article in news_articles if 'sentiment' in article]
            
            if not sentiments:
                logging.warning(f"No sentiment data found in news articles for {self.base_currency}")
                return None
            
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            # Get most positive and negative headlines
            if news_articles:
                news_with_sentiment = [(a, a.get('sentiment', 0)) for a in news_articles if 'sentiment' in a]
                
                if news_with_sentiment:
                    most_positive = max(news_with_sentiment, key=lambda x: x[1])
                    most_negative = min(news_with_sentiment, key=lambda x: x[1])
                else:
                    most_positive = most_negative = None
            else:
                most_positive = most_negative = None
            
            # Create news sentiment summary
            news_sentiment = {
                'average_sentiment': float(avg_sentiment),
                'article_count': len(news_articles),
                'most_positive': {
                    'title': most_positive[0]['title'],
                    'sentiment': float(most_positive[1]),
                    'source': most_positive[0]['source'],
                    'url': most_positive[0]['url']
                } if most_positive else None,
                'most_negative': {
                    'title': most_negative[0]['title'],
                    'sentiment': float(most_negative[1]),
                    'source': most_negative[0]['source'],
                    'url': most_negative[0]['url']
                } if most_negative else None,
                'timestamp': datetime.now().isoformat()
            }
            
            return news_sentiment
            
        except Exception as e:
            logging.error(f"Error getting news sentiment: {e}")
            return None
    
    def adjust_prediction(self, price_prediction, confidence, include_news=True):
        """
        Adjust a price prediction based on sentiment data
        
        Args:
            price_prediction: The predicted price change (as a decimal, e.g., 0.05 for 5%)
            confidence: The model's confidence in the prediction (0-1)
            include_news: Whether to include news sentiment in the adjustment
            
        Returns:
            Tuple of (adjusted_prediction, adjusted_confidence, sentiment_info)
        """
        # If we don't have sentiment data, try to fetch it
        if self.sentiment_summary is None:
            self.fetch_sentiment_data()
        
        # If we still don't have sentiment data, return original prediction
        if self.sentiment_summary is None:
            return price_prediction, confidence, None
        
        # Get news sentiment if requested
        news_sentiment = self.get_news_sentiment() if include_news else None
        
        # Calculate sentiments to use for adjustment
        recent_sentiment = self.sentiment_summary['recent_sentiment']
        sentiment_trend = self.sentiment_summary['sentiment_trend']
        volume_weighted = self.sentiment_summary['volume_weighted_sentiment']
        
        # If we have news sentiment, include it in the calculation
        if news_sentiment:
            news_weight = 0.3  # Weight for news sentiment
            combined_sentiment = (
                volume_weighted * 0.4 +  # 40% weight for volume-weighted sentiment
                recent_sentiment * 0.3 +  # 30% weight for recent sentiment
                news_sentiment['average_sentiment'] * news_weight  # 30% weight for news sentiment
            )
        else:
            combined_sentiment = (
                volume_weighted * 0.5 +  # 50% weight for volume-weighted sentiment
                recent_sentiment * 0.5    # 50% weight for recent sentiment
            )
        
        # Define sentiment adjustment factor (how much sentiment affects the prediction)
        # This is a tunable parameter - higher values give more weight to sentiment
        sentiment_factor = 0.2
        
        # Calculate sentiment influence on price direction
        # Sentiment ranges typically from -1 to 1, so we scale it to a smaller range for adjustment
        sentiment_adjustment = combined_sentiment * sentiment_factor
        
        # Adjust prediction based on sentiment
        adjusted_prediction = price_prediction + sentiment_adjustment
        
        # Adjust confidence based on sentiment alignment
        # If sentiment agrees with prediction, increase confidence
        # If sentiment disagrees, decrease confidence
        if (price_prediction > 0 and combined_sentiment > 0) or (price_prediction < 0 and combined_sentiment < 0):
            # Sentiment agrees with prediction, increase confidence
            agreement_factor = 0.1  # How much to increase confidence
            adjusted_confidence = min(confidence + (abs(combined_sentiment) * agreement_factor), 1.0)
        else:
            # Sentiment disagrees with prediction, decrease confidence
            disagreement_factor = 0.15  # How much to decrease confidence
            adjusted_confidence = max(confidence - (abs(combined_sentiment) * disagreement_factor), 0.1)
        
        # Create info dictionary about the sentiment adjustment
        sentiment_info = {
            'original_prediction': float(price_prediction),
            'adjusted_prediction': float(adjusted_prediction),
            'original_confidence': float(confidence),
            'adjusted_confidence': float(adjusted_confidence),
            'combined_sentiment': float(combined_sentiment),
            'sentiment_adjustment': float(sentiment_adjustment),
            'sentiment_summary': self.sentiment_summary,
            'news_sentiment': news_sentiment,
            'sentiment_aligned': (price_prediction > 0 and combined_sentiment > 0) or 
                                (price_prediction < 0 and combined_sentiment < 0)
        }
        
        return adjusted_prediction, adjusted_confidence, sentiment_info
    
    def get_sentiment_explanation(self, sentiment_info):
        """
        Generate a human-readable explanation of how sentiment affected a prediction
        
        Args:
            sentiment_info: Dictionary with sentiment adjustment information
            
        Returns:
            String explanation
        """
        if sentiment_info is None:
            return "No sentiment data was available to adjust the prediction."
        
        combined_sentiment = sentiment_info['combined_sentiment']
        sentiment_adjustment = sentiment_info['sentiment_adjustment']
        sentiment_aligned = sentiment_info['sentiment_aligned']
        original_prediction = sentiment_info['original_prediction']
        adjusted_prediction = sentiment_info['adjusted_prediction']
        
        # Format sentiment values as percentages
        formatted_original = f"{original_prediction * 100:.2f}%"
        formatted_adjusted = f"{adjusted_prediction * 100:.2f}%"
        
        # Determine sentiment direction and strength descriptors
        if combined_sentiment > 0.5:
            sentiment_desc = "very bullish"
        elif combined_sentiment > 0.1:
            sentiment_desc = "bullish"
        elif combined_sentiment > -0.1:
            sentiment_desc = "neutral"
        elif combined_sentiment > -0.5:
            sentiment_desc = "bearish"
        else:
            sentiment_desc = "very bearish"
        
        # Create explanation
        if abs(sentiment_adjustment) < 0.001:
            explanation = f"Market sentiment is {sentiment_desc}, but the impact on the prediction was minimal."
        else:
            # Direction of adjustment
            if sentiment_adjustment > 0:
                direction = "increased"
            else:
                direction = "decreased"
            
            # Magnitude of adjustment
            adjustment_pct = abs(sentiment_adjustment * 100)
            if adjustment_pct < 0.5:
                magnitude = "slightly"
            elif adjustment_pct < 1.5:
                magnitude = "moderately"
            else:
                magnitude = "significantly"
            
            explanation = (
                f"Market sentiment is {sentiment_desc}, which {magnitude} {direction} "
                f"the prediction from {formatted_original} to {formatted_adjusted}. "
            )
            
            # Add information about confidence adjustment
            if sentiment_aligned:
                explanation += (
                    f"The sentiment aligns with the technical prediction, "
                    f"increasing confidence in the forecast."
                )
            else:
                explanation += (
                    f"The sentiment contradicts the technical prediction, "
                    f"reducing confidence in the forecast."
                )
        
        # Add source information if available
        if 'sentiment_summary' in sentiment_info and sentiment_info['sentiment_summary']:
            summary = sentiment_info['sentiment_summary']
            
            if 'most_positive_source' in summary and summary['most_positive_source']:
                explanation += f"\n\nMost positive sentiment: {summary['most_positive_source']['source']}"
            
            if 'most_negative_source' in summary and summary['most_negative_source']:
                explanation += f"\n\nMost negative sentiment: {summary['most_negative_source']['source']}"
        
        # Add news headline if available
        if 'news_sentiment' in sentiment_info and sentiment_info['news_sentiment']:
            news = sentiment_info['news_sentiment']
            
            if news['average_sentiment'] > 0 and 'most_positive' in news and news['most_positive']:
                explanation += f"\n\nRecent bullish headline: \"{news['most_positive']['title']}\" ({news['most_positive']['source']})"
            
            elif news['average_sentiment'] < 0 and 'most_negative' in news and news['most_negative']:
                explanation += f"\n\nRecent bearish headline: \"{news['most_negative']['title']}\" ({news['most_negative']['source']})"
        
        return explanation


def integrate_sentiment_with_prediction(symbol, interval, prediction, confidence, include_news=True):
    """
    Helper function to integrate sentiment with a prediction
    
    Args:
        symbol: Trading pair symbol
        interval: Time interval
        prediction: Predicted price change (decimal)
        confidence: Model confidence (0-1)
        include_news: Whether to include news sentiment
        
    Returns:
        Tuple of (adjusted_prediction, adjusted_confidence, explanation)
    """
    try:
        # Initialize sentiment integrator
        integrator = SentimentIntegrator(symbol, interval)
        
        # Fetch sentiment data
        integrator.fetch_sentiment_data()
        
        # Adjust prediction based on sentiment
        adjusted_prediction, adjusted_confidence, sentiment_info = integrator.adjust_prediction(
            prediction, confidence, include_news
        )
        
        # Generate explanation
        explanation = integrator.get_sentiment_explanation(sentiment_info)
        
        return adjusted_prediction, adjusted_confidence, explanation
    
    except Exception as e:
        logging.error(f"Error integrating sentiment with prediction: {e}")
        return prediction, confidence, "Sentiment integration failed. Using original prediction."