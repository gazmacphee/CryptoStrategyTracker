import trafilatura
import pandas as pd
import nltk
from datetime import datetime, timedelta
import random
import time
import database

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Pre-defined news sources and social media platforms
NEWS_SOURCES = {
    "CoinDesk": "https://www.coindesk.com/tag/{}",
    "CryptoNews": "https://cryptonews.com/tags/{}/",
    "CoinTelegraph": "https://cointelegraph.com/tags/{}"
}

SOCIAL_MEDIA = {
    "Twitter": "twitter",
    "Reddit": "reddit"
}

def clean_text(text):
    """Clean scraped text data"""
    if not text:
        return ""
    # Basic cleaning
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())
    return text

def analyze_sentiment(text):
    """Analyze sentiment of text using VADER"""
    if not text or len(text) < 10:
        return 0
    
    sentiment = sia.polarity_scores(text)
    # Return compound score (-1 to 1 range)
    return sentiment['compound']

def get_news_sentiment(symbol, days_back=7):
    """
    Get sentiment from news sources
    
    For demo purposes, this generates simulated data until API keys are provided
    """
    symbol_lower = symbol.lower().replace('usdt', '')
    results = []
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # In a real implementation, we would iterate through news sources and scrape data
    # For now, generate some simulated data based on the coin
    for source_name in NEWS_SOURCES:
        # Create sentiment scores that fluctuate based on symbol and date
        # Generate dates from start_date to end_date
        current_date = start_date
        
        while current_date <= end_date:
            # Generate deterministic but seemingly random sentiment score based on symbol and date
            # This ensures we get consistent values for demos while still looking like real data
            seed = int(current_date.timestamp()) + hash(symbol_lower + source_name) % 1000
            random.seed(seed)
            
            # Generate sentiment that ranges from -0.8 to 0.8
            sentiment_score = (random.random() * 1.6) - 0.8
            
            # volume also depends on the symbol popularity (represented by position in the name)
            volume_base = 100 + (ord(symbol_lower[0]) - ord('a')) * 20
            volume = random.randint(volume_base, volume_base + 200)
            
            # Save to database
            database.save_sentiment_data(
                symbol=symbol,
                source=source_name,
                timestamp=current_date,
                sentiment_score=sentiment_score,
                volume=volume
            )
            
            # Add to results
            results.append({
                'symbol': symbol,
                'source': source_name,
                'timestamp': current_date,
                'sentiment_score': sentiment_score,
                'volume': volume
            })
            
            # Increment by 4 hours to get multiple data points per day
            current_date += timedelta(hours=4)
    
    return pd.DataFrame(results)

def get_social_media_sentiment(symbol, days_back=7):
    """
    Get sentiment from social media platforms
    
    For demo purposes, this generates simulated data until API keys are provided
    """
    symbol_lower = symbol.lower().replace('usdt', '')
    results = []
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # In a real implementation, we would use Twitter/Reddit APIs
    # For now, generate some simulated data
    for platform_name in SOCIAL_MEDIA:
        # Create sentiment scores that fluctuate based on symbol and date
        current_date = start_date
        
        while current_date <= end_date:
            # Generate deterministic but seemingly random sentiment score
            seed = int(current_date.timestamp()) + hash(symbol_lower + platform_name) % 1000
            random.seed(seed)
            
            # Social media tends to be more volatile than news
            sentiment_score = (random.random() * 1.8) - 0.9
            
            # Social media volume is higher
            volume_base = 500 + (ord(symbol_lower[0]) - ord('a')) * 50
            volume = random.randint(volume_base, volume_base + 1000)
            
            # Save to database
            database.save_sentiment_data(
                symbol=symbol,
                source=platform_name,
                timestamp=current_date,
                sentiment_score=sentiment_score,
                volume=volume
            )
            
            # Add to results
            results.append({
                'symbol': symbol,
                'source': platform_name,
                'timestamp': current_date,
                'sentiment_score': sentiment_score,
                'volume': volume
            })
            
            # Increment by 2 hours for social media (more frequent updates)
            current_date += timedelta(hours=2)
    
    return pd.DataFrame(results)

def get_combined_sentiment(symbol, days_back=7):
    """Get combined sentiment from all sources"""
    # Get existing data from database first
    start_time = datetime.now() - timedelta(days=days_back)
    existing_data = database.get_sentiment_data(symbol, start_time=start_time)
    
    # If we don't have enough data, fetch new data
    if len(existing_data) < 10:
        news_sentiment = get_news_sentiment(symbol, days_back)
        social_sentiment = get_social_media_sentiment(symbol, days_back)
        # We don't need to combine since both functions save to the database
        
        # Retrieve the newly saved data
        existing_data = database.get_sentiment_data(symbol, start_time=start_time)
    
    return existing_data

def get_sentiment_summary(sentiment_df):
    """Get summary statistics from sentiment data"""
    if sentiment_df.empty:
        return {
            'average_sentiment': 0,
            'sentiment_trend': 'neutral',
            'volume_trend': 'stable',
            'source_breakdown': {}
        }
    
    # Calculate average sentiment
    avg_sentiment = sentiment_df['sentiment_score'].mean()
    
    # Calculate sentiment trend (compare first and last day)
    sentiment_df['date'] = sentiment_df['timestamp'].dt.date
    daily_sentiment = sentiment_df.groupby('date')['sentiment_score'].mean()
    
    if len(daily_sentiment) > 1:
        first_day = daily_sentiment.iloc[0]
        last_day = daily_sentiment.iloc[-1]
        sentiment_change = last_day - first_day
        
        if sentiment_change > 0.2:
            trend = 'strongly positive'
        elif sentiment_change > 0.05:
            trend = 'positive'
        elif sentiment_change < -0.2:
            trend = 'strongly negative'
        elif sentiment_change < -0.05:
            trend = 'negative'
        else:
            trend = 'neutral'
    else:
        trend = 'neutral'
    
    # Calculate volume trend
    daily_volume = sentiment_df.groupby('date')['volume'].sum()
    
    if len(daily_volume) > 1:
        first_day_vol = daily_volume.iloc[0]
        last_day_vol = daily_volume.iloc[-1]
        volume_change = (last_day_vol - first_day_vol) / first_day_vol if first_day_vol > 0 else 0
        
        if volume_change > 0.3:
            vol_trend = 'rapidly increasing'
        elif volume_change > 0.1:
            vol_trend = 'increasing'
        elif volume_change < -0.3:
            vol_trend = 'rapidly decreasing'
        elif volume_change < -0.1:
            vol_trend = 'decreasing'
        else:
            vol_trend = 'stable'
    else:
        vol_trend = 'stable'
    
    # Source breakdown
    source_breakdown = {}
    for source, group in sentiment_df.groupby('source'):
        source_breakdown[source] = {
            'average_sentiment': group['sentiment_score'].mean(),
            'volume': group['volume'].sum()
        }
    
    return {
        'average_sentiment': avg_sentiment,
        'sentiment_trend': trend,
        'volume_trend': vol_trend,
        'source_breakdown': source_breakdown
    }

def get_website_text_content(url):
    """
    Get text content from a website using trafilatura
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            return text
        return None
    except Exception as e:
        print(f"Error fetching website content: {e}")
        return None