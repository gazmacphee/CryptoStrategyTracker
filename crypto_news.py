"""
Module for retrieving cryptocurrency news and articles.
This module provides functions to collect and display crypto news.
"""
import datetime
import logging
import pandas as pd
import os
import random
from typing import List, Dict, Any
from database import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_crypto_news(symbol=None, days_back=3, limit=10):
    """
    Get cryptocurrency news articles
    
    Args:
        symbol: Optional cryptocurrency symbol (e.g., 'BTC')
        days_back: Number of days to look back
        limit: Maximum number of articles to return
        
    Returns:
        List of news articles
    """
    # For now, return placeholder data
    # In a real implementation, we'd fetch from API or database
    current_time = datetime.datetime.now()
    
    # Generate reasonable placeholder dates
    dates = [current_time - datetime.timedelta(hours=x*8) for x in range(limit)]
    
    symbol_text = f' about {symbol}' if symbol else ''
    
    articles = [
        {
            'title': f'Latest market analysis{symbol_text}',
            'source': 'Crypto News Today',
            'url': '#',
            'published_at': dates[0].strftime('%Y-%m-%d %H:%M'),
            'summary': 'This article would contain recent cryptocurrency market analysis.'
        },
        {
            'title': f'Technical trends{symbol_text} this week',
            'source': 'Trading View Blog',
            'url': '#',
            'published_at': dates[1].strftime('%Y-%m-%d %H:%M'),
            'summary': 'This article would cover technical trends and analysis of recent cryptocurrency price movements.'
        },
        {
            'title': f'Regulatory updates affecting crypto{symbol_text}',
            'source': 'Blockchain Daily',
            'url': '#',
            'published_at': dates[2].strftime('%Y-%m-%d %H:%M'),
            'summary': 'This article would discuss recent regulatory developments affecting cryptocurrency markets.'
        }
    ]
    
    return articles

def get_news_summary(symbol=None):
    """
    Get a summarized view of recent cryptocurrency news
    
    Args:
        symbol: Optional cryptocurrency symbol (e.g., 'BTC')
        
    Returns:
        Summary text
    """
    return f"News summary functionality for {symbol if symbol else 'all cryptocurrencies'} will be available in a future update."

def save_news_to_db(articles):
    """
    Save news articles to database
    
    Args:
        articles: List of article dictionaries
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if news table exists, create if not
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT,
                url TEXT,
                published_at TIMESTAMP,
                summary TEXT,
                symbol TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert articles
        for article in articles:
            cursor.execute("""
                INSERT INTO news (title, source, url, published_at, summary, symbol)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                article.get('title'),
                article.get('source'),
                article.get('url'),
                article.get('published_at'),
                article.get('summary'),
                article.get('symbol', 'ALL')
            ))
        
        conn.commit()
        cursor.close()
        
    except Exception as e:
        logging.error(f"Error saving news to database: {str(e)}")
        
def get_news_from_db(symbol=None, limit=10):
    """
    Get news articles from database
    
    Args:
        symbol: Optional cryptocurrency symbol (e.g., 'BTC')
        limit: Maximum number of articles to return
        
    Returns:
        List of news articles
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT title, source, url, published_at, summary, symbol
            FROM news
        """
        
        if symbol:
            query += " WHERE symbol = %s OR symbol = 'ALL'"
            query += " ORDER BY published_at DESC LIMIT %s"
            cursor.execute(query, (symbol, limit))
        else:
            query += " ORDER BY published_at DESC LIMIT %s"
            cursor.execute(query, (limit,))
            
        results = cursor.fetchall()
        cursor.close()
        
        articles = []
        for row in results:
            articles.append({
                'title': row[0],
                'source': row[1],
                'url': row[2],
                'published_at': row[3],
                'summary': row[4],
                'symbol': row[5]
            })
            
        return articles
        
    except Exception as e:
        logging.error(f"Error getting news from database: {str(e)}")
        return []


def generate_personalized_digest(portfolio_symbols=None, max_articles=10, days_back=5):
    """
    Generate a personalized news digest for the user based on their portfolio
    and additional interests.
    
    Args:
        portfolio_symbols: List of cryptocurrency symbols in the user's portfolio
        max_articles: Maximum number of articles to include in the digest
        days_back: Number of days to look back for news
        
    Returns:
        Dictionary with personalized digest
    """
    try:
        logging.info(f"Generating personalized digest for symbols: {portfolio_symbols}")
        
        # Placeholder implementation - in a real implementation, we would:
        # 1. Fetch news from multiple sources
        # 2. Filter by relevance to the user's interests
        # 3. Use AI to personalize and summarize (if OpenAI API key available)
        
        all_articles = []
        
        # Define some sample news sources
        sources = ["CoinDesk", "CryptoNews", "CoinTelegraph"]
        topics = ["market analysis", "price predictions", "regulatory updates", 
                 "adoption news", "technology developments"]
        
        # If no portfolio symbols, use some defaults
        if not portfolio_symbols or len(portfolio_symbols) == 0:
            portfolio_symbols = ["BTC", "ETH", "SOL"]
        
        # Generate some articles
        count_per_symbol = max(1, max_articles // len(portfolio_symbols))
        remaining = max_articles
        
        current_time = datetime.datetime.now()
        
        for symbol in portfolio_symbols:
            # Fetch from database if available
            db_articles = get_news_from_db(symbol=symbol, limit=count_per_symbol)
            
            if db_articles:
                all_articles.extend(db_articles)
                remaining -= len(db_articles)
            else:
                # Generate articles for this symbol
                for i in range(min(count_per_symbol, remaining)):
                    source = random.choice(sources)
                    topic = random.choice(topics)
                    days_ago = random.randint(0, days_back)
                    hours_ago = random.randint(0, 23)
                    article_time = current_time - datetime.timedelta(days=days_ago, hours=hours_ago)
                    
                    article = {
                        'title': f"{symbol}: {topic.title()}",
                        'source': source,
                        'url': f"#",
                        'published_at': article_time.strftime('%Y-%m-%d %H:%M'),
                        'summary': f"This would be a summary of an article about {symbol} {topic}.",
                        'symbol': symbol,
                        'relevance_score': random.uniform(0.7, 0.95)
                    }
                    
                    all_articles.append(article)
                    remaining -= 1
        
        # Sort articles by date (newest first)
        all_articles.sort(key=lambda x: x.get('published_at', ''), reverse=True)
        
        # Create the digest
        summary = f"News digest for {', '.join(portfolio_symbols[:3])}"
        if len(portfolio_symbols) > 3:
            summary += f" and {len(portfolio_symbols) - 3} more symbols"
            
        # Check if OpenAI API key is available for enhanced features
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        
        return {
            'summary': summary,
            'articles': all_articles[:max_articles],
            'generated_at': current_time.strftime('%Y-%m-%d %H:%M'),
            'ai_enhanced': has_openai
        }
        
    except Exception as e:
        logging.error(f"Error generating personalized digest: {str(e)}")
        
        # Return a basic digest as fallback
        current_time = datetime.datetime.now()
        return {
            'summary': "Basic news digest",
            'articles': get_crypto_news(limit=max_articles),
            'generated_at': current_time.strftime('%Y-%m-%d %H:%M'),
            'ai_enhanced': False
        }