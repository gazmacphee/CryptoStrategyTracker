import trafilatura
import pandas as pd
import nltk
import json
import os
import requests
from datetime import datetime, timedelta
import database
from sentiment_scraper import analyze_sentiment, clean_text
import random
import time

# Check if OpenAI API key is available
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

# Try to import OpenAI if the API key is available
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        OPENAI_AVAILABLE = True
    except ImportError:
        OPENAI_AVAILABLE = False
else:
    OPENAI_AVAILABLE = False

# News sources for crypto
NEWS_SOURCES = {
    "CoinDesk": "https://www.coindesk.com/tag/{}",
    "CryptoNews": "https://cryptonews.com/tags/{}/",
    "CoinTelegraph": "https://cointelegraph.com/tags/{}"
}

# Cache for news articles to avoid repeated scraping
news_cache = {}
# Cache expiration time in hours
CACHE_EXPIRY = 3

def get_crypto_news(coin, max_articles=5, days_back=3):
    """
    Get news articles about a specific cryptocurrency
    
    Args:
        coin: The cryptocurrency symbol or name (e.g. 'BTC', 'Bitcoin')
        max_articles: Maximum number of articles to return
        days_back: How many days back to look for news
    
    Returns:
        A list of article dictionaries with title, source, url, summary, date and sentiment
    """
    coin_name = coin.replace('USDT', '')
    
    # Check cache first
    cache_key = f"{coin_name}_{max_articles}_{days_back}"
    current_time = datetime.now()
    
    if cache_key in news_cache:
        cache_entry = news_cache[cache_key]
        cache_time = cache_entry['timestamp']
        # If cache is still valid
        if current_time - cache_time < timedelta(hours=CACHE_EXPIRY):
            return cache_entry['data']
    
    articles = []
    
    # For real implementation, iterate through news sources and fetch articles
    # In this simulation, we'll generate placeholder articles
    for source_name, source_url in NEWS_SOURCES.items():
        # In a real implementation, we would:
        # 1. Format URL with coin name/symbol
        # 2. Fetch the page with requests
        # 3. Extract articles with beautifulsoup or similar
        # 4. Parse article dates, titles, and content
        
        # For the simulation, create some example articles
        for i in range(max_articles // len(NEWS_SOURCES) + 1):
            if len(articles) >= max_articles:
                break
                
            # Create a random date within the past 'days_back' days
            random.seed(hash(f"{coin_name}_{source_name}_{i}") % 10000)
            days_ago = random.uniform(0, days_back)
            article_date = current_time - timedelta(days=days_ago)
            
            # Create article data
            article = {
                'title': f"{get_random_title_prefix()} {coin_name} {get_random_title_suffix()}",
                'source': source_name,
                'url': f"{source_url.format(coin_name.lower())}article{i}",
                'date': article_date,
                'content': generate_sample_article_content(coin_name),
                'summary': None,  # Will be filled by AI summarization
                'sentiment': None  # Will be filled by sentiment analysis
            }
            
            # Add sentiment score 
            article['sentiment'] = analyze_sentiment(article['content'])
            
            articles.append(article)
    
    # Sort by date (most recent first)
    articles.sort(key=lambda x: x['date'], reverse=True)
    
    # Take only the max number requested
    articles = articles[:max_articles]
    
    # Update cache
    news_cache[cache_key] = {
        'timestamp': current_time,
        'data': articles
    }
    
    return articles

def get_real_crypto_news(coin, max_articles=5):
    """
    Attempt to get real crypto news by scraping sources
    
    Returns None if unable to fetch real news
    """
    # Only implement real news fetching if configured with proper API keys
    return None

def summarize_article(article_text, coin_name):
    """
    Summarize an article using OpenAI or a fallback method
    """
    # If OpenAI is available, use it for summarization
    if OPENAI_AVAILABLE:
        try:
            prompt = f"Summarize this cryptocurrency article about {coin_name} in 2-3 sentences:\n\n{article_text}"
            
            response = openai_client.chat.completions.create(
                model="gpt-4o", # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error using OpenAI for summarization: {e}")
    
    # Fallback: Use a simple extractive summarization approach
    # Extract first few sentences as a summary
    sentences = nltk.sent_tokenize(article_text)
    if len(sentences) > 3:
        return " ".join(sentences[:3])
    return article_text[:250] + "..."

def generate_personalized_digest(portfolio_symbols, interests=None, max_articles=10):
    """
    Generate a personalized news digest based on portfolio and interests
    
    Args:
        portfolio_symbols: List of cryptocurrency symbols in user's portfolio
        interests: Optional list of additional crypto interests
        max_articles: Maximum number of articles to include in digest
    
    Returns:
        A dictionary with personalized news digest and recommendations
    """
    if portfolio_symbols is None or len(portfolio_symbols) == 0:
        portfolio_symbols = ["BTCUSDT", "ETHUSDT"]  # Default to major coins if no portfolio
    
    if interests is None:
        interests = []
    
    # Combine portfolio and interests, removing USDT suffix
    coins_of_interest = set([symbol.replace('USDT', '') for symbol in portfolio_symbols])
    coins_of_interest.update([interest.replace('USDT', '') for interest in interests])
    
    all_articles = []
    
    # Get news for each coin
    for coin in coins_of_interest:
        articles_per_coin = max(2, max_articles // len(coins_of_interest))
        coin_articles = get_crypto_news(coin, max_articles=articles_per_coin, days_back=5)
        all_articles.extend(coin_articles)
    
    # Sort by date (most recent first)
    all_articles.sort(key=lambda x: x['date'], reverse=True)
    
    # Truncate to maximum articles
    all_articles = all_articles[:max_articles]
    
    # Generate summaries if they don't exist
    for article in all_articles:
        if article['summary'] is None:
            article['summary'] = summarize_article(article['content'], coin)
    
    # Generate recommendations
    recommendations = generate_recommendations(all_articles, coins_of_interest)
    
    return {
        'timestamp': datetime.now(),
        'articles': all_articles,
        'recommendations': recommendations
    }

def generate_recommendations(articles, coins_of_interest):
    """
    Generate AI recommendations based on news articles and user's interests
    """
    recommendations = []
    
    # If OpenAI is available, use it for more sophisticated recommendations
    if OPENAI_AVAILABLE:
        try:
            # Prepare article data for OpenAI
            article_data = []
            for article in articles[:5]:  # Use only top 5 articles for analysis
                article_data.append({
                    'title': article['title'],
                    'summary': article['summary'],
                    'sentiment': article['sentiment'],
                    'date': article['date'].strftime('%Y-%m-%d')
                })
            
            # Create prompt for OpenAI
            prompt = f"""Based on these recent cryptocurrency news articles:
            
{json.dumps(article_data, indent=2)}

And considering the user is interested in these cryptocurrencies: {', '.join(coins_of_interest)}

Generate 3 specific, actionable recommendations for the user. 
Each recommendation should be 1-2 sentences and based on the news data.
Format each as a JSON object with fields 'title' (short title for the recommendation) 
and 'details' (more specific advice or insight).
Return a JSON array of 3 recommendation objects.
"""

            response = openai_client.chat.completions.create(
                model="gpt-4o", # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=500
            )
            
            try:
                # Parse the JSON response
                recommendations_data = json.loads(response.choices[0].message.content)
                if isinstance(recommendations_data, dict) and 'recommendations' in recommendations_data:
                    recommendations = recommendations_data['recommendations']
                elif isinstance(recommendations_data, list):
                    recommendations = recommendations_data
                else:
                    # If it's not a proper format, create structured recommendations
                    recommendations = [
                        {'title': 'Market Analysis', 'details': 'Consider technical indicators alongside news sentiment for a more complete market view.'},
                        {'title': 'Portfolio Diversification', 'details': 'Recent news suggests diversifying across multiple cryptocurrency assets may reduce risk.'},
                        {'title': 'Stay Informed', 'details': 'Continue monitoring news from multiple sources to spot emerging trends early.'}
                    ]
            except json.JSONDecodeError:
                # If parsing fails, extract from text
                text_response = response.choices[0].message.content
                recommendations = extract_recommendations_from_text(text_response)
        except Exception as e:
            print(f"Error generating AI recommendations: {e}")
    
    # If we don't have recommendations yet (OpenAI failed or unavailable),
    # generate some basic recommendations
    if not recommendations:
        recommendations = generate_basic_recommendations(articles, coins_of_interest)
    
    return recommendations

def extract_recommendations_from_text(text):
    """Extract recommendations from text if JSON parsing fails"""
    # Simple extraction of numbered points
    recommendations = []
    lines = text.split('\n')
    current_recommendation = None
    
    for line in lines:
        if line.strip().startswith(('1.', '2.', '3.', '•')):
            if current_recommendation:
                recommendations.append(current_recommendation)
            current_recommendation = {
                'title': line.strip().split('.', 1)[1].strip() if '.' in line else line.strip(),
                'details': ''
            }
        elif current_recommendation and line.strip():
            current_recommendation['details'] += line.strip() + ' '
    
    if current_recommendation:
        recommendations.append(current_recommendation)
    
    # If we still don't have recommendations, create placeholders
    if not recommendations:
        recommendations = [
            {'title': 'Monitor market volatility', 'details': 'Recent news suggests increased market activity.'},
            {'title': 'Research new developments', 'details': 'Stay informed about regulatory changes mentioned in articles.'},
            {'title': 'Consider portfolio diversification', 'details': 'Based on news sentiment, diversification may reduce risk.'}
        ]
    
    return recommendations[:3]  # Limit to 3 recommendations

def generate_basic_recommendations(articles, coins_of_interest):
    """Generate basic recommendations based on article sentiment and content"""
    recommendations = []
    
    # Analyze overall sentiment
    sentiments = [article['sentiment'] for article in articles if article['sentiment'] is not None]
    if sentiments:
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        if avg_sentiment > 0.2:
            recommendations.append({
                'title': 'Positive Market Sentiment',
                'details': f'Recent news about {", ".join(list(coins_of_interest)[:2])} is predominantly positive with a sentiment score of {avg_sentiment:.2f}. Consider reviewing your portfolio for potential opportunities.'
            })
        elif avg_sentiment < -0.2:
            recommendations.append({
                'title': 'Cautious Approach Advised',
                'details': f'News sentiment is currently negative ({avg_sentiment:.2f}) for your coins of interest. Consider monitoring closely before making significant decisions.'
            })
    
    # Add recommendations based on most recent articles
    if articles:
        most_recent = articles[0]
        recommendations.append({
            'title': f'Recent News Impact',
            'details': f'"{most_recent["title"]}" may affect {most_recent["title"].split()[1]} pricing. Consider exploring the full article for more details.'
        })
    
    # Add a diversification recommendation
    if len(coins_of_interest) < 3:
        recommendations.append({
            'title': 'Portfolio Diversification',
            'details': 'Your interest is focused on a small number of cryptocurrencies. Consider researching additional assets to diversify your portfolio.'
        })
    
    # Add general recommendation if we need more
    if len(recommendations) < 3:
        recommendations.append({
            'title': 'Research Trading Patterns',
            'details': 'Use the technical analysis tools in this application to identify potential trading opportunities based on historical patterns.'
        })
    
    return recommendations[:3]  # Limit to 3 recommendations

def get_random_title_prefix():
    """Get a random title prefix for generated news articles"""
    prefixes = [
        "Breaking:",
        "Analysis:",
        "Market Update:",
        "Opinion:",
        "Trend Alert:",
        "Report:",
        "Exclusive:",
        "Just In:",
        "Weekly Review:",
        "Investors Watch:",
    ]
    return random.choice(prefixes)

def get_random_title_suffix():
    """Get a random title suffix for generated news articles"""
    suffixes = [
        "Price Analysis Shows Promising Signs",
        "Reaches New Milestone Amid Market Volatility",
        "Could Lead The Next Bull Run",
        "Adoption Rates Surge as Institutions Take Notice",
        "Technical Indicators Point to Potential Breakout",
        "Faces Regulatory Challenges in Key Markets",
        "Development Team Announces Major Protocol Upgrade",
        "Trading Volume Spikes Following Recent News",
        "Sets New Record for Transaction Processing",
        "Expert Predictions for Q2 2024",
    ]
    return random.choice(suffixes)

def generate_sample_article_content(coin_name):
    """Generate sample article content for testing"""
    paragraphs = [
        f"{coin_name} has been making headlines this week as investors closely monitor its price movements amid broader market fluctuations. Technical analysts have noted several key resistance levels that could determine the asset's trajectory in the coming weeks.",
        
        f"Market sentiment around {coin_name} remains mixed, with institutional investors continuing to accumulate while retail traders show more caution. Trading volumes have seen notable increases during Asian market hours, suggesting growing interest from markets in the East.",
        
        f"Regulatory developments could impact {coin_name}'s adoption rate, as several countries are finalizing framework proposals that would provide clearer guidelines for cryptocurrency transactions and taxation. Industry experts suggest these developments could ultimately benefit established cryptocurrencies like {coin_name} by providing more legitimacy to the sector.",
        
        f"Development activity for {coin_name} has remained robust, with contributor metrics showing steady growth over the past quarter. The upcoming protocol upgrade scheduled for Q3 2024 is expected to address several scaling issues and potentially improve transaction throughput significantly.",
        
        f"As always, investors are advised to conduct thorough research and consider their risk tolerance before making decisions regarding {coin_name} investments. The cryptocurrency market continues to demonstrate high volatility compared to traditional asset classes."
    ]
    
    # Select 2-4 paragraphs randomly
    selected_paragraphs = random.sample(paragraphs, random.randint(2, 4))
    return "\n\n".join(selected_paragraphs)