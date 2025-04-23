"""
Populate Essential Tables Script

This script populates essential tables with initial data to make
the application functional. It focuses on ML-related tables, benchmarks,
detected patterns, and news data.
"""

import os
import sys
import logging
import time
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("table_population.log")
    ]
)
logger = logging.getLogger(__name__)

# Import database modules
from database import get_db_connection

def get_historical_data(symbol, interval, limit=100):
    """Get historical data for symbol and interval"""
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        return None
    
    try:
        query = """
        SELECT * FROM historical_data 
        WHERE symbol = %s AND interval = %s
        ORDER BY timestamp DESC 
        LIMIT %s
        """
        
        cursor = conn.cursor()
        cursor.execute(query, (symbol, interval, limit))
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not data:
            logger.warning(f"No historical data found for {symbol}/{interval}")
            return None
        
        df = pd.DataFrame(data, columns=columns)
        return df
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        if conn:
            conn.close()
        return None

def populate_ml_predictions(symbols=None, intervals=None):
    """Populate ML predictions table with sample data"""
    logger.info("Populating ML predictions table...")
    
    if not symbols:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]
    
    if not intervals:
        intervals = ["1h", "4h", "1d"]
    
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        return False
    
    try:
        # First check if table already has data
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ml_predictions")
        count = cursor.fetchone()[0]
        
        if count > 0:
            logger.info(f"ML predictions table already has {count} records. Skipping.")
            cursor.close()
            conn.close()
            return True
        
        # Create sample predictions for each symbol and interval
        total_inserted = 0
        now = datetime.now()
        
        for symbol in symbols:
            for interval in intervals:
                # Get some historical data to base predictions on
                df = get_historical_data(symbol, interval, 30)
                if df is None or len(df) < 10:
                    logger.warning(f"Insufficient historical data for {symbol}/{interval}. Skipping.")
                    continue
                
                # Create sample predictions
                for i in range(10):
                    prediction_time = now - timedelta(hours=i*12)
                    target_time = prediction_time + timedelta(days=1)
                    
                    # Generate random prediction
                    current_price = float(df.iloc[0]['close']) if len(df) > 0 else 1000.0
                    predicted_change = random.uniform(-0.05, 0.05)
                    predicted_price = current_price * (1 + predicted_change)
                    confidence = random.uniform(0.6, 0.9)
                    
                    # Insert prediction
                    cursor.execute("""
                    INSERT INTO ml_predictions
                    (symbol, interval, prediction_time, target_time, current_price, 
                     predicted_price, predicted_change, confidence, model_version)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, interval, prediction_time, target_time) DO NOTHING
                    """, (
                        symbol, interval, prediction_time, target_time, current_price,
                        predicted_price, predicted_change, confidence, "sample_v1"
                    ))
                    total_inserted += 1
                
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Inserted {total_inserted} sample ML predictions")
        return total_inserted > 0
    except Exception as e:
        logger.error(f"Error populating ML predictions: {e}")
        if conn:
            conn.close()
        return False

def populate_ml_model_performance(symbols=None, intervals=None):
    """Populate ML model performance with sample data"""
    logger.info("Populating ML model performance table...")
    
    if not symbols:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]
    
    if not intervals:
        intervals = ["1h", "4h", "1d"]
    
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        return False
    
    try:
        # First check if table already has data
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ml_model_performance")
        count = cursor.fetchone()[0]
        
        if count > 0:
            logger.info(f"ML model performance table already has {count} records. Skipping.")
            cursor.close()
            conn.close()
            return True
        
        # Create sample model performance data
        total_inserted = 0
        now = datetime.now()
        
        for symbol in symbols:
            for interval in intervals:
                # Insert model performance
                cursor.execute("""
                INSERT INTO ml_model_performance
                (symbol, interval, model_version, training_date, accuracy, precision_metric, 
                 recall, f1_score, rmse, mae, r_squared, sample_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, interval, model_version) DO NOTHING
                """, (
                    symbol, interval, "sample_v1", now - timedelta(days=1),
                    random.uniform(0.6, 0.8), random.uniform(0.6, 0.8),
                    random.uniform(0.6, 0.8), random.uniform(0.6, 0.8),
                    random.uniform(0.1, 0.2), random.uniform(0.05, 0.15),
                    random.uniform(0.5, 0.7), random.randint(500, 1000)
                ))
                total_inserted += 1
                
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Inserted {total_inserted} sample ML model performance records")
        return total_inserted > 0
    except Exception as e:
        logger.error(f"Error populating ML model performance: {e}")
        if conn:
            conn.close()
        return False

def populate_ml_price_predictions(symbols=None, intervals=None):
    """Populate ML price predictions table with sample data"""
    logger.info("Populating ML price predictions table...")
    
    if not symbols:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]
    
    if not intervals:
        intervals = ["1h", "4h", "1d"]
    
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        return False
    
    try:
        # First check if table already has data
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ml_price_predictions")
        count = cursor.fetchone()[0]
        
        if count > 0:
            logger.info(f"ML price predictions table already has {count} records. Skipping.")
            cursor.close()
            conn.close()
            return True
        
        # Create sample price predictions
        total_inserted = 0
        now = datetime.now()
        
        for symbol in symbols:
            for interval in intervals:
                # Get historical data
                df = get_historical_data(symbol, interval, 30)
                if df is None or len(df) < 5:
                    logger.warning(f"Insufficient historical data for {symbol}/{interval}. Skipping.")
                    continue
                
                # Get the latest price
                latest_price = float(df.iloc[0]['close']) if len(df) > 0 else 1000.0
                
                # Create predictions for different timeframes
                timeframes = [1, 3, 7, 14, 30]  # days
                
                for days in timeframes:
                    # Generate prediction
                    change_pct = random.uniform(-0.15, 0.25) * days/10  # More days, potentially more change
                    predicted_price = latest_price * (1 + change_pct)
                    lower_bound = predicted_price * (1 - random.uniform(0.05, 0.15))
                    upper_bound = predicted_price * (1 + random.uniform(0.05, 0.15))
                    confidence = random.uniform(0.6, 0.9)
                    
                    cursor.execute("""
                    INSERT INTO ml_price_predictions
                    (symbol, interval, prediction_time, target_time, current_price,
                     predicted_price, predicted_change_pct, confidence,
                     lower_bound, upper_bound, model_version)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, interval, prediction_time, target_time) DO NOTHING
                    """, (
                        symbol, interval, now, now + timedelta(days=days), latest_price,
                        predicted_price, change_pct, confidence,
                        lower_bound, upper_bound, "sample_v1"
                    ))
                    total_inserted += 1
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Inserted {total_inserted} sample ML price predictions")
        return total_inserted > 0
    except Exception as e:
        logger.error(f"Error populating ML price predictions: {e}")
        if conn:
            conn.close()
        return False

def populate_ml_market_regimes(symbols=None, intervals=None):
    """Populate market regimes table with sample data"""
    logger.info("Populating market regimes table...")
    
    if not symbols:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]
    
    if not intervals:
        intervals = ["1d"]  # Market regimes typically apply to daily data
    
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        return False
    
    try:
        # First check if table already has data
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ml_market_regimes")
        count = cursor.fetchone()[0]
        
        if count > 0:
            logger.info(f"Market regimes table already has {count} records. Skipping.")
            cursor.close()
            conn.close()
            return True
        
        # Create sample market regime data
        total_inserted = 0
        now = datetime.now()
        
        # Define market regime types
        regime_types = ["bull", "bear", "sideways", "volatile", "recovery"]
        
        for symbol in symbols:
            for interval in intervals:
                # Get some historical data
                df = get_historical_data(symbol, interval, 60)
                if df is None or len(df) < 30:
                    logger.warning(f"Insufficient historical data for {symbol}/{interval}. Skipping.")
                    continue
                
                # Create regimes for past 30 days
                for i in range(10):
                    start_date = now - timedelta(days=30 + i*3)
                    end_date = start_date + timedelta(days=3)
                    
                    # Choose a random regime
                    regime = random.choice(regime_types)
                    
                    # Characteristics based on regime
                    if regime == "bull":
                        volatility = random.uniform(0.01, 0.04)
                        trend = random.uniform(0.01, 0.03)
                    elif regime == "bear":
                        volatility = random.uniform(0.02, 0.06)
                        trend = random.uniform(-0.03, -0.01)
                    elif regime == "sideways":
                        volatility = random.uniform(0.005, 0.02)
                        trend = random.uniform(-0.005, 0.005)
                    elif regime == "volatile":
                        volatility = random.uniform(0.05, 0.1)
                        trend = random.uniform(-0.02, 0.02)
                    else:  # recovery
                        volatility = random.uniform(0.02, 0.05)
                        trend = random.uniform(0.005, 0.02)
                    
                    cursor.execute("""
                    INSERT INTO ml_market_regimes
                    (symbol, interval, regime_start, regime_end, regime_type, 
                     volatility, trend, confidence, detection_time)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, interval, regime_start) DO NOTHING
                    """, (
                        symbol, interval, start_date, end_date, regime,
                        volatility, trend, random.uniform(0.7, 0.95), now
                    ))
                    total_inserted += 1
                
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Inserted {total_inserted} sample market regime records")
        return total_inserted > 0
    except Exception as e:
        logger.error(f"Error populating market regimes: {e}")
        if conn:
            conn.close()
        return False

def populate_detected_patterns(symbols=None, intervals=None):
    """Populate detected patterns table with sample data"""
    logger.info("Populating detected patterns table...")
    
    if not symbols:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]
    
    if not intervals:
        intervals = ["1h", "4h", "1d"]
    
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        return False
    
    try:
        # First check if table already has data
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM detected_patterns")
        count = cursor.fetchone()[0]
        
        if count > 0:
            logger.info(f"Detected patterns table already has {count} records. Skipping.")
            cursor.close()
            conn.close()
            return True
        
        # Define common chart patterns
        patterns = [
            "Double Top", "Double Bottom", "Head and Shoulders", "Inverse Head and Shoulders",
            "Triangle", "Wedge", "Flag", "Pennant", "Cup and Handle", "Rounding Bottom",
            "Rectangle"
        ]
        
        # Define common pattern implications
        implications = {
            "Double Top": "Bearish reversal pattern indicating potential downturn",
            "Double Bottom": "Bullish reversal pattern indicating potential upturn",
            "Head and Shoulders": "Bearish reversal pattern suggesting previous uptrend is ending",
            "Inverse Head and Shoulders": "Bullish reversal pattern suggesting previous downtrend is ending",
            "Triangle": "Continuation pattern showing price consolidation",
            "Wedge": "Can be reversal or continuation pattern depending on direction",
            "Flag": "Short-term continuation pattern after strong price movement",
            "Pennant": "Short-term continuation pattern similar to flag but more symmetrical",
            "Cup and Handle": "Bullish continuation pattern resembling a cup with handle",
            "Rounding Bottom": "Bullish reversal pattern indicating gradual change in trend",
            "Rectangle": "Consolidation pattern indicating a trading range"
        }
        
        # Create sample pattern detections
        total_inserted = 0
        now = datetime.now()
        
        for symbol in symbols:
            for interval in intervals:
                # Get historical data
                df = get_historical_data(symbol, interval, 60)
                if df is None or len(df) < 20:
                    logger.warning(f"Insufficient historical data for {symbol}/{interval}. Skipping.")
                    continue
                
                # Create pattern detections
                for i in range(5):  # 5 patterns per symbol/interval
                    pattern = random.choice(patterns)
                    start_time = now - timedelta(days=random.randint(5, 15))
                    
                    # Pattern properties
                    is_bullish = pattern in ["Double Bottom", "Inverse Head and Shoulders", "Cup and Handle", "Rounding Bottom"]
                    is_bearish = pattern in ["Double Top", "Head and Shoulders"]
                    is_continuation = pattern in ["Triangle", "Flag", "Pennant", "Rectangle"]
                    
                    if is_bullish:
                        expected_direction = "up"
                        target_change = random.uniform(0.05, 0.15)
                    elif is_bearish:
                        expected_direction = "down"
                        target_change = random.uniform(-0.15, -0.05)
                    else:
                        expected_direction = random.choice(["up", "down"])
                        target_change = random.uniform(-0.1, 0.1)
                    
                    cursor.execute("""
                    INSERT INTO detected_patterns
                    (symbol, interval, pattern_type, start_time, end_time, 
                     detection_time, confidence, expected_direction, target_change,
                     description, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, interval, pattern_type, start_time) DO NOTHING
                    """, (
                        symbol, interval, pattern, 
                        start_time, start_time + timedelta(days=random.randint(1, 5)),
                        now, random.uniform(0.7, 0.95), expected_direction, target_change,
                        implications.get(pattern, "Chart pattern detected by algorithm"),
                        random.choice(["active", "confirmed", "failed", "completed"])
                    ))
                    total_inserted += 1
                
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Inserted {total_inserted} sample detected patterns")
        return total_inserted > 0
    except Exception as e:
        logger.error(f"Error populating detected patterns: {e}")
        if conn:
            conn.close()
        return False

def populate_benchmarks():
    """Populate benchmarks table with sample data"""
    logger.info("Populating benchmarks table...")
    
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        return False
    
    try:
        # First check if table already has data
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM benchmarks")
        count = cursor.fetchone()[0]
        
        if count > 0:
            logger.info(f"Benchmarks table already has {count} records. Skipping.")
            cursor.close()
            conn.close()
            return True
        
        # Create sample benchmark results
        strategies = [
            "RSI_Strategy", "MACD_Strategy", "Bollinger_Strategy", 
            "EMA_Crossover", "Stochastic_RSI", "Combined_Strategy"
        ]
        
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]
        intervals = ["1h", "4h", "1d"]
        
        total_inserted = 0
        now = datetime.now()
        
        for strategy in strategies:
            for symbol in symbols:
                for interval in intervals:
                    # Generate random performance metrics
                    profit_factor = random.uniform(0.8, 2.5)
                    win_rate = random.uniform(0.4, 0.7)
                    max_drawdown = random.uniform(0.1, 0.4)
                    
                    cursor.execute("""
                    INSERT INTO benchmarks
                    (strategy_name, symbol, interval, start_date, end_date, 
                     total_trades, winning_trades, losing_trades, profit_factor,
                     win_rate, avg_win, avg_loss, max_drawdown, total_return,
                     sharpe_ratio, test_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (strategy_name, symbol, interval, start_date, end_date) DO NOTHING
                    """, (
                        strategy, symbol, interval, 
                        now - timedelta(days=90), now,
                        random.randint(30, 200),
                        int(random.randint(30, 200) * win_rate),
                        int(random.randint(30, 200) * (1 - win_rate)),
                        profit_factor,
                        win_rate,
                        random.uniform(0.01, 0.1),
                        random.uniform(0.01, 0.05),
                        max_drawdown,
                        random.uniform(-0.2, 0.8),
                        random.uniform(0, 2.5),
                        now
                    ))
                    total_inserted += 1
                
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Inserted {total_inserted} sample benchmark records")
        return total_inserted > 0
    except Exception as e:
        logger.error(f"Error populating benchmarks: {e}")
        if conn:
            conn.close()
        return False

def populate_news_data():
    """Populate news data table with sample crypto news"""
    logger.info("Populating news data table...")
    
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        return False
    
    try:
        # First check if table already has data
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM news_data")
        count = cursor.fetchone()[0]
        
        if count > 0:
            logger.info(f"News data table already has {count} records. Skipping.")
            cursor.close()
            conn.close()
            return True
        
        # Sample news data
        news_items = [
            {
                "symbol": "BTCUSDT",
                "title": "Bitcoin Surpasses $50,000 Mark After Months of Consolidation",
                "content": "Bitcoin has broken through the psychological barrier of $50,000 after several months of consolidation. Analysts attribute this movement to increased institutional adoption and positive regulatory developments.",
                "source": "CryptoNewsDigest",
                "author": "Alex Johnson",
                "sentiment_score": 0.8,
                "relevance_score": 0.9
            },
            {
                "symbol": "BTCUSDT",
                "title": "Major Bank Announces Bitcoin Custody Services",
                "content": "A major Wall Street bank has announced it will begin offering Bitcoin custody services to its clients, marking another step in the mainstream adoption of cryptocurrency.",
                "source": "Financial Times",
                "author": "Sarah Williams",
                "sentiment_score": 0.75,
                "relevance_score": 0.85
            },
            {
                "symbol": "ETHUSDT",
                "title": "Ethereum Completes Major Network Upgrade",
                "content": "Ethereum has successfully completed its latest network upgrade, which aims to reduce gas fees and increase transaction throughput. The update has been well-received by developers and users alike.",
                "source": "DeFi Daily",
                "author": "Michael Chen",
                "sentiment_score": 0.85,
                "relevance_score": 0.9
            },
            {
                "symbol": "ETHUSDT",
                "title": "DeFi Projects on Ethereum Reach New Milestone",
                "content": "The total value locked in decentralized finance projects on Ethereum has reached a new all-time high, showing continued growth in the ecosystem despite market volatility.",
                "source": "Blockchain Insider",
                "author": "Emma Davis",
                "sentiment_score": 0.7,
                "relevance_score": 0.8
            },
            {
                "symbol": "BNBUSDT",
                "title": "Binance Coin Rallies Following Exchange's Expansion",
                "content": "Binance Coin (BNB) has seen significant price appreciation following the exchange's announcement of expansion into new markets and the introduction of new features on its platform.",
                "source": "Crypto Briefing",
                "author": "Robert Miller",
                "sentiment_score": 0.75,
                "relevance_score": 0.85
            },
            {
                "symbol": "ADAUSDT",
                "title": "Cardano Announces New Smart Contract Capabilities",
                "content": "The Cardano Foundation has announced new smart contract capabilities that will allow developers to build more complex applications on the blockchain, potentially increasing its competitiveness with Ethereum.",
                "source": "Crypto Times",
                "author": "Laura Stevens",
                "sentiment_score": 0.8,
                "relevance_score": 0.9
            },
            {
                "symbol": "BTCUSDT",
                "title": "Regulatory Concerns Impact Crypto Markets",
                "content": "Recent statements from regulators have caused uncertainty in cryptocurrency markets, with Bitcoin experiencing increased volatility as traders react to potential regulatory changes.",
                "source": "Market Watch",
                "author": "Daniel Thompson",
                "sentiment_score": 0.3,
                "relevance_score": 0.85
            },
            {
                "symbol": "ETHUSDT",
                "title": "Major Ethereum Update Delayed",
                "content": "Developers have announced a delay in the upcoming Ethereum update, citing the need for additional testing to ensure security and stability of the network.",
                "source": "Tech Insights",
                "author": "Jennifer Adams",
                "sentiment_score": 0.4,
                "relevance_score": 0.8
            },
            {
                "symbol": "BTCUSDT",
                "title": "Mining Difficulty Reaches All-Time High",
                "content": "Bitcoin mining difficulty has reached an all-time high, indicating strong network security but potentially impacting miner profitability in the short term.",
                "source": "Mining Weekly",
                "author": "Thomas Wilson",
                "sentiment_score": 0.6,
                "relevance_score": 0.75
            },
            {
                "symbol": "BTCUSDT",
                "title": "Institutional Investors Increase Bitcoin Allocations",
                "content": "A new survey shows that institutional investors are increasing their allocations to Bitcoin, with many viewing it as a hedge against inflation and monetary policy risks.",
                "source": "Investment Times",
                "author": "Rachel Brown",
                "sentiment_score": 0.85,
                "relevance_score": 0.9
            }
        ]
        
        # Insert news items with dates spread over the last month
        now = datetime.now()
        total_inserted = 0
        
        for i, news in enumerate(news_items):
            # Spread news over last month
            published_at = now - timedelta(days=i*3)
            
            cursor.execute("""
            INSERT INTO news_data
            (symbol, title, content, source, author, published_at, 
             sentiment_score, relevance_score, url)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source, title, published_at) DO NOTHING
            """, (
                news["symbol"],
                news["title"],
                news["content"],
                news["source"],
                news["author"],
                published_at,
                news["sentiment_score"],
                news["relevance_score"],
                f"https://example.com/news/{i}"  # Dummy URL
            ))
            total_inserted += 1
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Inserted {total_inserted} sample news items")
        return total_inserted > 0
    except Exception as e:
        logger.error(f"Error populating news data: {e}")
        if conn:
            conn.close()
        return False

def populate_gap_analysis(symbols=None, intervals=None):
    """Populate ML gap analysis table with sample data"""
    logger.info("Populating ML gap analysis table...")
    
    if not symbols:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]
    
    if not intervals:
        intervals = ["1h", "4h", "1d"]
    
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        return False
    
    try:
        # First check if table already has data
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ml_gap_analysis")
        count = cursor.fetchone()[0]
        
        if count > 0:
            logger.info(f"ML gap analysis table already has {count} records. Skipping.")
            cursor.close()
            conn.close()
            return True
        
        # Create sample gap analysis data
        total_inserted = 0
        now = datetime.now()
        
        for symbol in symbols:
            for interval in intervals:
                # Get historical data
                df = get_historical_data(symbol, interval, 60)
                if df is None or len(df) < 30:
                    logger.warning(f"Insufficient historical data for {symbol}/{interval}. Skipping.")
                    continue
                
                # Create gap detections
                for i in range(3):  # 3 gaps per symbol/interval
                    gap_type = random.choice(["overnight", "weekend", "common", "exhaustion", "breakaway"])
                    
                    # Gap properties
                    if gap_type in ["breakaway", "continuation"]:
                        filled = random.choice([True, False])
                        significance = random.uniform(0.7, 0.95)
                    else:
                        filled = random.choice([True, False, True])  # More likely to be filled
                        significance = random.uniform(0.5, 0.85)
                        
                    gap_size = random.uniform(0.01, 0.05)
                    
                    cursor.execute("""
                    INSERT INTO ml_gap_analysis
                    (symbol, interval, gap_start_time, gap_end_time, detection_time,
                     gap_type, gap_size, gap_filled, fill_time, significance, notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, interval, gap_start_time) DO NOTHING
                    """, (
                        symbol, interval, 
                        now - timedelta(days=random.randint(5, 25)),
                        now - timedelta(days=random.randint(4, 24)),
                        now, gap_type, gap_size, filled,
                        now - timedelta(days=random.randint(1, 10)) if filled else None,
                        significance,
                        f"Sample {gap_type} gap detected during analysis"
                    ))
                    total_inserted += 1
                
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Inserted {total_inserted} sample gap analysis records")
        return total_inserted > 0
    except Exception as e:
        logger.error(f"Error populating gap analysis: {e}")
        if conn:
            conn.close()
        return False

def populate_all_essential_tables():
    """Populate all essential tables with sample data"""
    logger.info("Starting essential tables population...")
    
    success_count = 0
    total_tables = 8
    
    # Populate ML predictions
    if populate_ml_predictions():
        success_count += 1
        logger.info("ML predictions table populated successfully")
    else:
        logger.warning("Failed to populate ML predictions table")
    
    # Populate ML model performance
    if populate_ml_model_performance():
        success_count += 1
        logger.info("ML model performance table populated successfully")
    else:
        logger.warning("Failed to populate ML model performance table")
    
    # Populate ML price predictions
    if populate_ml_price_predictions():
        success_count += 1
        logger.info("ML price predictions table populated successfully")
    else:
        logger.warning("Failed to populate ML price predictions table")
    
    # Populate market regimes
    if populate_ml_market_regimes():
        success_count += 1
        logger.info("Market regimes table populated successfully")
    else:
        logger.warning("Failed to populate market regimes table")
    
    # Populate detected patterns
    if populate_detected_patterns():
        success_count += 1
        logger.info("Detected patterns table populated successfully")
    else:
        logger.warning("Failed to populate detected patterns table")
    
    # Populate benchmarks
    if populate_benchmarks():
        success_count += 1
        logger.info("Benchmarks table populated successfully")
    else:
        logger.warning("Failed to populate benchmarks table")
    
    # Populate news data
    if populate_news_data():
        success_count += 1
        logger.info("News data table populated successfully")
    else:
        logger.warning("Failed to populate news data table")
    
    # Populate gap analysis
    if populate_gap_analysis():
        success_count += 1
        logger.info("Gap analysis table populated successfully")
    else:
        logger.warning("Failed to populate gap analysis table")
    
    logger.info(f"Essential tables population completed. {success_count}/{total_tables} tables populated successfully.")
    
    return success_count == total_tables

if __name__ == "__main__":
    try:
        print("="*80)
        print("Essential Tables Population Script")
        print("="*80)
        
        # Populate all tables
        result = populate_all_essential_tables()
        
        if result:
            print("\n✅ All essential tables populated successfully!")
        else:
            print("\n⚠️ Some tables could not be populated. Check the logs for details.")
        
        print("="*80)
    except Exception as e:
        logger.error(f"Unexpected error during population: {e}")
        print(f"\n❌ Error: {e}")