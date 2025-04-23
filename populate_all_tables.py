#!/usr/bin/env python3
"""
Script to populate all database tables for the cryptocurrency trading platform.
This script ensures all the secondary data tables like news, sentiment, ML predictions,
detected patterns, and benchmarks are populated with data.
"""
import os
import sys
import logging
import time
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def populate_sentiment_data():
    """Populate the sentiment_data table with sentiment analysis for major cryptocurrencies"""
    try:
        logging.info("Populating sentiment data...")
        from sentiment_scraper import scrape_and_save_sentiment
        symbols = ["BTC", "ETH", "BNB", "ADA"]
        count = 0
        for symbol in symbols:
            try:
                num_saved = scrape_and_save_sentiment(symbol, days=7)
                count += num_saved
                logging.info(f"Saved {num_saved} sentiment records for {symbol}")
            except Exception as e:
                logging.error(f"Error saving sentiment for {symbol}: {e}")
        return count
    except Exception as e:
        logging.error(f"Error populating sentiment data: {e}")
        return 0

def populate_news_data():
    """Populate the news_data table with recent cryptocurrency news"""
    try:
        logging.info("Populating news data...")
        from crypto_news import get_crypto_news, save_news_to_db
        # Get general crypto news
        articles = get_crypto_news(days_back=5, limit=25)
        if articles:
            save_news_to_db(articles)
            logging.info(f"Saved {len(articles)} general news articles")
        
        # Get specific news for major coins
        symbols = ["BTC", "ETH", "BNB", "ADA"]
        for symbol in symbols:
            try:
                symbol_articles = get_crypto_news(symbol=symbol, days_back=5, limit=10)
                if symbol_articles:
                    save_news_to_db(symbol_articles)
                    logging.info(f"Saved {len(symbol_articles)} news articles for {symbol}")
            except Exception as e:
                logging.error(f"Error getting news for {symbol}: {e}")
        
        return True
    except Exception as e:
        logging.error(f"Error populating news data: {e}")
        return False

def run_ml_training():
    """Train machine learning models and generate predictions"""
    try:
        logging.info("Running ML training and predictions...")
        
        # Check if we have data to work with
        from database import get_db_connection
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check for historical data
        cur.execute("SELECT COUNT(*) FROM historical_data")
        data_count = cur.fetchone()[0]
        
        if data_count < 1000:
            logging.warning(f"Not enough historical data for ML training ({data_count} records). Need at least 1000.")
            return False
            
        # Train models and generate predictions using both advanced and simple ML
        try:
            # Try the advanced ML module first
            logging.info("Running advanced ML pattern detection...")
            import advanced_ml
            training_results = advanced_ml.train_all_pattern_models()
            logging.info(f"Pattern model training completed: {training_results['successful']}/{training_results['total']} models trained")
            
            patterns = advanced_ml.analyze_all_market_patterns()
            logging.info(f"Pattern analysis complete - found {len(patterns) if not patterns.empty else 0} patterns")
            
            # Save recommendations
            saved_count = advanced_ml.save_current_recommendations()
            logging.info(f"Saved {saved_count} pattern-based trading signals")
        except Exception as e:
            logging.error(f"Error in advanced ML: {e}")
        
        # Also run the simple ML module for basic predictions
        try:
            logging.info("Running simple ML price predictions...")
            from simple_ml import train_price_models, predict_prices_all
            
            # Train models for popular pairs
            symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]
            intervals = ["1h", "4h", "1d"]
            
            for symbol in symbols:
                for interval in intervals:
                    try:
                        logging.info(f"Training ML model for {symbol}/{interval}")
                        success = train_price_models(symbol, interval)
                        if success:
                            logging.info(f"Successfully trained model for {symbol}/{interval}")
                    except Exception as e:
                        logging.error(f"Error training model for {symbol}/{interval}: {e}")
            
            # Generate price predictions
            prediction_count = predict_prices_all(symbols=symbols, intervals=intervals)
            logging.info(f"Generated {prediction_count} price predictions")
        except Exception as e:
            logging.error(f"Error in simple ML: {e}")
            
        return True
    except Exception as e:
        logging.error(f"Error running ML training: {e}")
        return False

def run_benchmarks():
    """Run strategy benchmarks to populate the benchmarks table"""
    try:
        logging.info("Running strategy benchmarks...")
        
        # Import needed modules
        from strategy import backtest_strategy
        from trading_signals import get_available_strategies
        from database import save_benchmark_results
        
        # Get popular symbols and intervals
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]
        intervals = ["1h", "4h", "1d"]
        
        # Get available strategies
        strategies = get_available_strategies()
        if not strategies:
            strategies = ["RSI", "MACD", "Bollinger", "Stoch", "Combined"]
        
        # Run benchmarks for each combination
        benchmark_count = 0
        for symbol in symbols:
            for interval in intervals:
                for strategy_name in strategies:
                    try:
                        logging.info(f"Backtesting {strategy_name} for {symbol}/{interval}")
                        # Get last 30 days of data
                        lookback_days = 30
                        
                        # Run backtest
                        results = backtest_strategy(
                            symbol=symbol,
                            interval=interval,
                            strategy=strategy_name,
                            lookback_days=lookback_days
                        )
                        
                        if results and 'performance' in results:
                            # Save benchmark results
                            save_benchmark_results(
                                symbol=symbol,
                                interval=interval,
                                strategy_name=strategy_name,
                                start_date=results.get('start_date'),
                                end_date=results.get('end_date'),
                                total_trades=results['performance'].get('total_trades', 0),
                                win_rate=results['performance'].get('win_rate', 0),
                                profit_factor=results['performance'].get('profit_factor', 0),
                                net_profit_pct=results['performance'].get('net_profit_pct', 0),
                                max_drawdown_pct=results['performance'].get('max_drawdown_pct', 0)
                            )
                            benchmark_count += 1
                            logging.info(f"Saved benchmark for {strategy_name} on {symbol}/{interval}")
                    except Exception as e:
                        logging.error(f"Error benchmarking {strategy_name} for {symbol}/{interval}: {e}")
        
        logging.info(f"Completed {benchmark_count} strategy benchmarks")
        return benchmark_count
    except Exception as e:
        logging.error(f"Error running benchmarks: {e}")
        return 0

def main():
    """Main entry point - run all population processes"""
    start_time = time.time()
    logging.info("Starting to populate all tables...")
    
    # Run all population functions
    sentiment_count = populate_sentiment_data()
    logging.info(f"Populated sentiment data: {sentiment_count} records")
    
    news_success = populate_news_data()
    logging.info(f"Populated news data: {'Success' if news_success else 'Failed'}")
    
    ml_success = run_ml_training()
    logging.info(f"ML training and predictions: {'Success' if ml_success else 'Failed'}")
    
    benchmark_count = run_benchmarks()
    logging.info(f"Strategy benchmarks: {benchmark_count} created")
    
    end_time = time.time()
    duration = end_time - start_time
    
    logging.info(f"All table population completed in {duration:.2f} seconds")
    
    # Return a summary of what was done
    return {
        "sentiment_count": sentiment_count,
        "news_populated": news_success,
        "ml_populated": ml_success,
        "benchmark_count": benchmark_count,
        "duration_seconds": duration
    }

if __name__ == "__main__":
    main()