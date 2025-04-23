"""
ML Table Population Utility

This script helps populate ML tables and fix ML-related errors.
It will:
1. Train ML models for popular symbols and intervals
2. Generate predictions for those symbols and intervals
3. Save the ML data to the database

IMPORTANT: This script should be run after the database has been populated with 
price data for the selected symbols and intervals.
"""

import os
import sys
import time
import logging
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='ml_training.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

# Add console handler
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

def get_popular_symbols(limit=10):
    """Get a list of popular cryptocurrency symbols"""
    try:
        # Import here to avoid circular imports
        from database import get_available_symbols
        
        symbols = get_available_symbols(limit=limit)
        return symbols[:limit]
    except Exception as e:
        logger.error(f"Error getting popular symbols: {e}")
        # Default fallback symbols
        return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]

def populate_ml_tables(symbols=None, intervals=None, lookback_days=180):
    """
    Populate ML tables by training models and generating predictions
    
    Args:
        symbols: List of symbols to train models for (or None for popular symbols)
        intervals: List of intervals to train models for (or None for default intervals)
        lookback_days: Number of days of data to use for training
        
    Returns:
        Number of models trained successfully
    """
    try:
        from simple_ml import train_price_models, predict_prices_all
        
        # Use default symbols and intervals if not specified
        if symbols is None:
            symbols = get_popular_symbols(limit=4)
        
        if intervals is None:
            intervals = ["1h", "4h", "1d"]
            
        logger.info(f"Training ML models for {len(symbols)} symbols and {len(intervals)} intervals")
        logger.info(f"Symbols: {', '.join(symbols)}")
        logger.info(f"Intervals: {', '.join(intervals)}")
        
        # Train models for each symbol and interval
        success_count = 0
        for symbol in symbols:
            for interval in intervals:
                logger.info(f"Training ML model for {symbol}/{interval}")
                result = train_price_models(symbol, interval, lookback_days)
                if result:
                    success_count += 1
                    logger.info(f"Successfully trained model for {symbol}/{interval}")
                else:
                    logger.warning(f"Failed to train model for {symbol}/{interval}")
                
                # Sleep to avoid overwhelming the database
                time.sleep(1)
        
        # Generate predictions
        if success_count > 0:
            logger.info("Generating ML predictions for all trained models")
            prediction_count = predict_prices_all()
            logger.info(f"Generated {prediction_count} ML predictions")
        
        logger.info(f"ML table population completed. {success_count} models trained successfully.")
        return success_count
    
    except Exception as e:
        logger.error(f"Error populating ML tables: {e}")
        return 0

def populate_advanced_patterns(symbols=None, intervals=None, days=90):
    """
    Populate advanced pattern recognition tables
    
    Args:
        symbols: List of symbols to analyze (or None for popular symbols)
        intervals: List of intervals to analyze (or None for default intervals)
        days: Number of days of data to analyze
        
    Returns:
        Number of patterns detected and saved
    """
    try:
        # Import here to avoid circular imports
        import advanced_ml
        
        # Use default symbols and intervals if not specified
        if symbols is None:
            symbols = get_popular_symbols(limit=4)
        
        if intervals is None:
            intervals = ["1h", "4h", "1d"]
            
        logger.info(f"Training pattern models for {len(symbols)} symbols and {len(intervals)} intervals")
        logger.info(f"Symbols: {', '.join(symbols)}")
        logger.info(f"Intervals: {', '.join(intervals)}")
        
        # Train pattern models
        analyzer = advanced_ml.MultiSymbolPatternAnalyzer()
        result = analyzer.train_pattern_models(symbols, intervals, days)
        
        # Analyze patterns
        patterns = analyzer.analyze_all_patterns(symbols, intervals, days)
        
        # Save recommendations
        if patterns is not None and not patterns.empty:
            count = analyzer.save_all_recommendations(min_strength=0.7)
            logger.info(f"Saved {count} pattern recommendations")
            return count
        else:
            logger.warning("No patterns detected")
            return 0
    
    except Exception as e:
        logger.error(f"Error populating advanced pattern tables: {e}")
        return 0

def main():
    """Main entry point"""
    try:
        logger.info("Starting ML table population")
        
        # Populate simple ML tables
        success_count = populate_ml_tables()
        logger.info(f"Populated simple ML tables: {success_count} models trained")
        
        # Populate advanced pattern tables
        if success_count > 0:
            pattern_count = populate_advanced_patterns()
            logger.info(f"Populated advanced pattern tables: {pattern_count} patterns detected")
        
        logger.info("ML table population completed")
        return True
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        return False

if __name__ == "__main__":
    main()