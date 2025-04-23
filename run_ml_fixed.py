"""
Wrapper script to run ML analysis with standalone database-only data access.
This completely breaks the circular dependency between modules by using
a self-contained data retrieval system for ML operations.
"""

import sys
import logging
import argparse
import os
import pandas as pd
from datetime import datetime, timedelta

# Set up argument parsing
parser = argparse.ArgumentParser(description="Run ML analysis with standalone database-only data access")
parser.add_argument('--analyze', action='store_true', help='Analyze patterns across markets')
parser.add_argument('--train', action='store_true', help='Train pattern recognition models')
parser.add_argument('--save', action='store_true', help='Save recommendations as trading signals')
parser.add_argument('--symbol', type=str, help='Specific symbol to analyze (e.g., BTCUSDT)')
parser.add_argument('--interval', type=str, help='Specific interval to analyze (e.g., 1h)')
parser.add_argument('--days', type=int, default=90, help='Number of days of historical data to use')
parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
args = parser.parse_args()

# Configure logging
log_level = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ml_training.log')
    ]
)

print("Applying standalone database-only ML fix...")

# Import and apply our direct_ml_fix first - this is CRITICAL
# It breaks the circular dependency between database_extensions and binance_api
import direct_ml_fix
direct_ml_fix.fix_ml_modules()

# We'll use our own db-only get_historical_data function to prepare data for ML
def get_ml_ready_data(symbol, interval, lookback_days=90):
    """
    Get properly prepared data for ML with indicators, using our standalone
    database-only data retrieval function to avoid any circular dependencies.
    
    Args:
        symbol: Trading pair symbol
        interval: Time interval
        lookback_days: Number of days of historical data to use
        
    Returns:
        DataFrame with OHLCV data, indicators, and features, or empty DataFrame if error
    """
    try:
        # Get raw data using our standalone database-only function
        logging.info(f"Getting ML data for {symbol}/{interval} using standalone function")
        df = direct_ml_fix.get_ml_data(symbol, interval, lookback_days)
        
        if df.empty:
            logging.warning(f"No data found for {symbol}/{interval}")
            return pd.DataFrame()
        
        logging.info(f"Successfully prepared ML data for {symbol}/{interval} with {len(df)} records")
        return df
    
    except Exception as e:
        logging.error(f"Error preparing ML data for {symbol}/{interval}: {e}")
        return pd.DataFrame()

def get_popular_symbols(limit=10):
    """Get popular cryptocurrency symbols using direct database query"""
    # Default list of popular symbols
    default_symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT",
        "XRPUSDT", "DOTUSDT", "UNIUSDT", "LTCUSDT", "LINKUSDT"
    ]
    
    # If a specific symbol was provided via command line, use only that
    if args.symbol:
        return [args.symbol]
    
    if limit < len(default_symbols):
        return default_symbols[:limit]
    
    return default_symbols

def get_common_intervals():
    """Get common trading intervals"""
    # Default intervals including 30m which is often populated first
    intervals = ['30m', '1h', '4h', '1d']
    
    # If a specific interval was provided via command line, use only that
    if args.interval:
        return [args.interval]
    
    return intervals

def run_pattern_analysis():
    """Run pattern analysis using our standalone data retrieval"""
    print("\n=== Running Pattern Analysis ===")
    
    try:
        # First import the ML modules after our fix is applied
        import advanced_ml
        
        # Use our helper functions to get symbols and intervals
        symbols = get_popular_symbols()
        intervals = get_common_intervals()
        
        print(f"Analyzing patterns for {len(symbols)} symbols and {len(intervals)} intervals...")
        
        # Initialize the pattern analyzer with explicit data retrieval
        analyzer = advanced_ml.MultiSymbolPatternAnalyzer()
        
        # Prepare data for each symbol/interval combination
        data_dict = {}
        for symbol in symbols:
            for interval in intervals:
                print(f"  Preparing data for {symbol}/{interval}...")
                df = get_ml_ready_data(symbol, interval, args.days)
                if not df.empty:
                    data_dict[(symbol, interval)] = df
                    print(f"    ✓ Got {len(df)} records with indicators")
                else:
                    print(f"    ✗ No data available")
        
        # If we have data, analyze patterns
        if data_dict:
            print("\nAnalyzing patterns in prepared data...")
            patterns = analyzer.analyze_patterns_in_data(data_dict)
            
            if not patterns.empty:
                print(f"\n✓ Found {len(patterns)} patterns")
                print(patterns[['symbol', 'interval', 'pattern_type', 'strength', 'detected_at']].head())
                return patterns
            else:
                print("✗ No patterns detected")
        else:
            print("✗ No data available for pattern analysis")
        
        return pd.DataFrame()
        
    except Exception as e:
        logging.error(f"Error in pattern analysis: {e}")
        print(f"✗ Error analyzing patterns: {e}")
        return pd.DataFrame()

def train_pattern_models():
    """Train pattern models using our standalone data retrieval"""
    print("\n=== Training Pattern Models ===")
    
    try:
        # First import the ML modules after our fix is applied
        import advanced_ml
        
        # Use our helper functions to get symbols and intervals
        symbols = get_popular_symbols(limit=5)  # Limit for training
        intervals = get_common_intervals()
        
        print(f"Training models for {len(symbols)} symbols and {len(intervals)} intervals...")
        
        # Initialize the pattern recognition model
        model = advanced_ml.PatternRecognitionModel(rebuild_models=True)
        
        # Track training results
        results = {'successful': 0, 'failed': 0, 'total': len(symbols) * len(intervals)}
        
        # Train models for each symbol/interval combination
        for symbol in symbols:
            for interval in intervals:
                print(f"  Training model for {symbol}/{interval}...")
                df = get_ml_ready_data(symbol, interval, args.days)
                
                if not df.empty and len(df) > 100:  # Need sufficient data for training
                    try:
                        success = model.train_pattern_model(df, symbol, interval)
                        if success:
                            results['successful'] += 1
                            print(f"    ✓ Model trained successfully")
                        else:
                            results['failed'] += 1
                            print(f"    ✗ Model training failed")
                    except Exception as e:
                        results['failed'] += 1
                        print(f"    ✗ Error training model: {e}")
                else:
                    results['failed'] += 1
                    print(f"    ✗ Insufficient data for training")
        
        # Print summary
        print(f"\nTraining complete: {results['successful']}/{results['total']} models trained")
        return results
        
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        print(f"✗ Error training models: {e}")
        return {'successful': 0, 'failed': 0, 'total': 0}

def save_trading_recommendations():
    """Save high-confidence trading recommendations as signals"""
    print("\n=== Saving Trading Recommendations ===")
    
    try:
        # First run pattern analysis to get current patterns
        patterns = run_pattern_analysis()
        
        if patterns.empty:
            print("✗ No patterns to save as trading signals")
            return 0
        
        # First import the ML modules after our fix is applied
        import advanced_ml
        
        # Initialize the pattern analyzer
        analyzer = advanced_ml.MultiSymbolPatternAnalyzer()
        
        # Save high-confidence patterns (strength >= 0.75)
        high_conf_patterns = patterns[patterns['strength'] >= 0.75]
        if high_conf_patterns.empty:
            print("✗ No high-confidence patterns to save")
            return 0
        
        print(f"Saving {len(high_conf_patterns)} high-confidence patterns as trading signals...")
        saved_count = 0
        
        for idx, pattern in high_conf_patterns.iterrows():
            try:
                success = analyzer.save_trading_opportunity(pattern)
                if success:
                    saved_count += 1
            except Exception as e:
                logging.error(f"Error saving pattern: {e}")
        
        print(f"\n✓ Saved {saved_count} trading signals")
        return saved_count
        
    except Exception as e:
        logging.error(f"Error saving recommendations: {e}")
        print(f"✗ Error saving recommendations: {e}")
        return 0

# Run the requested ML operation
if args.analyze:
    print("\nRunning pattern analysis with standalone database-only data...")
    patterns = run_pattern_analysis()
    
if args.train:
    print("\nTraining ML models with standalone database-only data...")
    results = train_pattern_models()
    
if args.save:
    print("\nSaving trading recommendations with standalone database-only data...")
    saved_count = save_trading_recommendations()
    
if not (args.analyze or args.train or args.save):
    print("\nNo operation specified. Use --analyze, --train, or --save.")
    
print("\nML operation completed")