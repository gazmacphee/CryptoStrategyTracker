#!/usr/bin/env python
"""
Trading Signal Generator Script

This script analyzes historical price data and generates trading signals
continuously or on-demand.

Usage:
    python generate_signals.py [--continuous]
    
    --continuous: Run in continuous mode, generating signals as new data arrives
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("signals.log"),
        logging.StreamHandler()
    ]
)

def setup_argparse():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate trading signals')
    parser.add_argument('--continuous', action='store_true', 
                      help='Run in continuous mode, generating signals as new data arrives')
    return parser.parse_args()

def generate_signals():
    """Generate trading signals from historical data."""
    try:
        logging.info("Generating trading signals...")
        
        # Import modules inside function to handle import errors gracefully
        import pandas as pd
        from trading_signals import save_trading_signals, calculate_exit_levels
        from strategy import evaluate_buy_sell_signals
        from database import get_db_connection
        from binance_api import get_available_symbols
        from indicators import add_bollinger_bands, add_macd, add_rsi, add_ema
        
        # Get available trading pairs
        available_symbols = get_available_symbols()
        
        # Define intervals to analyze
        intervals = ['1h', '4h', '1d']
        
        conn = get_db_connection()
        if not conn:
            logging.error("Failed to connect to database")
            return False
        
        cursor = conn.cursor()
        signals_generated = 0
        
        # Process each symbol and interval
        for symbol in available_symbols[:10]:  # Process top 10 symbols
            for interval in intervals:
                try:
                    logging.info(f"Processing {symbol}/{interval}")
                    
                    # Get historical data from database
                    query = """
                    SELECT timestamp, open, high, low, close, volume
                    FROM historical_data
                    WHERE symbol = %s AND interval = %s
                    ORDER BY timestamp DESC
                    LIMIT 500
                    """
                    
                    cursor.execute(query, (symbol, interval))
                    rows = cursor.fetchall()
                    
                    if not rows:
                        logging.warning(f"No data found for {symbol}/{interval}")
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df = df.sort_values('timestamp')
                    
                    # Add technical indicators
                    df = add_bollinger_bands(df)
                    df = add_macd(df)
                    df = add_rsi(df)
                    df = add_ema(df)
                    
                    # Apply strategy to generate signals
                    result_df = evaluate_buy_sell_signals(df)
                    
                    # Calculate exit levels (stop loss and take profit) for buy signals
                    for idx, row in result_df.iterrows():
                        if row.get('buy_signal'):
                            # Calculate exit levels
                            exit_levels = calculate_exit_levels(
                                df.iloc[:idx+1],  # Use data up to this point
                                row['close'],     # Entry price
                                stop_loss_method='support',
                                take_profit_method='risk_reward',
                                risk_reward_ratio=2.5
                            )
                            
                            # Add exit levels to the DataFrame
                            result_df.loc[idx, 'stop_loss'] = exit_levels['stop_loss']
                            result_df.loc[idx, 'take_profit'] = exit_levels['take_profit']
                            result_df.loc[idx, 'stop_loss_method'] = exit_levels['stop_loss_method']
                            result_df.loc[idx, 'take_profit_method'] = exit_levels['take_profit_method']
                            result_df.loc[idx, 'risk_reward_ratio'] = exit_levels['risk_reward_ratio']
                    
                    # Save signals to database
                    success = save_trading_signals(result_df, symbol, interval, strategy_name="Combined Strategy")
                    
                    if success:
                        signals = result_df[(result_df['buy_signal'] == True) | (result_df['sell_signal'] == True)]
                        signals_count = len(signals)
                        signals_generated += signals_count
                        logging.info(f"Generated {signals_count} signals for {symbol}/{interval}")
                    
                except Exception as e:
                    logging.error(f"Error generating signals for {symbol}/{interval}: {e}")
        
        if conn:
            conn.close()
        
        logging.info(f"Generated a total of {signals_generated} trading signals")
        return signals_generated > 0
    
    except ImportError as e:
        logging.error(f"Import error: {e}")
        return False
    except Exception as e:
        logging.error(f"Error generating signals: {e}")
        return False

def main():
    """Main entry point for signal generation."""
    args = setup_argparse()
    
    if args.continuous:
        logging.info("Starting continuous signal generation...")
        try:
            while True:
                generate_signals()
                logging.info("Waiting for next signal generation cycle...")
                time.sleep(3600)  # Run every hour
        except KeyboardInterrupt:
            logging.info("Signal generation interrupted by user")
    else:
        generate_signals()
        logging.info("Signal generation complete")

if __name__ == "__main__":
    main()