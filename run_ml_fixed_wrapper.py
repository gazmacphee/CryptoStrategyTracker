#!/usr/bin/env python3
"""
Wrapper for run_ml_fixed.py that fixes the column name mismatch issue.
This script intercepts the results of run_pattern_analysis() and
ensures the 'strength' column exists before passing it to save_trading_recommendations().
"""

import sys
import logging
import subprocess
import pandas as pd
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# Fix the environment by adding our local directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the original module
import run_ml_fixed

# Save the original function
original_run_pattern_analysis = run_ml_fixed.run_pattern_analysis

# Define our patched function
def patched_run_pattern_analysis():
    """Patched version of run_pattern_analysis that ensures the 'strength' column exists"""
    # Call the original function
    patterns = original_run_pattern_analysis()
    
    # Check if we have a dataframe and if it has any rows
    if not isinstance(patterns, pd.DataFrame) or patterns.empty:
        return patterns
    
    # Handle the column name differences
    if 'pattern_strength' in patterns.columns and 'strength' not in patterns.columns:
        patterns['strength'] = patterns['pattern_strength']
        logger.info("Added 'strength' column based on 'pattern_strength'")
    
    if 'strength' in patterns.columns and 'pattern_strength' not in patterns.columns:
        patterns['pattern_strength'] = patterns['strength']
        logger.info("Added 'pattern_strength' column based on 'strength'")
    
    return patterns

# Replace the original function with our patched version
run_ml_fixed.run_pattern_analysis = patched_run_pattern_analysis

# Also patch MultiSymbolPatternAnalyzer.analyze_patterns_in_data to include both columns
if hasattr(run_ml_fixed, 'advanced_ml'):
    original_analyze = run_ml_fixed.advanced_ml.MultiSymbolPatternAnalyzer.analyze_patterns_in_data
    
    def patched_analyze_patterns_in_data(self, data_dict):
        """Patched version that ensures both strength columns exist"""
        # Call the original function
        patterns = original_analyze(self, data_dict)
        
        # Check if we have a dataframe and if it has any rows
        if not isinstance(patterns, pd.DataFrame) or patterns.empty:
            return patterns
        
        # Handle the column name differences
        if 'pattern_strength' in patterns.columns and 'strength' not in patterns.columns:
            patterns['strength'] = patterns['pattern_strength']
            logger.info("Added 'strength' column based on 'pattern_strength'")
        
        if 'strength' in patterns.columns and 'pattern_strength' not in patterns.columns:
            patterns['pattern_strength'] = patterns['strength']
            logger.info("Added 'pattern_strength' column based on 'strength'")
        
        return patterns
    
    # Apply the patch
    run_ml_fixed.advanced_ml.MultiSymbolPatternAnalyzer.analyze_patterns_in_data = patched_analyze_patterns_in_data

# Print information
print("\n==== Running ML Fixed with Column Name Fix ====")
print("This wrapper ensures 'strength' and 'pattern_strength' columns both exist")

# Call the appropriate function based on arguments
if __name__ == "__main__":
    # Pass all the arguments to run_ml_fixed
    if len(sys.argv) > 1:
        # Just reuse the argument parsing logic from run_ml_fixed
        run_ml_fixed.args = run_ml_fixed.parser.parse_args()
        
        # Run the requested ML operation
        if run_ml_fixed.args.analyze:
            print("\nRunning pattern analysis with column name fix...")
            patterns = run_ml_fixed.run_pattern_analysis()
            
        if run_ml_fixed.args.train:
            print("\nTraining ML models...")
            results = run_ml_fixed.train_pattern_models()
            
        if run_ml_fixed.args.save:
            print("\nSaving trading recommendations with column name fix...")
            saved_count = run_ml_fixed.save_trading_recommendations()
            
        if not (run_ml_fixed.args.analyze or run_ml_fixed.args.train or run_ml_fixed.args.save):
            print("\nNo operation specified. Use --analyze, --train, or --save.")
            
        print("\nML operation completed")
    else:
        print("\nNo arguments provided. Use --analyze, --train, or --save.")
        print("Example: python run_ml_fixed_wrapper.py --analyze --days 1065")