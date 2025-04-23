"""
Wrapper script to run ML analysis with database-only data access.
This applies our fix before running the ML code.
"""

import sys
import logging
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Run ML analysis with database-only data access")
parser.add_argument('--analyze', action='store_true', help='Analyze patterns across markets')
parser.add_argument('--train', action='store_true', help='Train pattern recognition models')
parser.add_argument('--save', action='store_true', help='Save recommendations as trading signals')
parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
args = parser.parse_args()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

print("Applying database-only ML fix...")

# Import and apply our direct_ml_fix first
import direct_ml_fix
direct_ml_fix.fix_ml_modules()

# Now import advanced_ml after the fix is applied
import advanced_ml

# Run the requested ML operation
if args.analyze:
    print("Running pattern analysis with database-only data...")
    advanced_ml.analyze_all_market_patterns()
    
if args.train:
    print("Training ML models with database-only data...")
    advanced_ml.train_all_pattern_models()
    
if args.save:
    print("Saving trading recommendations with database-only data...")
    advanced_ml.save_current_recommendations()
    
if not (args.analyze or args.train or args.save):
    print("No operation specified. Use --analyze, --train, or --save.")
    
print("ML operation completed")