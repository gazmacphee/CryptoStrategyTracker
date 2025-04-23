#!/usr/bin/env python3
"""
Integrated ML testing and analysis script.
This script provides a unified interface to run ML tests, train models,
analyze patterns, and save recommendations.
"""

import os
import argparse
import logging
import time
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_testing.log')
    ]
)
logger = logging.getLogger()

def print_header(text):
    """Print a formatted header to the console"""
    width = 80
    padding = (width - len(text) - 2) // 2
    print("\n" + "=" * width)
    print(" " * padding + text + " " * padding)
    print("=" * width + "\n")

def run_command(command, title):
    """Run a shell command and print the output"""
    print_header(title)
    logger.info(f"Running: {' '.join(command)}")
    
    try:
        # Run the command
        start_time = time.time()
        result = subprocess.run(command, capture_output=True, text=True)
        end_time = time.time()
        
        # Print the output
        print(result.stdout)
        
        # Check the result
        if result.returncode == 0:
            duration = end_time - start_time
            logger.info(f"Command completed successfully in {duration:.2f} seconds")
            print(f"\n✅ Success! Completed in {duration:.2f} seconds")
            return True
        else:
            logger.error(f"Command failed with return code {result.returncode}")
            print("\n❌ Error! Command failed with output:")
            print(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"Error running command: {e}")
        print(f"\n❌ Error: {e}")
        return False

def check_database():
    """Check if the database has data"""
    print_header("CHECKING DATABASE")
    
    try:
        # Run a test script to check for data
        command = ["python", "check_ml_data.py", "--symbol", "BTCUSDT", "--interval", "30m"]
        logger.info(f"Checking database with command: {' '.join(command)}")
        
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)
        
        # Check if there's a message about sufficient data
        if "Found" in result.stdout and "records" in result.stdout:
            print("\n✅ Database has sufficient data for ML operations")
            return True
        else:
            print("\n⚠️ Database may not have sufficient data yet")
            return False
            
    except Exception as e:
        logger.error(f"Error checking database: {e}")
        print(f"\n❌ Error checking database: {e}")
        return False

def run_integration_test(wait=False, wait_time=5):
    """Run the ML integration test suite"""
    command = ["python", "test_ml_integration.py", "--verbose"]
    
    if wait:
        command.extend(["--wait", "--wait-time", str(wait_time)])
        
    return run_command(command, "ML INTEGRATION TEST")

def train_models(min_records=500, max_symbols=None, wait=False):
    """Train all available pattern models"""
    command = ["python", "train_all_pattern_models.py", "--min-records", str(min_records)]
    
    if max_symbols:
        command.extend(["--max-symbols", str(max_symbols)])
        
    if wait:
        command.extend(["--wait", "--max-wait", "30", "--wait-interval", "5"])
        
    return run_command(command, "TRAINING PATTERN MODELS")

def analyze_patterns(save=True):
    """Analyze patterns across all markets"""
    command = ["python", "analyze_patterns_and_save.py", "--output", "latest_pattern_analysis.json"]
    
    if not save:
        command.append("--no-save")
        
    return run_command(command, "ANALYZING PATTERNS")

def run_all(wait_for_data=True):
    """Run all ML tests and operations in sequence"""
    print_header("RUNNING COMPLETE ML TEST SUITE")
    
    # Start time
    start_time = datetime.now()
    logger.info(f"Starting ML test suite at {start_time.isoformat()}")
    
    # Record successes
    success_count = 0
    total_tests = 4
    
    # Check database first
    db_success = check_database()
    
    # Only proceed with real tests if we have data or user wants to wait
    if db_success or wait_for_data:
        # Run integration test
        if run_integration_test(wait=wait_for_data, wait_time=10):
            success_count += 1
            
        # Train models
        if train_models(wait=wait_for_data):
            success_count += 1
            
        # Analyze patterns
        if analyze_patterns():
            success_count += 1
            
        # Update ML process config
        if run_command(["python", "update_ml_process_config.py"], "UPDATING ML PROCESS CONFIG"):
            success_count += 1
    else:
        logger.warning("Skipping tests due to insufficient data")
        print("\n⚠️ Skipping tests due to insufficient data")
        
    # End time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60.0
    
    # Print summary
    print_header("ML TEST SUITE SUMMARY")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration:.2f} minutes")
    print(f"Success: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("\n✅ All ML components are working correctly!")
    else:
        print(f"\n⚠️ {total_tests - success_count} test(s) failed. Check the logs for details.")
        
    return success_count == total_tests

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML tests and operations")
    parser.add_argument("--integration", action="store_true", help="Run ML integration tests")
    parser.add_argument("--train", action="store_true", help="Train pattern models")
    parser.add_argument("--analyze", action="store_true", help="Analyze patterns")
    parser.add_argument("--wait", action="store_true", help="Wait for data if not available")
    parser.add_argument("--all", action="store_true", help="Run all tests and operations")
    args = parser.parse_args()
    
    # If no arguments, run all
    if not (args.integration or args.train or args.analyze or args.all):
        args.all = True
        
    # Run the requested operations
    if args.all:
        run_all(wait_for_data=args.wait)
    else:
        if args.integration:
            run_integration_test(wait=args.wait)
        if args.train:
            train_models(wait=args.wait)
        if args.analyze:
            analyze_patterns()