#!/usr/bin/env python3
"""
Comprehensive integration test for ML components.
This script tests all ML functionality to ensure everything works together properly.
"""

import os
import sys
import logging
import time
import argparse
import psycopg2
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_integration_test.log')
    ]
)
logger = logging.getLogger()

class MlIntegrationTester:
    """Test runner for ML component integration tests"""
    
    def __init__(self, verbose=False):
        """Initialize the tester"""
        self.verbose = verbose
        self.test_results = {}
        self.db_url = os.environ.get("DATABASE_URL")
        if not self.db_url:
            logger.error("DATABASE_URL environment variable not set")
        
    def _run_command(self, command, description):
        """Run a command and log the output"""
        logger.info(f"Running: {description}")
        if self.verbose:
            logger.info(f"Command: {' '.join(command)}")
            
        try:
            proc = subprocess.run(command, capture_output=True, text=True)
            
            if self.verbose:
                for line in proc.stdout.splitlines():
                    logger.info(f"  {line}")
                    
            if proc.returncode != 0:
                logger.error(f"Command failed with return code {proc.returncode}")
                for line in proc.stderr.splitlines():
                    logger.error(f"  {line}")
                return False, proc.stdout, proc.stderr
            
            return True, proc.stdout, proc.stderr
            
        except Exception as e:
            logger.error(f"Error running command: {e}")
            return False, "", str(e)
    
    def test_database_connection(self):
        """Test database connection"""
        test_name = "Database Connection"
        logger.info(f"Running test: {test_name}")
        
        try:
            if not self.db_url:
                self.test_results[test_name] = {
                    "success": False,
                    "message": "DATABASE_URL environment variable not set"
                }
                return False
                
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            cursor.execute("SELECT NOW()")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            self.test_results[test_name] = {
                "success": True,
                "message": f"Successfully connected to database. Server time: {result[0]}"
            }
            logger.info(f"Test {test_name}: PASSED")
            return True
            
        except Exception as e:
            self.test_results[test_name] = {
                "success": False,
                "message": f"Error connecting to database: {e}"
            }
            logger.error(f"Test {test_name}: FAILED - {e}")
            return False
    
    def test_recursion_fix(self):
        """Test the recursion fix implementation"""
        test_name = "ML Recursion Fix"
        logger.info(f"Running test: {test_name}")
        
        try:
            # Import the direct_ml_fix module
            sys.path.append(os.getcwd())
            import direct_ml_fix
            
            # Check if it has the required functions
            required_functions = ['get_historical_data_direct_from_db', 'fix_ml_modules']
            missing_functions = [f for f in required_functions if not hasattr(direct_ml_fix, f)]
            
            if missing_functions:
                self.test_results[test_name] = {
                    "success": False,
                    "message": f"Missing required functions: {', '.join(missing_functions)}"
                }
                logger.error(f"Test {test_name}: FAILED - Missing functions")
                return False
                
            # Apply the fix
            fixed = direct_ml_fix.fix_ml_modules()
            
            self.test_results[test_name] = {
                "success": True,
                "message": "Successfully applied ML recursion fix"
            }
            logger.info(f"Test {test_name}: PASSED")
            return True
            
        except Exception as e:
            self.test_results[test_name] = {
                "success": False,
                "message": f"Error applying ML recursion fix: {e}"
            }
            logger.error(f"Test {test_name}: FAILED - {e}")
            return False
    
    def test_data_availability(self):
        """Test if historical data is available in the database"""
        test_name = "Historical Data Availability"
        logger.info(f"Running test: {test_name}")
        
        try:
            if not self.db_url:
                self.test_results[test_name] = {
                    "success": False,
                    "message": "DATABASE_URL environment variable not set"
                }
                return False
                
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            
            # Check for data in historical_data table
            cursor.execute("SELECT COUNT(*) FROM historical_data")
            count = cursor.fetchone()[0]
            
            # If no data, check if backfill is running
            if count == 0:
                cursor.execute("SELECT COUNT(*) FROM pg_stat_activity WHERE query LIKE '%INSERT INTO historical_data%'")
                backfill_count = cursor.fetchone()[0]
                
                cursor.close()
                conn.close()
                
                if backfill_count > 0:
                    self.test_results[test_name] = {
                        "success": False,
                        "message": f"No historical data yet, but backfill is running. {backfill_count} backfill processes active."
                    }
                else:
                    self.test_results[test_name] = {
                        "success": False,
                        "message": "No historical data and no backfill processes running"
                    }
                logger.warning(f"Test {test_name}: WAITING - No data yet")
                return False
            
            # Check which symbols and intervals have data
            cursor.execute("""
                SELECT symbol, interval, COUNT(*) as count 
                FROM historical_data 
                GROUP BY symbol, interval
                ORDER BY count DESC
            """)
            symbol_data = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # Format the results
            if symbol_data:
                symbol_info = []
                for symbol, interval, count in symbol_data:
                    symbol_info.append(f"{symbol}/{interval}: {count} records")
                
                self.test_results[test_name] = {
                    "success": True,
                    "message": f"Historical data available: {len(symbol_data)} symbol/interval combinations. Top: {', '.join(symbol_info[:3])}"
                }
                logger.info(f"Test {test_name}: PASSED")
                return True
            else:
                self.test_results[test_name] = {
                    "success": False,
                    "message": f"No symbol/interval combinations found despite {count} total records"
                }
                logger.warning(f"Test {test_name}: FAILED - No symbol data")
                return False
                
        except Exception as e:
            self.test_results[test_name] = {
                "success": False,
                "message": f"Error checking historical data: {e}"
            }
            logger.error(f"Test {test_name}: FAILED - {e}")
            return False
    
    def test_model_training(self):
        """Test ML model training with a sample symbol/interval"""
        test_name = "ML Model Training"
        logger.info(f"Running test: {test_name}")
        
        # Test model training on first available symbol with data
        try:
            if not self.db_url:
                self.test_results[test_name] = {
                    "success": False,
                    "message": "DATABASE_URL environment variable not set"
                }
                return False
                
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            
            # Find a symbol/interval with sufficient data
            cursor.execute("""
                SELECT symbol, interval, COUNT(*) as count 
                FROM historical_data 
                GROUP BY symbol, interval
                HAVING COUNT(*) >= 100
                ORDER BY count DESC
                LIMIT 1
            """)
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if not result:
                self.test_results[test_name] = {
                    "success": False,
                    "message": "No symbol/interval with sufficient data (min 100 records)"
                }
                logger.warning(f"Test {test_name}: WAITING - Insufficient data")
                return False
                
            symbol, interval, count = result
            
            # Run the model training command
            command = [
                "python", "run_ml_fixed.py",
                "--train",
                "--symbol", symbol,
                "--interval", interval,
                "--verbose"
            ]
            
            success, stdout, stderr = self._run_command(
                command, 
                f"Training ML model for {symbol}/{interval} with {count} records"
            )
            
            if success:
                # Check if model file was created
                model_path = f"models/pattern_recognition/pattern_model_{symbol}_{interval}.joblib"
                if os.path.exists(model_path):
                    self.test_results[test_name] = {
                        "success": True,
                        "message": f"Successfully trained model for {symbol}/{interval}"
                    }
                    logger.info(f"Test {test_name}: PASSED")
                    return True
                else:
                    self.test_results[test_name] = {
                        "success": False,
                        "message": f"Model training reported success but no model file found at {model_path}"
                    }
                    logger.error(f"Test {test_name}: FAILED - No model file")
                    return False
            else:
                self.test_results[test_name] = {
                    "success": False,
                    "message": f"Model training failed for {symbol}/{interval}"
                }
                logger.error(f"Test {test_name}: FAILED - Training error")
                return False
                
        except Exception as e:
            self.test_results[test_name] = {
                "success": False,
                "message": f"Error during model training test: {e}"
            }
            logger.error(f"Test {test_name}: FAILED - {e}")
            return False
    
    def test_pattern_analysis(self):
        """Test pattern analysis functionality"""
        test_name = "Pattern Analysis"
        logger.info(f"Running test: {test_name}")
        
        # Run the pattern analysis command
        command = [
            "python", "run_ml_fixed.py",
            "--analyze",
            "--verbose"
        ]
        
        success, stdout, stderr = self._run_command(
            command, 
            "Running pattern analysis across markets"
        )
        
        if success:
            self.test_results[test_name] = {
                "success": True,
                "message": "Pattern analysis ran successfully"
            }
            logger.info(f"Test {test_name}: PASSED")
            return True
        else:
            self.test_results[test_name] = {
                "success": False,
                "message": "Pattern analysis failed to run"
            }
            logger.error(f"Test {test_name}: FAILED")
            return False
    
    def test_save_recommendations(self):
        """Test saving pattern recommendations as trading signals"""
        test_name = "Save Recommendations"
        logger.info(f"Running test: {test_name}")
        
        # Run the save recommendations command
        command = [
            "python", "run_ml_fixed.py",
            "--save",
            "--verbose"
        ]
        
        success, stdout, stderr = self._run_command(
            command, 
            "Saving trading recommendations"
        )
        
        if success:
            self.test_results[test_name] = {
                "success": True,
                "message": "Saving recommendations completed successfully"
            }
            logger.info(f"Test {test_name}: PASSED")
            return True
        else:
            self.test_results[test_name] = {
                "success": False,
                "message": "Failed to save recommendations"
            }
            logger.error(f"Test {test_name}: FAILED")
            return False
    
    def test_integration_scripts(self):
        """Test the integration scripts we created"""
        test_name = "Integration Scripts"
        logger.info(f"Running test: {test_name}")
        
        # Check if our scripts exist
        scripts = [
            "train_all_pattern_models.py",
            "analyze_patterns_and_save.py",
            "update_ml_process_config.py",
            "check_ml_data.py"
        ]
        
        missing_scripts = [s for s in scripts if not os.path.exists(s)]
        
        if missing_scripts:
            self.test_results[test_name] = {
                "success": False,
                "message": f"Missing integration scripts: {', '.join(missing_scripts)}"
            }
            logger.error(f"Test {test_name}: FAILED - Missing scripts")
            return False
            
        # Try running the process config update script
        command = ["python", "update_ml_process_config.py"]
        
        success, stdout, stderr = self._run_command(
            command, 
            "Updating process manager config for ML tasks"
        )
        
        if success:
            self.test_results[test_name] = {
                "success": True,
                "message": "Successfully ran integration scripts"
            }
            logger.info(f"Test {test_name}: PASSED")
            return True
        else:
            self.test_results[test_name] = {
                "success": False,
                "message": "Failed to run integration scripts"
            }
            logger.error(f"Test {test_name}: FAILED")
            return False
    
    def run_all_tests(self):
        """Run all integration tests"""
        logger.info("Starting ML integration tests")
        
        # Tests to run in order
        tests = [
            self.test_database_connection,
            self.test_recursion_fix,
            self.test_data_availability,
            self.test_model_training,
            self.test_pattern_analysis,
            self.test_save_recommendations,
            self.test_integration_scripts
        ]
        
        # Run all tests and track pass/fail counts
        passed = 0
        failed = 0
        waiting = 0
        
        for test_func in tests:
            result = test_func()
            
            if result:
                passed += 1
            else:
                test_name = test_func.__name__.replace('test_', '')
                test_result = self.test_results.get(test_name.replace('_', ' ').title(), {})
                message = test_result.get('message', '')
                
                if 'waiting' in message.lower() or 'no data' in message.lower():
                    waiting += 1
                else:
                    failed += 1
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("ML INTEGRATION TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Tests: {len(tests)}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Waiting: {waiting}")
        logger.info("="*60)
        
        for name, result in self.test_results.items():
            status = "PASS" if result['success'] else "FAIL"
            logger.info(f"{name}: {status} - {result['message']}")
        
        logger.info("="*60)
        
        return passed, failed, waiting

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Integration Tests")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--wait", action="store_true", help="Wait for data if none is available")
    parser.add_argument("--wait-time", type=int, default=5, help="Minutes to wait for data")
    args = parser.parse_args()
    
    # Run tests
    tester = MlIntegrationTester(verbose=args.verbose)
    
    if args.wait:
        # Initial test to check data availability
        tester.test_data_availability()
        
        # If no data available, wait and check periodically
        wait_count = 0
        while wait_count < args.wait_time:
            # If data is available, run all tests
            if tester.test_data_availability():
                passed, failed, waiting = tester.run_all_tests()
                sys.exit(0 if failed == 0 else 1)
                
            # Wait before trying again
            wait_count += 1
            logger.info(f"Waiting for data to be available... (attempt {wait_count}/{args.wait_time})")
            time.sleep(60)  # Wait 1 minute
            
        logger.error(f"Timed out waiting for data after {args.wait_time} minutes")
        sys.exit(1)
    else:
        # Run all tests immediately
        passed, failed, waiting = tester.run_all_tests()
        sys.exit(0 if failed == 0 else 1)