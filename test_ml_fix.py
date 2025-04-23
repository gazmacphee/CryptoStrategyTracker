#!/usr/bin/env python
"""
Test script to verify the ML pattern model training fix
"""

import logging
import sys
from advanced_ml import train_all_pattern_models

# Configure logging to show in console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ml_test")

if __name__ == "__main__":
    logger.info("Testing ML pattern model training...")
    logger.info("Initializing pattern model training...")
    
    try:
        logger.info("Calling train_all_pattern_models()...")
        train_results = train_all_pattern_models()
        
        logger.info(f"Got result type: {type(train_results)}")
        logger.info(f"Result keys: {train_results.keys() if isinstance(train_results, dict) else 'Not a dictionary'}")
        
        logger.info(f"Training completed: {train_results.get('successful', 0)}/{train_results.get('total', 0)} models trained")
        logger.info(f"Full results: {train_results}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())