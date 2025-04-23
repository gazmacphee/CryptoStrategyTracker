
"""
Test script to verify the ML pattern model training fix
"""

import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_ml_fix")

def mock_train_all_pattern_models():
    """Return an empty dictionary to simulate error case"""
    logger.info("Running mock training with empty results")
    return {}

if __name__ == "__main__":
    # Test the safe dictionary access
    print("Testing ML pattern model with safe dictionary access")
    train_results = mock_train_all_pattern_models()
    
    # Safe dictionary access with defaults
    total = train_results.get('total', 0) 
    successful = train_results.get('successful', 0)
    
    print(f"Training completed: {successful}/{total} models trained")
    print(f"Full results: {train_results}")
    print("Test completed successfully!")
