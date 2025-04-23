#!/usr/bin/env python3
"""
Script to update Process Manager configuration to include ML-related tasks.
This ensures our ML models are automatically trained and run on a schedule.
"""

import os
import json
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# File that stores process information
PROCESS_INFO_FILE = '.process_info.json'

def load_process_info():
    """Load the current process information from file"""
    if os.path.exists(PROCESS_INFO_FILE):
        try:
            with open(PROCESS_INFO_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading process info: {e}")
            return {}
    else:
        logger.warning(f"Process info file {PROCESS_INFO_FILE} not found")
        return {}

def save_process_info(process_info):
    """Save updated process information to file"""
    try:
        with open(PROCESS_INFO_FILE, 'w') as f:
            json.dump(process_info, f, indent=2)
        logger.info(f"Process info saved to {PROCESS_INFO_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving process info: {e}")
        return False

def update_ml_process_config():
    """Update process manager configuration to include ML-related tasks"""
    # Load current process info
    process_info = load_process_info()
    
    # Get current processes
    processes = process_info.get('processes', {})
    
    # Update or add ML model training process
    ml_training_process = {
        'name': 'ML Model Training',
        'id': 'ml_training',
        'command': 'python train_all_pattern_models.py --min-records 500',
        'autostart': False,
        'type': 'scheduled',
        'schedule': {
            'interval': 'daily',
            'time': '04:00'  # Run at 4 AM
        },
        'last_run': datetime.now().isoformat(),
        'next_run': (datetime.now() + timedelta(days=1)).replace(hour=4, minute=0, second=0).isoformat(),
        'dependencies': [],
        'restarts': 0
    }
    
    # Update or add ML pattern analysis process
    ml_pattern_analysis = {
        'name': 'ML Pattern Analysis',
        'id': 'ml_pattern_analysis',
        'command': 'python analyze_patterns_and_save.py --output pattern_analysis_results.json',
        'autostart': False,
        'type': 'scheduled',
        'schedule': {
            'interval': 'hourly',
            'minute': 15  # Run at 15 minutes past the hour
        },
        'last_run': datetime.now().isoformat(),
        'next_run': (datetime.now() + timedelta(hours=1)).replace(minute=15, second=0).isoformat(),
        'dependencies': [],
        'restarts': 0
    }
    
    # Update the process dictionary
    processes['ml_training'] = ml_training_process
    processes['ml_pattern_analysis'] = ml_pattern_analysis
    
    # Update process_info
    process_info['processes'] = processes
    
    # Save updated process info
    success = save_process_info(process_info)
    
    if success:
        logger.info("ML process configuration updated successfully")
        logger.info(f"Added ML Model Training: Daily at 4 AM - {ml_training_process['command']}")
        logger.info(f"Added ML Pattern Analysis: Hourly at :15 - {ml_pattern_analysis['command']}")
    else:
        logger.error("Failed to update ML process configuration")
        
    return success

if __name__ == "__main__":
    logger.info("Updating process manager configuration to include ML tasks")
    update_ml_process_config()
    logger.info("Configuration update complete")