"""
Fix for decimal.Decimal and float type incompatibility issues.

This script adds support for converting Decimal values to float for arithmetic operations.
"""

import os
import sys
import traceback
import logging
from decimal import Decimal
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_float_values(df):
    """
    Ensures all numeric columns in a DataFrame are converted to float type.
    This prevents issues with decimal.Decimal values.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        DataFrame with all numeric columns as float
    """
    if df is None or df.empty:
        return df
        
    for col in df.select_dtypes(include=['number']).columns:
        # Check if column contains any Decimal objects
        has_decimal = False
        for x in df[col].dropna().head():
            if isinstance(x, Decimal):
                has_decimal = True
                break
                
        if has_decimal:
            df[col] = df[col].astype(float)
    
    return df

def fix_get_dxy_data():
    """Update get_dxy_data function to convert Decimal values to float"""
    try:
        filepath = 'economic_indicators.py'
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Find the get_full_dxy_data function
        start_idx = content.find("def get_full_dxy_data")
        if start_idx == -1:
            logger.error("Could not find get_full_dxy_data function in economic_indicators.py")
            return False
        
        # Find the end of the function (next def statement)
        next_def_idx = content.find("def ", start_idx + 10)
        if next_def_idx == -1:
            next_def_idx = len(content)
        
        # Find the return statement
        return_idx = content.rfind("return df", start_idx, next_def_idx)
        if return_idx == -1:
            logger.error("Could not find return statement in get_full_dxy_data function")
            return False
        
        # Check if we already added the conversion
        if "ensure_float_values" in content[start_idx:next_def_idx]:
            logger.info("Decimal conversion already added to get_full_dxy_data function")
            return True
        
        # Add conversion before the return statement
        modified_content = (
            content[:return_idx] + 
            "    # Convert any decimal.Decimal columns to float\n" +
            "    df = ensure_float_values(df)\n    \n    " + 
            content[return_idx:]
        )
        
        # Write the modified content back
        with open(filepath, 'w') as f:
            f.write(modified_content)
        
        logger.info("Successfully updated get_full_dxy_data function to convert Decimal values")
        return True
    
    except Exception as e:
        logger.error(f"Error updating get_dxy_data function: {e}")
        logger.error(traceback.format_exc())
        return False

def fix_get_liquidity_data():
    """Update get_liquidity_data function to convert Decimal values to float"""
    try:
        filepath = 'economic_indicators.py'
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Find the get_liquidity_data function
        start_idx = content.find("def get_liquidity_data(")
        if start_idx == -1:
            logger.error("Could not find get_liquidity_data function in economic_indicators.py")
            return False
        
        # Find the end of the function (next def statement)
        next_def_idx = content.find("def ", start_idx + 10)
        if next_def_idx == -1:
            next_def_idx = len(content)
        
        # Find the return statement
        return_idx = content.rfind("return df", start_idx, next_def_idx)
        if return_idx == -1:
            logger.error("Could not find return statement in get_liquidity_data function")
            return False
        
        # Check if we already added the conversion
        if "ensure_float_values" in content[start_idx:next_def_idx]:
            logger.info("Decimal conversion already added to get_liquidity_data function")
            return True
        
        # Add conversion before the return statement
        modified_content = (
            content[:return_idx] + 
            "    # Convert any decimal.Decimal columns to float\n" +
            "    df = ensure_float_values(df)\n    \n    " + 
            content[return_idx:]
        )
        
        # Write the modified content back
        with open(filepath, 'w') as f:
            f.write(modified_content)
        
        logger.info("Successfully updated get_liquidity_data function to convert Decimal values")
        return True
    
    except Exception as e:
        logger.error(f"Error updating get_liquidity_data function: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all fixes"""
    print("Starting fixes for decimal.Decimal issues...")
    
    # Fix economic_indicators.py functions
    print("1. Fixing get_dxy_data function...")
    result = fix_get_dxy_data()
    print(f"   Result: {'Success' if result else 'Failed'}")
    
    print("2. Fixing get_liquidity_data function...")
    result = fix_get_liquidity_data()
    print(f"   Result: {'Success' if result else 'Failed'}")
    
    print("Decimal.Decimal fixes completed.")

if __name__ == "__main__":
    main()