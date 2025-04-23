"""
Fix for decimal.Decimal and float type incompatibility issues.

This script adds a decorator to safely convert Decimal values to float
for arithmetic operations.
"""

import os
import sys
import traceback
import logging
from decimal import Decimal
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("decimal_float_fix")

def fix_economic_ui():
    """Fix decimal issues in economic_ui.py"""
    logger.info("Fixing decimal issues in economic_ui.py")
    
    try:
        economic_ui_path = 'economic_ui.py'
        if not os.path.exists(economic_ui_path):
            logger.error(f"File not found: {economic_ui_path}")
            return False
        
        with open(economic_ui_path, 'r') as f:
            content = f.read()
        
        # Add a helper function to convert Decimal values to float
        if "def safe_float_convert" not in content:
            # Find a good spot to add the function
            import_section_end = content.find("import plotly.subplots") + len("import plotly.subplots")
            next_newline = content.find("\n", import_section_end)
            
            helper_function = """

def safe_float_convert(value):
    """Convert Decimal values to float safely"""
    if isinstance(value, Decimal):
        return float(value)
    return value

def ensure_float_series(series):
    """Ensure a pandas Series contains only float values, not Decimal"""
    if series.empty:
        return series
    return series.apply(safe_float_convert)

"""
            
            # Insert the helper function after imports
            modified_content = content[:next_newline+1] + helper_function + content[next_newline+1:]
            
            # Replace occurrences of direct operations on potentially Decimal columns
            # For mean calculations
            modified_content = modified_content.replace(
                "dxy_df['close'].mean()",
                "ensure_float_series(dxy_df['close']).mean()"
            )
            
            # For other operations
            modified_content = modified_content.replace(
                "dxy_df['close'].rolling(window=20).mean()",
                "ensure_float_series(dxy_df['close']).rolling(window=20).mean()"
            )
            
            modified_content = modified_content.replace(
                "wm2_df['value'].pct_change()",
                "ensure_float_series(wm2_df['value']).pct_change()"
            )
            
            modified_content = modified_content.replace(
                "fed_df['value'].pct_change()", 
                "ensure_float_series(fed_df['value']).pct_change()"
            )
            
            # Write the modified content back
            with open(economic_ui_path, 'w') as f:
                f.write(modified_content)
                
            logger.info("Successfully fixed decimal issues in economic_ui.py")
            return True
        else:
            logger.info("Helper function already exists in economic_ui.py")
            return True
            
    except Exception as e:
        logger.error(f"Error fixing economic_ui.py: {e}")
        logger.error(traceback.format_exc())
        return False

def fix_indicators():
    """Fix decimal issues in indicators.py"""
    logger.info("Fixing decimal issues in indicators.py")
    
    try:
        indicators_path = 'indicators.py'
        if not os.path.exists(indicators_path):
            logger.error(f"File not found: {indicators_path}")
            return False
        
        with open(indicators_path, 'r') as f:
            content = f.read()
        
        # Add a helper function to convert Decimal values to float
        if "def safe_float_convert" not in content:
            # Find a good spot to add the function
            import_section_end = content.find("import pandas as pd") + len("import pandas as pd")
            next_newline = content.find("\n", import_section_end)
            
            helper_function = """
import numpy as np
from decimal import Decimal

def safe_float_convert(value):
    """Convert Decimal values to float safely"""
    if isinstance(value, Decimal):
        return float(value)
    return value

def ensure_float_series(series):
    """Ensure a pandas Series contains only float values, not Decimal"""
    if series.empty:
        return series
    return series.apply(safe_float_convert)

def ensure_float_df(df, columns=None):
    """Ensure specified columns in DataFrame contain only float values, not Decimal"""
    if df.empty:
        return df
        
    if columns is None:
        columns = ['open', 'high', 'low', 'close', 'volume']
        columns = [col for col in columns if col in df.columns]
    
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(safe_float_convert)
    
    return df

"""
            
            # Insert the helper function after imports
            modified_content = content[:next_newline+1] + helper_function + content[next_newline+1:]
            
            # Modify the add_bollinger_bands function to convert decimal values
            modified_content = modified_content.replace(
                "def add_bollinger_bands(df, window=20, num_std=2):",
                "def add_bollinger_bands(df, window=20, num_std=2):\n    # Ensure we have float values, not Decimal\n    df = ensure_float_df(df)"
            )
            
            # Modify other indicator functions similarly
            modified_content = modified_content.replace(
                "def add_rsi(df, window=14):",
                "def add_rsi(df, window=14):\n    # Ensure we have float values, not Decimal\n    df = ensure_float_df(df)"
            )
            
            modified_content = modified_content.replace(
                "def add_macd(df, fast=12, slow=26, signal=9):",
                "def add_macd(df, fast=12, slow=26, signal=9):\n    # Ensure we have float values, not Decimal\n    df = ensure_float_df(df)"
            )
            
            modified_content = modified_content.replace(
                "def add_stoch_rsi(df, k_window=3, d_window=3, window=14):",
                "def add_stoch_rsi(df, k_window=3, d_window=3, window=14):\n    # Ensure we have float values, not Decimal\n    df = ensure_float_df(df)"
            )
            
            # Write the modified content back
            with open(indicators_path, 'w') as f:
                f.write(modified_content)
                
            logger.info("Successfully fixed decimal issues in indicators.py")
            return True
        else:
            logger.info("Helper function already exists in indicators.py")
            return True
            
    except Exception as e:
        logger.error(f"Error fixing indicators.py: {e}")
        logger.error(traceback.format_exc())
        return False

def fix_app_signals_display():
    """Fix decimal issues in app.py signals display"""
    logger.info("Fixing decimal issues in app.py signals display")
    
    try:
        app_path = 'app.py'
        if not os.path.exists(app_path):
            logger.error(f"File not found: {app_path}")
            return False
        
        with open(app_path, 'r') as f:
            lines = f.readlines()
        
        # Find and fix the problematic line with '* 0.995'
        fixed_lines = []
        found_issue = False
        
        for line in lines:
            if "buy_signals['low'] * 0.995" in line:
                # Replace with safe conversion
                indent = len(line) - len(line.lstrip())
                fixed_line = ' ' * indent + "y=buy_signals['low'].apply(lambda x: float(x) * 0.995 if isinstance(x, Decimal) else x * 0.995),  # Place just below the candle\n"
                fixed_lines.append(fixed_line)
                found_issue = True
            else:
                fixed_lines.append(line)
        
        # Add import for Decimal if not present
        if found_issue:
            has_decimal_import = any("from decimal import Decimal" in line for line in lines)
            if not has_decimal_import:
                # Find the imports section
                for i, line in enumerate(fixed_lines):
                    if "import " in line:
                        fixed_lines.insert(i, "from decimal import Decimal\n")
                        break
        
        # Write the modified content back
        if found_issue:
            with open(app_path, 'w') as f:
                f.writelines(fixed_lines)
            
            logger.info("Successfully fixed decimal issues in app.py signals display")
            return True
        else:
            logger.info("No decimal issues found in app.py signals display")
            return True
            
    except Exception as e:
        logger.error(f"Error fixing app.py signals display: {e}")
        logger.error(traceback.format_exc())
        return False

def fix_database_decimal_handling():
    """Add decimal handling to database.py when fetching data"""
    logger.info("Adding decimal handling to database.py")
    
    try:
        database_path = 'database.py'
        if not os.path.exists(database_path):
            logger.error(f"File not found: {database_path}")
            return False
        
        with open(database_path, 'r') as f:
            content = f.read()
        
        # Add a helper function to convert Decimal values to float
        if "def convert_decimal_to_float" not in content:
            # Find a good spot to add the function
            import_section_end = content.find("import pandas as pd") + len("import pandas as pd")
            next_newline = content.find("\n", import_section_end)
            
            helper_function = """
from decimal import Decimal

def convert_decimal_to_float(df):
    """Convert all Decimal values in a DataFrame to float"""
    for col in df.columns:
        if df[col].dtype == object:  # Check for object dtype which might contain Decimal
            # Check if first non-null value is Decimal
            first_value = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(first_value, Decimal):
                df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
    return df

"""
            
            # Insert the helper function after imports
            modified_content = content[:next_newline+1] + helper_function + content[next_newline+1:]
            
            # Modify the get_ohlcv_data and similar functions to convert decimal values
            # Find the get_ohlcv_data function
            get_ohlcv_data_idx = modified_content.find("def get_ohlcv_data(")
            if get_ohlcv_data_idx != -1:
                # Find the return statement in this function
                return_idx = modified_content.find("return df", get_ohlcv_data_idx)
                if return_idx != -1:
                    # Replace the return statement to include conversion
                    modified_content = modified_content[:return_idx] + "    # Convert any Decimal values to float\n    df = convert_decimal_to_float(df)\n    " + modified_content[return_idx:]
            
            # Similarly for other data retrieval functions
            for func_name in ["get_crypto_data", "get_economic_indicator", "get_trading_signals"]:
                func_idx = modified_content.find(f"def {func_name}(")
                if func_idx != -1:
                    # Find the return statement in this function
                    return_idx = modified_content.find("return df", func_idx)
                    if return_idx != -1:
                        # Replace the return statement to include conversion
                        modified_content = modified_content[:return_idx] + "    # Convert any Decimal values to float\n    df = convert_decimal_to_float(df)\n    " + modified_content[return_idx:]
            
            # Write the modified content back
            with open(database_path, 'w') as f:
                f.write(modified_content)
                
            logger.info("Successfully added decimal handling to database.py")
            return True
        else:
            logger.info("Decimal handling function already exists in database.py")
            return True
            
    except Exception as e:
        logger.error(f"Error fixing database.py: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("Starting decimal-float compatibility fixes...")
    
    # Fix economic_ui.py
    print("1. Fixing decimal issues in economic_ui.py...")
    result = fix_economic_ui()
    print(f"Result: {'Success' if result else 'Failed'}")
    
    # Fix indicators.py
    print("2. Fixing decimal issues in indicators.py...")
    result = fix_indicators()
    print(f"Result: {'Success' if result else 'Failed'}")
    
    # Fix app.py signals display
    print("3. Fixing decimal issues in app.py signals display...")
    result = fix_app_signals_display()
    print(f"Result: {'Success' if result else 'Failed'}")
    
    # Fix database.py
    print("4. Adding decimal handling to database.py...")
    result = fix_database_decimal_handling()
    print(f"Result: {'Success' if result else 'Failed'}")
    
    print("Decimal-float compatibility fixes completed.")