"""
Fix numpy dtype issues in trading signals.

This script adds type conversion to ensure proper database compatibility.
"""

import numpy as np
import logging
import traceback
import os
import re
from decimal import Decimal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix_numpy_dtype_errors")

def safe_float_convert(value):
    """Convert any numeric type to float safely"""
    if value is None:
        return None
    if hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.number):
        return float(value)
    if isinstance(value, Decimal):
        return float(value)
    return value

def fix_calculate_exit_levels():
    """Update calculate_exit_levels function to ensure float conversions"""
    logger.info("Fixing calculate_exit_levels function in trading_signals.py")
    
    try:
        # Path to trading_signals.py
        file_path = "trading_signals.py"
        
        # Read the file
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Find the calculate_exit_levels function
        pattern = r'def calculate_exit_levels\([^)]*\):.*?return exit_levels'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            logger.error("Could not find calculate_exit_levels function")
            return False
        
        original_function = match.group(0)
        
        # Replace with version that converts numpy types to Python float
        modified_function = original_function.replace(
            'return exit_levels',
            """
    # Convert all numeric values to standard Python float
    for key in ['stop_loss', 'take_profit', 'risk_reward_ratio']:
        if key in exit_levels and exit_levels[key] is not None:
            if hasattr(exit_levels[key], 'dtype') and np.issubdtype(exit_levels[key].dtype, np.number):
                exit_levels[key] = float(exit_levels[key])
            elif isinstance(exit_levels[key], Decimal):
                exit_levels[key] = float(exit_levels[key])
    
    return exit_levels"""
        )
        
        # Add import for Decimal if needed
        if 'from decimal import Decimal' not in content:
            content = 'from decimal import Decimal\n' + content
        
        # Add import for numpy if needed
        if 'import numpy as np' not in content:
            content = 'import numpy as np\n' + content
        
        # Replace the function in the content
        updated_content = content.replace(original_function, modified_function)
        
        # Write the updated content back to the file
        with open(file_path, 'w') as file:
            file.write(updated_content)
        
        logger.info("Successfully updated calculate_exit_levels function")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing calculate_exit_levels: {e}")
        logger.error(traceback.format_exc())
        return False

def add_safe_float_conversions():
    """Add safe float conversions for row values in save_trading_signals"""
    logger.info("Adding safe float conversions to save_trading_signals function")
    
    try:
        # Path to trading_signals.py
        file_path = "trading_signals.py"
        
        # Read the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Find the location where to add safe float conversions
        in_save_function = False
        in_for_loop = False
        found_close_conversion = False
        insert_line = None
        
        for i, line in enumerate(lines):
            if "def save_trading_signals(" in line:
                in_save_function = True
            elif in_save_function and "def " in line:
                in_save_function = False
            
            if in_save_function and "for idx, row in df.iterrows():" in line:
                in_for_loop = True
            elif in_save_function and in_for_loop and "}" in line:
                in_for_loop = False
            
            if in_save_function and in_for_loop and "float(row['close'])" in line:
                found_close_conversion = True
                
            # Look for place right after we get entry_price but before exit levels calc
            if in_save_function and in_for_loop and "entry_price = float(row['close'])" in line:
                insert_line = i + 1
        
        if not found_close_conversion:
            logger.warning("Did not find float conversion for close price")
        
        if insert_line:
            # Add safe conversion function to the beginning of the file if it's not there
            has_safe_convert = False
            for line in lines:
                if "def safe_float_convert" in line:
                    has_safe_convert = True
                    break
            
            if not has_safe_convert:
                # Find the imports section to add the function
                imports_end = 0
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith("import ") and not line.startswith("from "):
                        imports_end = i
                        break
                
                safe_convert_function = """
def safe_float_convert(value):
    """Convert any numeric type to float safely"""
    if value is None:
        return None
    if hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.number):
        return float(value)
    if isinstance(value, Decimal):
        return float(value)
    return value

"""
                lines.insert(imports_end, safe_convert_function)
                insert_line += len(safe_convert_function.split('\n'))
            
            # Add conversion before exit_levels calculation
            conversion_code = """                    # Ensure all numeric values are Python floats, not numpy types
                    for col in recent_rows.select_dtypes(include=[np.number]).columns:
                        recent_rows[col] = recent_rows[col].apply(lambda x: float(x) if x is not None else None)
                    
"""
            lines.insert(insert_line, conversion_code)
            
            # Write updated content back to file
            with open(file_path, 'w') as file:
                file.writelines(lines)
            
            logger.info("Successfully added safe float conversions")
            return True
        else:
            logger.warning("Could not find insertion point for safe float conversions")
            return False
    
    except Exception as e:
        logger.error(f"Error adding safe float conversions: {e}")
        logger.error(traceback.format_exc())
        return False

def fix_np_schema_error():
    """Fix the schema 'np' not exists error in trading_signals.py"""
    logger.info("Fixing 'schema np does not exist' error in trading_signals.py")
    
    try:
        # Try to identify where we need to force float conversions
        patterns_to_search = [
            r"np\.float64", 
            r"np\.float32",
            r"np\."
        ]
        
        results = []
        for pattern in patterns_to_search:
            cmd = f"grep -n '{pattern}' trading_signals.py"
            output = os.popen(cmd).read().strip()
            if output:
                results.append(output)
        
        if not results:
            logger.warning("Could not find direct references to np types")
            
            # Try a more general fix by ensuring all calculations convert to float explicitly
            fix_calculate_exit_levels()
            add_safe_float_conversions()
            logger.info("Applied general fixes to prevent numpy dtype issues")
            return True
        else:
            logger.info(f"Found potential numpy type issues: {results}")
            
            # Fix each instance
            for result in results:
                # Extract line number and content
                line_parts = result.split(':', 1)
                if len(line_parts) != 2:
                    continue
                
                line_num = int(line_parts[0])
                line_content = line_parts[1]
                
                logger.info(f"Fixing line {line_num}: {line_content}")
                
                # Read the file
                with open("trading_signals.py", 'r') as file:
                    lines = file.readlines()
                
                # Replace np.float64/np.float32 with float()
                if line_num <= len(lines):
                    original_line = lines[line_num - 1]
                    modified_line = re.sub(r'np\.float\d*', 'float', original_line)
                    lines[line_num - 1] = modified_line
                
                # Write the file back
                with open("trading_signals.py", 'w') as file:
                    file.writelines(lines)
            
            # Apply general fixes too
            fix_calculate_exit_levels()
            add_safe_float_conversions()
            
            logger.info("Successfully fixed numpy dtype issues")
            return True
    
    except Exception as e:
        logger.error(f"Error fixing schema 'np' error: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("Starting fix for numpy dtype errors...")
    result = fix_np_schema_error()
    print(f"Fix completed: {'Success' if result else 'Failed'}")