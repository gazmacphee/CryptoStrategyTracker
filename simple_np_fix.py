"""
Simple fix for the 'schema "np" does not exist' error in trading signals.
This script finds and replaces direct references to np.float64 with float()
"""

import os
import sys
import re

def fix_np_schema_error():
    """Find and fix np.float64 in trading_signals.py"""
    print("Looking for np.float64 references in trading_signals.py...")
    
    # Get the file path
    file_path = "trading_signals.py"
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return False
    
    # Read the file content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Look for np.float64 pattern
    np_float_pattern = r'np\.float\d+'
    matches = re.findall(np_float_pattern, content)
    
    if not matches:
        print("No np.float references found.")
        return False
    
    print(f"Found {len(matches)} np.float references: {matches}")
    
    # Replace np.float64 with float
    modified_content = re.sub(np_float_pattern, 'float', content)
    
    # Write the modified content back
    with open(file_path, 'w') as file:
        file.write(modified_content)
    
    print(f"Successfully replaced {len(matches)} np.float references with float")
    return True

def calculate_exit_levels_fix():
    """Add explicit float conversion in calculate_exit_levels"""
    print("Adding explicit float conversion in calculate_exit_levels...")
    
    # Get the file path
    file_path = "trading_signals.py"
    
    # Read the file content
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # We know from grep that calculate_exit_levels is at line 188
    start_line = 187  # 0-indexed
    
    # Find the end of the function and the return statement
    end_line = None
    return_line = None
    
    for i in range(start_line, len(lines)):
        if "def " in lines[i] and i > start_line:
            end_line = i
            break
        elif "return exit_levels" in lines[i]:
            return_line = i
    
    if return_line is None:
        print("Could not find return statement in calculate_exit_levels function")
        return False
    
    # Set end_line to the end of the file if not found
    if end_line is None:
        end_line = len(lines)
    
    # Add float conversion before the return statement
    float_conversion = """    # Convert all numeric values to standard Python float to prevent np.float64 errors
    for key in ['stop_loss', 'take_profit', 'risk_reward_ratio']:
        if key in exit_levels and exit_levels[key] is not None:
            exit_levels[key] = float(exit_levels[key])
    
"""
    
    # Check if this fix has already been applied
    already_fixed = False
    for i in range(max(0, return_line - 5), return_line):
        if "Convert all numeric values" in lines[i]:
            already_fixed = True
            break
    
    if not already_fixed:
        # Insert before the return statement
        lines.insert(return_line, float_conversion)
        
        # Write the modified content back
        with open(file_path, 'w') as file:
            file.writelines(lines)
        
        print("Successfully added float conversion to calculate_exit_levels")
    else:
        print("Float conversion already exists in calculate_exit_levels")
    
    return True

if __name__ == "__main__":
    print("Starting fix for np.float64 references...")
    
    # Fix direct np.float64 references
    fix_result = fix_np_schema_error()
    
    # Add explicit float conversion
    exit_levels_fix = calculate_exit_levels_fix()
    
    if fix_result or exit_levels_fix:
        print("Successfully applied fixes for np.float64 issues")
    else:
        print("No changes were needed or issues were not found")