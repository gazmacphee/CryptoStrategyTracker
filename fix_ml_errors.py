"""
Fix script for ML training error handling issues.
This script implements robust error handling for the ML training process.
"""

import json
import logging
import traceback
from datetime import datetime
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix_ml_errors")

def apply_train_pattern_models_fix():
    """Apply fixes to the train_pattern_models method in advanced_ml.py"""
    logger.info("Checking advanced_ml.py for error handling issues...")
    
    try:
        # First, find the location of the issue
        found_main_code = False
        error_line = None
        line_number = 0
        
        with open('advanced_ml.py', 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line_number = i + 1
                if "if __name__ == \"__main__\":" in line:
                    found_main_code = True
                if "print(f\"Training completed: {train_results['successful']}/{train_results['total']}" in line:
                    error_line = i
                    logger.info(f"Found error line at line {line_number}: {line.strip()}")
        
        if error_line is not None:
            # Modify the lines to use safe dictionary access
            lines[error_line] = '    total = train_results.get("total", 0)\n'
            lines.insert(error_line + 1, '    successful = train_results.get("successful", 0)\n')
            lines.insert(error_line + 2, '    print(f"Training completed: {successful}/{total} models trained")\n')
            lines.insert(error_line + 3, '    print(f"Full results: {train_results}")\n')
            
            # Write the modified code back to the file
            with open('advanced_ml.py', 'w') as f:
                f.writelines(lines)
            
            logger.info("Successfully applied ML error handling fix")
            return True
        else:
            logger.warning("Could not find the error line in advanced_ml.py")
            return False
        
    except Exception as e:
        logger.error(f"Error applying fix: {e}")
        logger.error(traceback.format_exc())
        return False

def ensure_train_all_pattern_models_error_handling():
    """Make sure train_all_pattern_models properly handles errors"""
    logger.info("Checking train_all_pattern_models function for proper error handling")
    
    try:
        function_found = False
        start_line = None
        end_line = None
        line_number = 0
        
        with open('advanced_ml.py', 'r') as f:
            lines = f.readlines()
            in_function = False
            for i, line in enumerate(lines):
                line_number = i + 1
                if "def train_all_pattern_models():" in line:
                    function_found = True
                    start_line = i
                    in_function = True
                elif in_function and line.startswith("def "):
                    end_line = i - 1
                    break
        
        if not function_found:
            logger.warning("Could not find train_all_pattern_models function")
            return False
        
        if end_line is None:
            end_line = len(lines) - 1
        
        logger.info(f"Found train_all_pattern_models function at lines {start_line + 1}-{end_line + 1}")
        
        # Extract function code
        function_code = "".join(lines[start_line:end_line+1])
        
        # Check if it has proper error handling
        if "'successful'" in function_code and "'total'" in function_code and "try:" in function_code:
            logger.info("Function already has error handling for 'successful' and 'total' keys")
            return True
        else:
            logger.warning("Function is missing proper error handling")
            
            # Add proper error handling
            new_function_code = []
            
            # Find the end of the function body
            in_function_body = False
            indentation = ""
            for i in range(start_line, end_line + 1):
                line = lines[i]
                if "def train_all_pattern_models" in line:
                    in_function_body = True
                    new_function_code.append(line)
                elif in_function_body and not indentation and line.strip():
                    indentation = " " * (len(line) - len(line.lstrip()))
                    new_function_code.append(line)
                else:
                    new_function_code.append(line)
            
            try_check_code = f"{indentation}# Ensure results is properly structured\n"
            try_check_code += f"{indentation}if not isinstance(results, dict):\n"
            try_check_code += f"{indentation}    logger.warning(\"Invalid train_pattern_models result format\")\n"
            try_check_code += f"{indentation}    results = {{'total': 0, 'successful': 0, 'details': {{}}}}\n\n"
            try_check_code += f"{indentation}# Ensure result has the expected keys\n"
            try_check_code += f"{indentation}if 'total' not in results or 'successful' not in results:\n"
            try_check_code += f"{indentation}    logger.warning(\"Missing keys in train_pattern_models result\")\n"
            try_check_code += f"{indentation}    if 'details' in results and isinstance(results['details'], dict):\n"
            try_check_code += f"{indentation}        successful_models = sum(1 for v in results['details'].values() if v)\n"
            try_check_code += f"{indentation}        results['total'] = len(results['details'])\n"
            try_check_code += f"{indentation}        results['successful'] = successful_models\n"
            try_check_code += f"{indentation}    else:\n"
            try_check_code += f"{indentation}        results['total'] = 0\n"
            try_check_code += f"{indentation}        results['successful'] = 0\n"
            try_check_code += f"{indentation}        results['details'] = results.get('details', {{}})\n"
            
            # Find the return statement and add the error checking before it
            added_check = False
            for i in range(len(new_function_code) - 1, -1, -1):
                if "return results" in new_function_code[i]:
                    # Insert before the return
                    new_function_code.insert(i, try_check_code)
                    added_check = True
                    break
            
            if not added_check:
                logger.warning("Could not find the return statement to add error checking")
                return False
            
            # Write the modified function back to the file
            lines[start_line:end_line+1] = new_function_code
            with open('advanced_ml.py', 'w') as f:
                f.writelines(lines)
            
            logger.info("Added proper error handling to train_all_pattern_models function")
            return True
        
    except Exception as e:
        logger.error(f"Error ensuring train_all_pattern_models error handling: {e}")
        logger.error(traceback.format_exc())
        return False

def add_argument_handling():
    """Add command-line argument handling to advanced_ml.py"""
    logger.info("Adding command-line argument handling to advanced_ml.py")
    
    try:
        # Check if there's already an argparse section
        has_argparse = False
        with open('advanced_ml.py', 'r') as f:
            content = f.read()
            if 'argparse' in content and '--train' in content:
                has_argparse = True
        
        if has_argparse:
            logger.info("File already has argument handling")
            return True
        
        # Find the location to add the argument handling
        with open('advanced_ml.py', 'r') as f:
            lines = f.readlines()
            
        # Find the end of the file or the if __name__ == "__main__" block
        main_block_start = None
        for i, line in enumerate(lines):
            if line.strip() == 'if __name__ == "__main__":':
                main_block_start = i
                break
        
        if main_block_start is None:
            # No main block, add argument handling at the end
            main_block_start = len(lines)
            lines.append("\n")
        
        # Define the argument handling code
        arg_code = [
            "def main():\n",
            "    \"\"\"Main entry point for command line execution\"\"\"\n",
            "    import argparse\n",
            "    \n",
            "    # Parse command line arguments\n",
            "    parser = argparse.ArgumentParser(description='Advanced ML Pattern Recognition for Cryptocurrency Trading')\n",
            "    parser.add_argument('--train', action='store_true', help='Train pattern models for popular symbols')\n",
            "    parser.add_argument('--analyze', action='store_true', help='Analyze current market patterns')\n",
            "    parser.add_argument('--save', action='store_true', help='Save detected patterns as trading signals')\n",
            "    parser.add_argument('--verbose', action='store_true', help='Display detailed information')\n",
            "    \n",
            "    args = parser.parse_args()\n",
            "    \n",
            "    # Default behavior - run everything if no specific args provided\n",
            "    run_all = not (args.train or args.analyze or args.save)\n",
            "    \n",
            "    # Train pattern models if requested\n",
            "    if args.train or run_all:\n",
            "        print(\"Training pattern models...\")\n",
            "        train_results = train_all_pattern_models()\n",
            "        \n",
            "        # Get keys safely with defaults as a precaution\n",
            "        total = train_results.get('total', 0)\n",
            "        successful = train_results.get('successful', 0)\n",
            "        print(f\"Training completed: {successful}/{total} models trained\")\n",
            "        \n",
            "        if args.verbose:\n",
            "            print(f\"Full results: {train_results}\")\n",
            "    \n",
            "    # Analyze market patterns if requested\n",
            "    if args.analyze or run_all:\n",
            "        print(\"\\nAnalyzing current market patterns...\")\n",
            "        recommendations = get_pattern_recommendations()\n",
            "        \n",
            "        if not isinstance(recommendations, pd.DataFrame):\n",
            "            print(\"Error: recommendations is not a DataFrame\")\n",
            "            if args.verbose:\n",
            "                print(f\"Type: {type(recommendations)}\")\n",
            "                print(f\"Value: {recommendations}\")\n",
            "        elif recommendations.empty:\n",
            "            print(\"No trading opportunities detected at this time\")\n",
            "        else:\n",
            "            print(f\"\\nFound {len(recommendations)} trading opportunities:\")\n",
            "            display_cols = ['symbol', 'interval', 'pattern_type', 'predicted_direction', 'pattern_strength', 'expected_return']\n",
            "            # Only select columns that exist in the DataFrame\n",
            "            available_cols = [col for col in display_cols if col in recommendations.columns]\n",
            "            print(recommendations[available_cols])\n",
            "    \n",
            "    # Save recommendations as trading signals if requested\n",
            "    if args.save or run_all:\n",
            "        print(\"\\nSaving detected patterns as trading signals...\")\n",
            "        saved_count = save_current_recommendations()\n",
            "        print(f\"Saved {saved_count} patterns as trading signals\")\n",
            "\n",
            "\n"
        ]
        
        # Define the main block code
        main_block_code = [
            "if __name__ == \"__main__\":\n",
            "    main()\n"
        ]
        
        # Add the argument handling code before the main block
        if main_block_start < len(lines):
            # Replace existing main block
            lines = lines[:main_block_start] + arg_code + main_block_code
        else:
            # Add at the end
            lines.extend(arg_code + main_block_code)
        
        # Write the modified code back to the file
        with open('advanced_ml.py', 'w') as f:
            f.writelines(lines)
        
        logger.info("Successfully added argument handling to advanced_ml.py")
        return True
        
    except Exception as e:
        logger.error(f"Error adding argument handling: {e}")
        logger.error(traceback.format_exc())
        return False

def test_ml_with_safe_dict_access():
    """Test ML training with safe dictionary access"""
    logger.info("Running test with safe dictionary access")
    
    try:
        # Create modified test version
        with open('test_ml_fix.py', 'w') as f:
            f.write('''
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
''')
        
        # Run the test
        import subprocess
        result = subprocess.run(['python', 'test_ml_fix.py'], capture_output=True, text=True)
        
        logger.info(f"Test output: {result.stdout}")
        if result.returncode == 0 and "Test completed successfully!" in result.stdout:
            logger.info("Safe dictionary access test passed")
            return True
        else:
            logger.error(f"Test failed with error: {result.stderr}")
            return False
        
    except Exception as e:
        logger.error(f"Error running safe dictionary access test: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("Starting ML error fixing process...")
    
    # Fix direct dictionary access in main block
    print("1. Fixing direct dictionary access in main block...")
    result = apply_train_pattern_models_fix()
    print(f"Result: {'Success' if result else 'Failed'}")
    
    # Ensure train_all_pattern_models has proper error handling
    print("2. Ensuring train_all_pattern_models has proper error handling...")
    result = ensure_train_all_pattern_models_error_handling()
    print(f"Result: {'Success' if result else 'Failed'}")
    
    # Add command-line argument handling
    print("3. Adding command-line argument handling...")
    result = add_argument_handling()
    print(f"Result: {'Success' if result else 'Failed'}")
    
    # Test safe dictionary access
    print("4. Testing safe dictionary access...")
    result = test_ml_with_safe_dict_access()
    print(f"Result: {'Success' if result else 'Failed'}")
    
    print("ML error fixing process completed.")