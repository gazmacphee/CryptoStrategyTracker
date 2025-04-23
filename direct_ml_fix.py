"""
Direct fix for ML training error handling.
"""

import sys
import os

# Get the location of the main function and error line
error_line = None
error_line_content = None
error_file = "advanced_ml.py"

with open(error_file, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "train_results['successful']" in line and "train_results['total']" in line:
            error_line = i
            error_line_content = line
            break

if error_line is None:
    print(f"Error line not found in {error_file}")
    sys.exit(1)

print(f"Found error line at line {error_line + 1}: {error_line_content.strip()}")

# Modify the error line to use .get() method with defaults
new_lines = []
for i, line in enumerate(lines):
    if i == error_line:
        # Replace with safe dictionary access
        indentation = len(line) - len(line.lstrip())
        spaces = ' ' * indentation
        new_lines.append(f"{spaces}total = train_results.get('total', 0)\n")
        new_lines.append(f"{spaces}successful = train_results.get('successful', 0)\n")
        new_lines.append(f"{spaces}print(f\"Training completed: {{successful}}/{{total}} models trained\")\n")
    else:
        new_lines.append(line)

# Write the modified file
with open(error_file, 'w') as f:
    f.writelines(new_lines)

print(f"Successfully fixed error line in {error_file}")