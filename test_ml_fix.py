#!/usr/bin/env python
"""
Test script to verify the ML pattern model training fix
"""

from advanced_ml import train_all_pattern_models

if __name__ == "__main__":
    print("Testing ML pattern model training...")
    train_results = train_all_pattern_models()
    print(f"Training completed: {train_results['successful']}/{train_results['total']} models trained")
    print(f"Full results: {train_results}")