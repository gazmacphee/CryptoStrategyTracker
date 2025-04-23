# Machine Learning Setup Guide

This guide will help you set up and use the advanced machine learning features of the Cryptocurrency Trading Analysis Platform.

## Overview

The platform includes advanced ML capabilities for:

1. **Pattern Recognition**: Automatically detect chart patterns like support/resistance bounces, momentum shifts, and double bottoms
2. **Trading Recommendations**: Generate actionable trading signals with confidence scores
3. **Multi-Timeframe Analysis**: Analyze patterns across different symbols and intervals

## First-Time Setup

Before using the ML features, ensure you have:

1. Set up the PostgreSQL database (see main README.md)
2. Run the backfill process to populate historical data
3. Created the machine learning models directory (if not exists)

```bash
# Create models directory if it doesn't exist
mkdir -p models/pattern_recognition
```

## Using the ML Features

### Step 1: Ensure you have sufficient data

The ML models require sufficient historical data to train. Check if you have enough data:

```bash
python check_ml_data.py --symbol BTCUSDT --interval 30m
```

### Step 2: Train the pattern recognition models

```bash
# Train models for all symbols with sufficient data
python train_all_pattern_models.py

# Or, to wait for data to become available:
python train_all_pattern_models.py --wait --max-wait 30
```

### Step 3: Analyze patterns and save recommendations

```bash
# Analyze patterns across all markets
python analyze_patterns_and_save.py
```

### Step 4: Set up automated ML processes

```bash
# Update process manager configuration to include ML tasks
python update_ml_process_config.py
```

## Running Tests

To verify all ML components are working correctly:

```bash
# Run all ML tests
python run_ml_tests.py --all

# Or run specific tests
python run_ml_tests.py --integration
python run_ml_tests.py --train
python run_ml_tests.py --analyze
```

## Advanced Usage

### Adjusting Pattern Recognition Thresholds

Open `advanced_ml.py` and adjust:

- `MIN_PATTERN_STRENGTH`: Minimum strength for a pattern to be recognized (default: 0.7)
- `MIN_PATTERN_INSTANCES`: Minimum number of patterns required for training (default: 10)
- `PRICE_MOVEMENT_THRESHOLD`: Minimum price movement to be considered significant (default: 0.02)

### Adding New Pattern Types

To add new pattern types, modify the `extract_labeled_patterns` method in the `PatternRecognitionModel` class in `advanced_ml.py`.

### Understanding Pattern Types

The system currently recognizes these pattern types:

1. **Support Bounce**: Price bounces off a support level
2. **Resistance Breakdown**: Price breaks down through a resistance level
3. **Momentum Shift Bullish**: Change from downtrend to uptrend
4. **Momentum Shift Bearish**: Change from uptrend to downtrend
5. **Double Bottom**: W-shaped reversal pattern

### Checking Model Performance

After training, model performance metrics are stored in the database. You can query these metrics with:

```sql
SELECT * FROM ml_model_performance ORDER BY training_date DESC LIMIT 10;
```

## Troubleshooting

### Error: Maximum Recursion Depth Exceeded

This error indicates a circular import between modules. Run:

```bash
python direct_ml_fix.py
```

### Error: No Data Retrieved

This means there's insufficient historical data in the database. Ensure the backfill process is running:

```bash
python backfill_database.py
```

### Error: Model Training Failed

Check that you have scikit-learn and related dependencies installed:

```bash
pip install scikit-learn joblib
```

Also ensure your data has enough records for training (minimum 500 recommended).

## Further Development

- Add more pattern types
- Implement deep learning models for price prediction
- Add correlation analysis with economic indicators
- Develop reinforcement learning for adaptive trading strategies