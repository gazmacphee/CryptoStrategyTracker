# Cryptocurrency Trading Platform Command Reference

This document provides quick reference commands for running, maintaining, and troubleshooting the Cryptocurrency Trading Analysis Platform.

## Basic Operations

### Starting the Application
```bash
# Start the application with database reset and backfill
python reset_and_start.py

# Start only the Streamlit interface (without database reset)
streamlit run app.py --server.port=5000 --server.address=0.0.0.0
```

### Database Operations
```bash
# Reset the database (drop and recreate all tables)
python -c "from database import get_db_connection; from reset_and_start import reset_database; reset_database()"

# Create fresh database tables
python -c "from database import create_tables; create_tables()"

# Check database statistics
python -c "from app import get_database_stats; print(get_database_stats())"
```

### Data Management
```bash
# Start a background backfill process
python -c "from start_improved_backfill import start_background_backfill; start_background_backfill()"

# Run a quick backfill (one-time, not continuous)
python -c "from start_improved_backfill import start_background_backfill; start_background_backfill(continuous=False)"

# Run a full backfill with more symbols and longer history
python -c "from app import run_background_backfill; run_background_backfill(full=True)"
```

## Advanced Operations

### Monitoring and Troubleshooting
```bash
# Check if backfill process is running
ls -la .backfill_lock

# View backfill progress
cat backfill_progress.json

# Check latest data updates
python -c "from app import get_last_update_time; print(get_last_update_time())"

# View log files (after moving them back from archive)
tail -n 100 improved_backfill.log
tail -n 100 reset_and_start.log
```

### Working with Specific Cryptocurrency Pairs
```bash
# Download data for a specific symbol and interval
python -c "from download_binance_data import download_monthly_klines; download_monthly_klines('BTCUSDT', '1h', 2024, 1)"

# Calculate indicators for a specific symbol
python -c "import pandas as pd; from indicators import add_bollinger_bands; from database import get_historical_data; from datetime import datetime, timedelta; end_date = datetime.now(); start_date = end_date - timedelta(days=30); df = get_historical_data('BTCUSDT', '1d', start_date, end_date); df_with_indicators = add_bollinger_bands(df); print(df_with_indicators.tail())"
```

### Git Operations
```bash
# Initialize Git repository (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit of Cryptocurrency Trading Platform"

# Add remote repository
git remote add origin [REPOSITORY_URL]

# Push to remote repository
git push -u origin main

# Clone repository
git clone [REPOSITORY_URL]
```

### Environment Management
```bash
# Check installed packages
pip list

# Update dependencies
pip install -r dependencies.txt

# Check environment variables
python -c "import os; print(dict(os.environ))"
```

## Custom Functions

### Get Latest Data for Any Symbol
```python
def get_latest_data(symbol, interval='1d', days=30):
    """Get the latest data for any symbol and interval"""
    from app import get_cached_data
    from datetime import datetime, timedelta
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = get_cached_data(symbol, interval, days, start_date, end_date)
    return df

# Usage example:
# python -c "from command_reference import get_latest_data; print(get_latest_data('ETHUSDT', '4h', 7))"
```

### Run Backtest for Trading Strategy
```python
def run_quick_backtest(symbol, interval='1d', days=90, strategy_params=None):
    """Run a quick backtest for a given symbol and strategy parameters"""
    from app import get_cached_data
    from strategy import backtest_strategy
    from datetime import datetime, timedelta
    
    if strategy_params is None:
        strategy_params = {
            'rsi_window': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'ema_window': 20
        }
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = get_cached_data(symbol, interval, days, start_date, end_date)
    results = backtest_strategy(df, strategy_params)
    
    return results

# Usage example:
# python -c "from command_reference import run_quick_backtest; print(run_quick_backtest('BTCUSDT'))"
```