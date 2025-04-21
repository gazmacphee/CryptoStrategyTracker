# Cryptocurrency Trading Analysis Platform

A streamlined cryptocurrency trading analysis platform that leverages data from Binance, technical indicators, and interactive visualizations.

## Project Structure

- **app.py**: Main Streamlit application with interactive UI
- **reset_and_start.py**: Entry point that sets up the database and launches the application
- **core/**: Core functionality modules
  - **database.py**: Database operations and connectivity
  - **binance_api.py**: Binance API integration
  - **download_binance_data.py**: Data downloading and processing
  - **backfill_database.py**: Database backfill operations
- **utils/**: Utility modules
  - **indicators.py**: Technical indicators calculation
  - **strategy.py**: Trading strategy evaluation
  - **utils.py**: General utility functions
- **data/**: Directory for storing data files (if needed)

## Getting Started

1. Run the reset_and_start.py script to initialize the database and start the application:
   ```
   python reset_and_start.py
   ```

2. Access the Streamlit interface at http://localhost:5000

## Features

- Historical cryptocurrency data visualization
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Trading strategy backtesting
- Portfolio tracking and performance analysis
- Interactive charts and customizable timeframes

## Environment Variables

The application requires the following environment variables:

- `DATABASE_URL`: PostgreSQL database connection string

## Dependencies

- Python 3.11+
- PostgreSQL database
- Streamlit
- Pandas, NumPy
- Plotly
- SQLAlchemy
- Requests

## Maintenance

This version of the application uses a simplified architecture with direct imports instead of the previous dependency injection pattern. This makes the code easier to maintain and understand.

For advanced features like machine learning and sentiment analysis, see the legacy_archive directory.