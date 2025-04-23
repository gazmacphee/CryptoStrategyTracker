# Cryptocurrency Trading Analysis Platform

A comprehensive Streamlit-based cryptocurrency trading analysis platform that provides advanced portfolio tracking, performance visualization, and interactive tools for crypto traders and enthusiasts.

## Features

- Real-time price data from Binance API
- Technical indicators: Bollinger Bands, MACD, RSI, EMA
- Trading signals based on technical analysis
- Portfolio performance tracking
- Interactive candlestick charts
- Background database updates
- Clean, simplified architecture
- Advanced machine learning pattern recognition
- Automated trading recommendations
- Multi-timeframe market analysis
- Economic indicator correlation analysis

## Prerequisites

- Python 3.11 or higher
- PostgreSQL database
- Binance API key and secret (optional but recommended for real data)
- OpenAI API key (optional, for news and sentiment features)

## Quick Start

1. Clone this repository:
   ```
   git clone https://github.com/gazmacphee/CryptoStrategyTracker.git
   cd CryptoStrategyTracker
   ```

2. Install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r dependencies.txt
   ```

3. Set up your local PostgreSQL database:
   
   **Option 1:** Use the automated setup script (recommended):
   ```bash
   # Make sure PostgreSQL is running
   python check_local_db.py
   ```
   
   This script will:
   - Check if the `crypto` database exists, and create it if needed
   - Create all required tables with proper indexes
   - Grant necessary permissions to the postgres user
   - Update your `.env` file with the correct database connection settings
   
   **Option 2:** Manual setup by creating a `.env` file:
   ```
   # Create .env file from example
   cp .env.example .env
   
   # Edit the .env file to update database connection and API keys
   ```
   
   Make sure your `.env` file contains valid database connection information:
   ```
   # Database configuration
   DATABASE_URL=postgresql://postgres:2212@localhost:5432/crypto
   PGHOST=localhost
   PGPORT=5432
   PGUSER=postgres
   PGPASSWORD=2212
   PGDATABASE=crypto
   
   # API Keys (optional but recommended)
   BINANCE_API_KEY=your_binance_api_key
   BINANCE_API_SECRET=your_binance_api_secret
   OPENAI_API_KEY=your_openai_api_key
   
   # Application settings
   RESET_DB=false
   BACKFILL_ON_START=true
   ```

4. Run the application:
   ```
   streamlit run app.py
   ```

5. Access the application in your browser:
   ```
   http://localhost:5001
   ```

6. Populate your database with cryptocurrency data:
   ```
   python backfill_database.py
   ```
   
   Or set `BACKFILL_ON_START=true` in your `.env` file to automatically backfill when starting the app.

## Using Your Local PostgreSQL Database

The application has been configured to automatically detect and use your local PostgreSQL database:

- **Credentials**: Username "postgres" with password "2212"
- **Database**: The application will create a "crypto" database if it doesn't exist
- **Connection**: The application will automatically try to connect to localhost:5432

See `local_db_setup_guide.md` for detailed instructions on working with your local database.

## Environment Variables

- `RESET_DB`: Set to `true` to reset the database on startup (default: `false`)
- `BACKFILL_ON_START`: Set to `true` to start data backfill on startup (default: `true`)
- `BINANCE_API_KEY`: Your Binance API key
- `BINANCE_API_SECRET`: Your Binance API secret
- `OPENAI_API_KEY`: Your OpenAI API key

## Architecture

- **Frontend**: Streamlit web application
- **Database**: PostgreSQL for storing price data, indicators, and trading signals
- **Data Sources**: Binance API for real-time and historical price data
- **Background Processes**: Continuous data updates every 15 minutes

## Project Structure

The project has been cleaned up and organized for better maintainability:

```
/
├── app.py                         # Main Streamlit application
├── reset_and_start.py             # Application entry point script
├── database.py                    # Core database operations
├── database_extensions.py         # Advanced database functions
├── binance_api.py                 # Binance API integration
├── download_binance_data.py       # Data downloading functionality
├── backfill_database.py           # Database backfill operations
├── indicators.py                  # Technical indicators calculation
├── strategy.py                    # Trading strategy evaluation
├── advanced_ml.py                 # Advanced machine learning pattern recognition
├── direct_ml_fix.py               # ML recursion fix implementation
├── run_ml_fixed.py                # ML execution module
├── train_all_pattern_models.py    # Pattern model training utility
├── analyze_patterns_and_save.py   # Pattern analysis and recommendation tool
├── test_ml_integration.py         # Comprehensive ML testing utility
├── check_ml_data.py               # Database data verification utility
├── process_manager.py             # Background process orchestration
├── update_ml_process_config.py    # ML process scheduling configuration
├── utils.py                       # Utility functions
├── data/                          # Data storage directory
├── models/                        # Saved ML models directory
│   └── pattern_recognition/       # Pattern recognition models
└── legacy_archive/                # Archive of old/unused files
    ├── documentation/             # Project documentation
    ├── json/                      # JSON configuration files
    ├── logs/                      # Log files
    ├── scripts/                   # Utility scripts
    ├── ml_modules/                # Legacy ML modules
    ├── src/                       # Old modular architecture
    └── archive/                   # Additional archived files
```

## Machine Learning Capabilities

The platform includes sophisticated machine learning capabilities for pattern recognition and trading recommendations:

### Pattern Recognition

- Automatically detects chart patterns (e.g., support/resistance bounces, momentum shifts, double bottoms)
- Trains models on historical price data to recognize profitable trading patterns
- Analyzes patterns across multiple symbols and timeframes

### Trading Recommendations

- Generates actionable trading signals based on recognized patterns
- Assigns confidence scores to each recommendation
- Provides entry, exit, and stop-loss levels for each trade

### ML Architecture

- Database-first approach: All ML operations use data stored in PostgreSQL
- On-demand model training: New models are trained as sufficient data becomes available
- Scheduled pattern analysis: Hourly checks for new trading opportunities
- Integrated with process manager for reliable execution

### Using the ML Features

1. **Training Models**
   ```
   python train_all_pattern_models.py
   ```
   This script:
   - Checks for symbols with sufficient historical data
   - Trains pattern recognition models
   - Saves models to the `models/pattern_recognition/` directory

2. **Analyzing Patterns**
   ```
   python analyze_patterns_and_save.py
   ```
   This script:
   - Analyzes patterns across all markets
   - Identifies high-confidence trading opportunities
   - Saves recommendations to the database

3. **Testing ML Integration**
   ```
   python test_ml_integration.py
   ```
   This script:
   - Tests all ML components end-to-end
   - Verifies data accessibility, model training, and pattern detection
   - Reports success/failure status for each component

## Customization

- Edit `backfill_database.py` to modify which cryptocurrencies are tracked
- Adjust trading parameters in the UI
- Modify technical indicators in `indicators.py`
- Change trading strategies in `strategy.py`
- Adjust pattern detection thresholds in `advanced_ml.py`

## Data Management

- Historical cryptocurrency data is stored in your PostgreSQL database
- You can reset the database by setting `RESET_DB=true` in your environment variables
- Background updates run every 15 minutes to keep data current

## Using the Cleaned Project Structure

The project has been simplified to make it more maintainable and easier to understand:

1. **Starting the Application**
   ```
   python reset_and_start.py
   ```
   This script:
   - Resets the database if `RESET_DB=true`
   - Creates fresh tables
   - Starts the backfill process if `BACKFILL_ON_START=true`
   - Launches the Streamlit application

2. **Running Just the Streamlit App**
   ```
   streamlit run app.py
   ```

3. **Running Just the Backfill Process**
   ```
   python backfill_database.py
   ```

4. **Archived Code**
   
   Previous versions of the code, including the modular architecture and machine learning modules,
   are preserved in the `legacy_archive` directory for reference or future use.