# Cryptocurrency Trading Analysis Platform

A comprehensive Streamlit-based cryptocurrency trading analysis platform that provides advanced portfolio tracking, performance visualization, and interactive tools for crypto traders and enthusiasts.

## Features

- Real-time price data from Binance API
- Technical indicators: Bollinger Bands, MACD, RSI, EMA
- Trading signals based on technical analysis
- Portfolio performance tracking
- Sentiment analysis from social media and news
- Personalized crypto news digest
- Interactive candlestick charts
- Background database updates

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

## Customization

- Edit `backfill_database.py` to modify which cryptocurrencies are tracked
- Adjust trading parameters in the UI
- Modify technical indicators in `indicators.py`
- Change trading strategies in `strategy.py`

## Data Management

- Historical cryptocurrency data is stored in your PostgreSQL database
- You can reset the database by setting `RESET_DB=true` in your environment variables
- Background updates run every 15 minutes to keep data current