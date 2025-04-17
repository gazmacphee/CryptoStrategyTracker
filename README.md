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
   git clone <repository-url>
   cd cryptocurrency-trading-platform
   ```

2. Install dependencies:
   ```
   pip install -e .
   ```
   
   Or install from dependencies.txt:
   ```
   pip install -r dependencies.txt
   ```

3. Set up environment variables:
   ```
   # Database configuration
   export DATABASE_URL=postgresql://username:password@hostname:port/database
   
   # API Keys (optional but recommended)
   export BINANCE_API_KEY=your_binance_api_key
   export BINANCE_API_SECRET=your_binance_api_secret
   export OPENAI_API_KEY=your_openai_api_key
   
   # Application settings
   export RESET_DB=false
   export BACKFILL_ON_START=true
   ```

4. Run the application:
   ```
   streamlit run app.py
   ```

5. Access the application in your browser:
   ```
   http://localhost:5000
   ```

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