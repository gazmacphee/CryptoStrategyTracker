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

- Docker and Docker Compose
- Binance API key and secret (optional but recommended for real data)
- OpenAI API key (optional, for news and sentiment features)

## Quick Start

1. Clone this repository:
   ```
   git clone <repository-url>
   cd cryptocurrency-trading-platform
   ```

2. Create a `.env` file from the example:
   ```
   cp .env.example .env
   ```

3. Edit the `.env` file to add your API keys:
   ```
   BINANCE_API_KEY=your_binance_api_key
   BINANCE_API_SECRET=your_binance_api_secret
   OPENAI_API_KEY=your_openai_api_key
   ```

4. Build and run the Docker containers:
   ```
   docker-compose up -d
   ```

5. Access the application in your browser:
   ```
   http://localhost:5000
   ```

## Environment Variables

- `RESET_DB`: Set to `true` to reset the database on startup (default: `true`)
- `BACKFILL_ON_START`: Set to `true` to start data backfill on startup (default: `true`)
- `BINANCE_API_KEY`: Your Binance API key
- `BINANCE_API_SECRET`: Your Binance API secret
- `OPENAI_API_KEY`: Your OpenAI API key

## Deployment Options

### Local Development

```
docker-compose up
```

### Production Deployment

1. Create a `.env` file with your production configuration
2. Run:
   ```
   docker-compose -f docker-compose.yml up -d
   ```

### Manual Docker Build

If you prefer to build and run the Docker image manually:

```
docker build -t crypto-trading-platform .
docker run -p 5000:5000 --env-file .env crypto-trading-platform
```

## Accessing the Database

The PostgreSQL database is exposed on port 5432. You can connect to it using any PostgreSQL client:

```
Host: localhost
Port: 5432
User: postgres
Password: postgres
Database: crypto
```

## Building a Custom Image

To build and publish your own custom Docker image:

```
docker build -t yourusername/crypto-trading-platform:latest .
docker push yourusername/crypto-trading-platform:latest
```

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

- Database is persisted in Docker volumes
- You can reset the database by setting `RESET_DB=true`
- Background updates run every 15 minutes to keep data current