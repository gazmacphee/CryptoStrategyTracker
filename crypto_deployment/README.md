# Crypto Trading Analysis Platform

A comprehensive Streamlit-based cryptocurrency trading analysis platform that provides advanced portfolio tracking, performance visualization, and interactive learning tools for crypto enthusiasts.

## Features

- Real-time and historical cryptocurrency price tracking
- Technical indicator analysis (Bollinger Bands, MACD, RSI, etc.)
- Trading strategy backtesting
- Portfolio performance tracking
- Data loading progress monitoring
- Sentiment analysis from social media and news
- AI-generated crypto news digest

## Deployment Instructions

### Prerequisites

- Docker and Docker Compose installed
- Binance API key and secret (optional for real-time data)
- OpenAI API key (optional for news summarization)

### Quick Start

1. Clone this repository
2. Create a `.env` file based on `.env.example` with your credentials
3. Run `docker-compose up -d` to start the application
4. Access the application at `http://localhost:5001`

### Environment Variables

Copy the `.env.example` file to `.env` and modify as needed:

```bash
cp .env.example .env
nano .env  # Edit with your values
```

### Docker Commands

Start the application:
```bash
docker-compose up -d
```

View logs:
```bash
docker-compose logs -f app
```

Stop the application:
```bash
docker-compose down
```

## Data Sources

The application uses:
- Binance Data Vision repository for historical data
- Binance API for real-time data (if credentials are provided)
- News APIs and web scraping for sentiment analysis

## Troubleshooting

- If you encounter database connection issues, make sure the PostgreSQL container is running:
  ```bash
  docker-compose ps
  ```
- To restart the application without losing data:
  ```bash
  docker-compose restart app
  ```
- To completely reset and rebuild the application:
  ```bash
  docker-compose down -v
  docker-compose up -d --build
  ```