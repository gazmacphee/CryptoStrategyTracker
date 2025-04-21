# Cryptocurrency Trading Analysis Platform Architecture

## Overview
This document outlines the architecture and process flow of the Cryptocurrency Trading Analysis Platform. The platform provides real-time cryptocurrency data visualization, technical analysis, portfolio tracking, and trading signals through a Streamlit-based web interface.

## System Components

### 1. Main Application Components
- **app.py**: Main Streamlit application entry point that handles UI rendering and user interactions
- **reset_and_start.py**: Initial startup script that resets the database, creates tables, and starts the data backfill process
- **start_improved_backfill.py**: Manages the background data collection process

### 2. Data Collection and Processing
- **binance_api.py**: Interfaces with Binance API for real-time and historical data
- **binance_file_listing.py**: Manages file listings from Binance Data Vision for historical data
- **download_binance_data.py**: Handles downloading historical data from Binance
- **download_single_pair.py**: Specialized module for downloading data for a single trading pair
- **backfill_database.py**: Coordinates the database backfill process

### 3. Data Storage and Management
- **database.py**: Core database functions (connection, queries, data retrieval)
- **data_loader.py**: Manages data loading process and tracks backfill progress

### 4. Analysis and Strategy
- **indicators.py**: Calculates technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **strategy.py**: Implements trading strategies and signal generation
- **utils.py**: Utility functions for data transformation and calculation

### 5. Content and External Data
- **crypto_news.py**: Fetches and processes cryptocurrency news
- **sentiment_scraper.py**: Collects and analyzes sentiment data

## Process Flow

### Application Startup Process
1. **Workflow Execution**: The workflow system executes `reset_and_start.py`
2. **Database Reset**: All existing database tables are dropped
3. **Table Creation**: Fresh database tables are created via `database.py`
4. **Backfill Initiation**: The background backfill process starts via `start_improved_backfill.py`
5. **Streamlit Launch**: The Streamlit application (`app.py`) is launched

### Data Acquisition Process
1. **Symbol and Interval Selection**: The system selects cryptocurrency pairs and time intervals to process
2. **Date Range Determination**: Available data date ranges are identified using `binance_file_listing.py`
3. **Data Download**: Historical data is downloaded using `download_binance_data.py` or `download_single_pair.py`
4. **Data Processing**: Data is cleaned, indicators are calculated using `indicators.py`
5. **Database Storage**: Processed data is saved to the PostgreSQL database

### User Interface Flow
1. **Main Navigation**: Users navigate through different tabs in the sidebar (Analysis, Portfolio, Sentiment, etc.)
2. **Data Selection**: Users select cryptocurrency pairs, time intervals, and date ranges
3. **Visualization**: The application renders charts and indicators based on selected data
4. **Technical Analysis**: Trading signals and strategies are evaluated using `strategy.py`
5. **Portfolio Management**: Users can track their cryptocurrency portfolios
6. **News and Sentiment**: Users can view news and sentiment analysis via `crypto_news.py` and `sentiment_scraper.py`

## Database Schema

The system uses a PostgreSQL database with the following main tables:

1. **historical_data**: Stores OHLCV (Open, High, Low, Close, Volume) price data
2. **indicators**: Stores calculated technical indicators
3. **trades**: Stores trade signals and actual/simulated trades
4. **portfolio**: Stores user portfolio entries
5. **benchmark_data**: Stores market benchmark data for comparison
6. **sentiment_data**: Stores sentiment analysis results

## Data Flow Diagram

```
┌─────────────────┐          ┌────────────────┐          ┌───────────────┐
│  Binance API &  │          │ Data Processing │          │   Database    │
│  Data Sources   │ ─────────▶  & Calculation  │ ─────────▶  (PostgreSQL) │
└─────────────────┘          └────────────────┘          └───────────────┘
                                                                 │
                                                                 ▼
┌─────────────────┐          ┌────────────────┐          ┌───────────────┐
│    User Web     │◀─────────│   Streamlit    │◀─────────│  Data Queries  │
│    Browser      │          │   Interface    │          │  & Analysis    │
└─────────────────┘          └────────────────┘          └───────────────┘
```

## Deployment Architecture

The application is deployed on Replit with the following configuration:

1. **Python 3.11**: Main programming language
2. **PostgreSQL 16**: Database for storing cryptocurrency data
3. **Streamlit**: Web framework for user interface
4. **NodeJS 20**: Supporting runtime for certain operations

The workflow is configured to automatically start the application by running `reset_and_start.py`, which initializes the database and launches the Streamlit server on port 5000.