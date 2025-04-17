# Cryptocurrency Trading Analysis Platform

A comprehensive Streamlit-based cryptocurrency trading analysis platform that provides advanced portfolio tracking, performance visualization, and interactive learning tools for crypto enthusiasts.

## Features

- **Real-time Data**: Fetches cryptocurrency price data from Binance API
- **Technical Analysis**: Implements advanced indicators like Bollinger Bands, MACD, RSI, and more
- **Portfolio Tracking**: Monitor your cryptocurrency holdings and performance
- **Backtesting**: Test trading strategies on historical data
- **Sentiment Analysis**: Get insights from social media and news sources
- **News Digest**: AI-curated personalized crypto news
- **Trend Visualization**: Visual representation of market trends with emoji indicators

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Database**: PostgreSQL
- **Data Visualization**: Plotly
- **Data Analysis**: Pandas, NumPy
- **API Integration**: Binance API
- **AI Features**: OpenAI API

## Getting Started

### Local Development

1. Clone the repository
2. Install dependencies:
   ```
   pip install -e .
   ```
3. Set up environment variables (copy `.env.example` to `.env` and fill in the values)
4. Run the application:
   ```
   ./start.sh
   ```

### Docker Deployment

See the [DEPLOYMENT.md](DEPLOYMENT.md) file for Docker deployment instructions.

## API Keys

To use all features of this application, you'll need:

- **Binance API Key and Secret**: For real cryptocurrency data
- **OpenAI API Key**: For AI-powered news summaries and recommendations

Add these to your `.env` file or as environment variables.

## License

This project is licensed under the MIT License - see the LICENSE file for details.