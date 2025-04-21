"""
Configuration settings for the application

This module centralizes all configuration settings, making it easy to
modify application behavior without changing code.
"""

import os
from datetime import datetime, timedelta

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Data retrieval settings
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", 
    "XRPUSDT", "SOLUSDT", "DOTUSDT", "LTCUSDT", "LINKUSDT", 
    "MATICUSDT", "AVAXUSDT"
]

DEFAULT_INTERVALS = ["15m", "30m", "1h", "4h", "1d"]

# Map of interval display name to Binance API interval value
INTERVAL_MAPPING = {
    "1 minute": "1m",
    "3 minutes": "3m", 
    "5 minutes": "5m",
    "15 minutes": "15m",
    "30 minutes": "30m",
    "1 hour": "1h",
    "2 hours": "2h",
    "4 hours": "4h",
    "6 hours": "6h",
    "8 hours": "8h",
    "12 hours": "12h",
    "1 day": "1d",
    "3 days": "3d",
    "1 week": "1w",
    "1 month": "1M"
}

# Backfilling configuration
BACKFILL_DEFAULT_DAYS = 180
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
DOWNLOAD_TIMEOUT_SECONDS = 60

# Binance API configuration
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
BINANCE_API_ENDPOINTS = ["https://api.binance.com/api/v3", "https://api-gcp.binance.com/api/v3"]
BINANCE_DATA_VISION_BASE_URL = "https://data.binance.vision"

# Parallel processing configuration
MAX_DOWNLOAD_WORKERS = 4
MAX_PROCESSING_WORKERS = os.cpu_count() or 2  # Default to number of CPUs or 2
TASK_QUEUE_SIZE = 100

# Logging configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FILE = "app.log"

# Technical indicators configuration
TECHNICAL_INDICATORS = {
    "Bollinger Bands": {
        "enabled": True,
        "params": {"window": 20, "window_dev": 2}
    },
    "RSI": {
        "enabled": True,
        "params": {"window": 14}
    },
    "MACD": {
        "enabled": True,
        "params": {"fast": 12, "slow": 26, "signal": 9}
    },
    "EMA": {
        "enabled": True,
        "params": {"windows": [9, 21, 50, 200]}
    },
    "Stochastic": {
        "enabled": True,
        "params": {"k": 14, "d": 3, "smooth_k": 3}
    },
    "ATR": {
        "enabled": True,
        "params": {"window": 14}
    },
    "ADX": {
        "enabled": True,
        "params": {"window": 14}
    }
}

# Default trading strategy parameters
DEFAULT_STRATEGY_PARAMS = {
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "ema_fast": 9,
    "ema_slow": 21,
    "bb_threshold": 0.8,
    "macd_signal_threshold": 0,
    "stoch_overbought": 80,
    "stoch_oversold": 20,
    "adx_threshold": 25,
    "trailing_stop_pct": 0.02,
    "take_profit_pct": 0.05,
    "stop_loss_pct": 0.03
}

# ML settings
ML_MODELS_DIR = "models"
ML_DATA_LOOKBACK_DAYS = 90
ML_FEATURE_WINDOW = 30
ML_PREDICTION_HORIZON = 1
ML_CONFIDENCE_THRESHOLD = 0.65

# UI settings
THEME = {
    "primary_color": "#1E88E5",
    "background_color": "#FFFFFF",
    "secondary_background_color": "#F0F2F6",
    "text_color": "#262730",
    "font": "sans serif"
}

# Continuous learning configuration
CONTINUOUS_LEARNING_INTERVAL_HOURS = 24