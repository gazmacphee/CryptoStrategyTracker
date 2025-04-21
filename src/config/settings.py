"""
Settings Module

This module contains all configurable parameters for the application.
Centralizing configuration here makes it easier to maintain and update.
"""

import os
import logging
from typing import Dict, Any, List
from datetime import datetime


# Database Configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://user:password@localhost:5432/crypto_db")

# Binance API Configuration
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
BINANCE_API_ENDPOINTS = [
    "https://api.binance.com/api/v3",
    "https://api1.binance.com/api/v3",
    "https://api2.binance.com/api/v3",
    "https://api3.binance.com/api/v3"
]
BINANCE_DATA_VISION_BASE_URL = "https://data.binance.vision"

# Logging Configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Data Configuration
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", 
    "XRPUSDT", "SOLUSDT", "DOTUSDT", "LTCUSDT", "LINKUSDT",
    "MATICUSDT", "AVAXUSDT"
]
AVAILABLE_SYMBOLS = DEFAULT_SYMBOLS + [
    "UNIUSDT", "SHIBUSDT", "ATOMUSDT", "ALGOUSDT", "LUNAUSDT", 
    "VETUSDT", "TRXUSDT", "ETCUSDT", "FILUSDT", "XLMUSDT",
    "THETAUSDT", "AXSUSDT", "ICPUSDT", "AAVEUSDT", "NEARUSDT"
]
DEFAULT_INTERVALS = ["15m", "30m", "1h", "4h", "1d"]
AVAILABLE_INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]

# Download Configuration
MAX_DOWNLOAD_WORKERS = 3
DOWNLOAD_TIMEOUT_SECONDS = 30
DOWNLOAD_RETRIES = 3
DOWNLOAD_RETRY_DELAY_SECONDS = 5

# Technical Indicators Configuration
TECHNICAL_INDICATORS = {
    "Bollinger Bands": {
        "enabled": True,
        "params": {
            "window": 20,
            "window_dev": 2.0
        }
    },
    "RSI": {
        "enabled": True,
        "params": {
            "window": 14
        }
    },
    "MACD": {
        "enabled": True,
        "params": {
            "fast": 12,
            "slow": 26,
            "signal": 9
        }
    },
    "EMA": {
        "enabled": True,
        "params": {
            "windows": [9, 21, 50, 200]
        }
    },
    "Stochastic": {
        "enabled": True,
        "params": {
            "k": 14,
            "d": 3,
            "smooth_k": 3
        }
    },
    "ATR": {
        "enabled": True,
        "params": {
            "window": 14
        }
    },
    "ADX": {
        "enabled": True,
        "params": {
            "window": 14
        }
    }
}

# ML Configuration
ML_MODELS_DIR = "models"
ML_FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "bb_upper", "bb_middle", "bb_lower", "rsi",
    "macd", "macd_signal", "macd_histogram",
    "ema_9", "ema_21", "ema_50", "ema_200",
    "stoch_k", "stoch_d", "atr", "adx"
]
ML_TARGET_COLUMN = "target"
ML_PREDICTION_PERIODS = 3
ML_TRAIN_TEST_SPLIT = 0.8
ML_LOOKBACK_PERIODS = 20

# UI Configuration
THEME = {
    "primary_color": "#1E88E5",
    "background_color": "#FAFAFA",
    "secondary_background_color": "#F0F2F6",
    "text_color": "#262730",
    "font": "sans-serif"
}
DEFAULT_CHART_HEIGHT = 500

# OpenAI Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4"

# Cache Configuration
CACHE_EXPIRY_SECONDS = 3600  # 1 hour

# System Configuration
APP_NAME = "Crypto Trading Analysis Platform"
APP_VERSION = "2.0.0"
APP_AUTHOR = "Data Science Team"

# Performance Configuration
THREAD_POOL_SIZE = 4