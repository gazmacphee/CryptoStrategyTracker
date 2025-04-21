"""
Machine Learning Service Module

This module provides machine learning capabilities including predictions,
model training, market regime detection, and sentiment analysis.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from joblib import dump, load

from src.config.container import container
from src.config import settings


class MLService:
    """Service for machine learning operations"""
    
    def __init__(self):
        """Initialize the ML service"""
        self.logger = container.get("logger")
        self.data_service = container.get("data_service")
        
        # Make sure models directory exists
        os.makedirs(settings.ML_MODELS_DIR, exist_ok=True)
    
    def predict_price_movement(
        self,
        symbol: str,
        interval: str,
        prediction_periods: int = 1,
        model_type: str = "Basic"
    ) -> Dict[str, Any]:
        """
        Generate price movement predictions
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            prediction_periods: Number of periods to predict
            model_type: Type of model to use
            
        Returns:
            Dictionary with prediction results
        """
        # This is a placeholder for real ML prediction functionality
        # In a real implementation, this would load trained models and make predictions
        
        self.logger.info(f"Generating predictions for {symbol}/{interval} using {model_type} model")
        
        # Get recent price data for context
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        
        df = self.data_service.get_klines_data(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time
        )
        
        if df.empty:
            return {
                "status": "error",
                "message": f"No data available for {symbol}/{interval}"
            }
        
        # Get the last price
        last_price = df['close'].iloc[-1]
        
        # For demo purposes, generate random predictions
        # In a real implementation, this would use trained ML models
        np.random.seed(int(datetime.now().timestamp()))
        
        predictions = []
        current_price = last_price
        
        for i in range(prediction_periods):
            # Generate random movement (-2% to +2%)
            change_pct = (np.random.random() - 0.5) * 0.04
            predicted_price = current_price * (1 + change_pct)
            
            # Adjust confidence based on prediction period
            # (confidence decreases for predictions further in the future)
            confidence = max(0.8 - (i * 0.05), 0.5)
            
            predictions.append({
                "period": i + 1,
                "price": predicted_price,
                "change_pct": change_pct * 100,
                "confidence": confidence
            })
            
            # Use predicted price as basis for next prediction
            current_price = predicted_price
        
        return {
            "status": "success",
            "symbol": symbol,
            "interval": interval,
            "current_price": last_price,
            "model_type": model_type,
            "predictions": predictions,
            "generated_at": datetime.now().isoformat()
        }
    
    def train_model(
        self,
        symbol: str,
        interval: str,
        training_days: int = 90,
        model_type: str = "Basic"
    ) -> Dict[str, Any]:
        """
        Train a machine learning model
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            training_days: Number of days of data to use for training
            model_type: Type of model to train
            
        Returns:
            Dictionary with training results
        """
        # This is a placeholder for real ML training functionality
        
        self.logger.info(f"Training {model_type} model for {symbol}/{interval} using {training_days} days of data")
        
        # Get training data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=training_days)
        
        df = self.data_service.get_klines_data(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time
        )
        
        if df.empty:
            return {
                "status": "error",
                "message": f"No data available for {symbol}/{interval}"
            }
        
        # For demo purposes, pretend to train a model
        # In a real implementation, this would create and train an actual ML model
        model_path = os.path.join(settings.ML_MODELS_DIR, f"{symbol}_{interval}_model.joblib")
        
        # Save some metadata as the "model"
        model_data = {
            "symbol": symbol,
            "interval": interval,
            "training_days": training_days,
            "model_type": model_type,
            "data_points": len(df),
            "training_date": datetime.now().isoformat()
        }
        
        # Save metadata to disk
        try:
            dump(model_data, model_path)
            
            return {
                "status": "success",
                "symbol": symbol,
                "interval": interval,
                "model_type": model_type,
                "data_points": len(df),
                "model_path": model_path,
                "training_date": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            
            return {
                "status": "error",
                "message": f"Error saving model: {e}"
            }
    
    def detect_market_regime(
        self,
        symbol: str,
        interval: str,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        Detect the current market regime
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            lookback_days: Number of days to analyze
            
        Returns:
            Dictionary with market regime detection results
        """
        # This is a placeholder for real market regime detection
        
        self.logger.info(f"Detecting market regime for {symbol}/{interval}")
        
        # Get data for analysis
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        df = self.data_service.get_klines_data(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time
        )
        
        if df.empty:
            return {
                "status": "error",
                "message": f"No data available for {symbol}/{interval}"
            }
        
        # Calculate basic metrics for regime detection
        price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
        volatility = df['close'].pct_change().std() * 100
        
        # Determine market regime based on price change and volatility
        if abs(price_change) < 5 and volatility < 2:
            regime = "Ranging"
            description = "The market is in a sideways movement pattern with low volatility."
        elif price_change > 0 and volatility < 3:
            regime = "Bullish Trend"
            description = "The market is in a steady upward trend."
        elif price_change < 0 and volatility < 3:
            regime = "Bearish Trend"
            description = "The market is in a steady downward trend."
        elif volatility > 4:
            regime = "Volatile"
            description = "The market is showing high volatility with unpredictable movements."
        else:
            regime = "Mixed"
            description = "The market is showing mixed signals."
        
        return {
            "status": "success",
            "symbol": symbol,
            "interval": interval,
            "regime": regime,
            "description": description,
            "metrics": {
                "price_change_pct": price_change,
                "volatility_pct": volatility,
                "analysis_period_days": lookback_days
            }
        }
    
    def analyze_sentiment(
        self,
        symbol: str,
        lookback_days: int = 7
    ) -> Dict[str, Any]:
        """
        Analyze market sentiment
        
        Args:
            symbol: Trading pair symbol
            lookback_days: Number of days to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # This is a placeholder for real sentiment analysis
        
        self.logger.info(f"Analyzing sentiment for {symbol}")
        
        # Generate sample sentiment data
        # In a real implementation, this would retrieve actual sentiment data from a database
        
        # Extract the base currency (e.g., BTC from BTCUSDT)
        base_currency = symbol.split("USDT")[0] if "USDT" in symbol else symbol
        
        # Generate random sentiment scores for demonstration
        np.random.seed(int(datetime.now().timestamp()))
        
        end_time = datetime.now()
        sentiment_data = []
        
        sources = ["Twitter", "Reddit", "News", "Trading View"]
        
        for i in range(lookback_days):
            day = end_time - timedelta(days=i)
            
            for source in sources:
                # Generate a random sentiment score (-1 to 1)
                sentiment_score = (np.random.random() * 2) - 1
                
                sentiment_data.append({
                    "date": day.strftime("%Y-%m-%d"),
                    "source": source,
                    "sentiment_score": sentiment_score,
                    "sentiment_label": "Positive" if sentiment_score > 0.3 else "Negative" if sentiment_score < -0.3 else "Neutral"
                })
        
        # Calculate overall sentiment
        sentiment_scores = [item["sentiment_score"] for item in sentiment_data]
        average_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        if average_sentiment > 0.5:
            overall_sentiment = "Very Positive"
            description = f"The market sentiment for {base_currency} is very positive, indicating potential bullish momentum."
        elif average_sentiment > 0.1:
            overall_sentiment = "Positive"
            description = f"The market sentiment for {base_currency} is positive, but not overwhelmingly so."
        elif average_sentiment > -0.1:
            overall_sentiment = "Neutral"
            description = f"The market sentiment for {base_currency} is neutral, with mixed signals."
        elif average_sentiment > -0.5:
            overall_sentiment = "Negative"
            description = f"The market sentiment for {base_currency} is negative, indicating potential bearish pressure."
        else:
            overall_sentiment = "Very Negative"
            description = f"The market sentiment for {base_currency} is very negative, indicating strong bearish pressure."
        
        return {
            "status": "success",
            "symbol": symbol,
            "overall_sentiment": overall_sentiment,
            "sentiment_score": average_sentiment,
            "description": description,
            "sentiment_data": sentiment_data
        }


# Register in the container
def ml_service_factory(container):
    return MLService()

container.register_service("ml_service", ml_service_factory)