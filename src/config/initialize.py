"""
Container Initialization Module

This module properly initializes the dependency container
with all necessary services in the correct order.
"""

import logging
import os
import psycopg2
from typing import Dict, Any, Optional, Callable

# Import the existing container
from src.config.container import container
from src.config import settings


def initialize_container():
    """Initialize the dependency container with all services"""
    # Import container methods from src.config.container
    from src.config.container import initialize_container as setup_core_container
    
    # Initialize the core container first (logger, db_connection)
    # This creates the logger and db_connection
    container_with_core = setup_core_container()
    
    # Ensure the logger is available before proceeding
    logger = container_with_core.get("logger")
    logger.info("Initializing dependency container with all services")
    
    # Import services here to avoid circular imports
    try:
        from src.data.repositories import (
            HistoricalDataRepository,
            TechnicalIndicatorsRepository,
            SentimentRepository,
            TradeRepository
        )
        
        # Register repositories
        container.register_instance("historical_data_repo", HistoricalDataRepository())
        container.register_instance("indicators_repo", TechnicalIndicatorsRepository())
        container.register_instance("sentiment_repo", SentimentRepository())
        container.register_instance("trade_repo", TradeRepository())
        
        # Import and register services
        from src.services.data_service import data_service_factory
        from src.services.indicators_service import indicators_service_factory
        
        # Register service factories
        container.register_service("data_service", data_service_factory)
        container.register_service("indicators_service", indicators_service_factory)
        
        # Try to import and register ML service if available
        try:
            from src.ml.ml_service import ml_service_factory
            container.register_service("ml_service", ml_service_factory)
        except ImportError:
            # ML service is optional, log a warning but continue
            logger.warning("ML service not available, skipping initialization")
        
        # Register backfill service last since it depends on other services
        from src.services.backfill_service import backfill_service_factory
        container.register_service("backfill_service", backfill_service_factory)
        
        logger.info("Dependency container initialization complete")
    except Exception as e:
        logger.error(f"Error initializing container: {e}")
        raise
    
    # Return the initialized container
    return container