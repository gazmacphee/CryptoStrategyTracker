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
        # First import all service factories
        from src.services.data_service import data_service_factory
        from src.services.indicators_service import indicators_service_factory
        from src.services.backfill_service import backfill_service_factory
        
        # Get logger for status updates
        logger = container.get("logger")
        logger.info("Registering services in the dependency container")
        
        # Register repositories first
        from src.data.repositories import (
            HistoricalDataRepository, 
            TechnicalIndicatorsRepository,
            SentimentRepository,
            TradeRepository
        )
        
        # Get a database connection
        db_connection = container.get("db_connection")
        
        # Create repository instances with explicit dependencies
        historical_repo = HistoricalDataRepository(db_connection=db_connection, logger=logger)
        indicators_repo = TechnicalIndicatorsRepository(db_connection=db_connection, logger=logger)
        sentiment_repo = SentimentRepository(db_connection=db_connection, logger=logger)
        trade_repo = TradeRepository(db_connection=db_connection, logger=logger)
        
        # Register repositories in the container
        container.register_instance("historical_data_repo", historical_repo)
        container.register_instance("indicators_repo", indicators_repo)
        container.register_instance("sentiment_repo", sentiment_repo)
        container.register_instance("trade_repo", trade_repo)
        
        logger.info("Repositories registered successfully")
        
        # Step 1: Register factories without creating instances
        logger.info("Registering service factories")
        container.register_factory("data_service_factory", lambda c: data_service_factory(c))
        container.register_factory("indicators_service_factory", lambda c: indicators_service_factory(c))
        container.register_factory("backfill_service_factory", lambda c: backfill_service_factory(c))
        
        # Step 2: Create independent service instances first
        logger.info("Creating service instances in the correct order")
        
        # Create data service
        logger.info("Creating data_service")
        data_service = data_service_factory(container)
        container.register_instance("data_service", data_service)
        
        # Create indicators service (without data_service for now)
        logger.info("Creating indicators_service")
        indicators_service = indicators_service_factory(container)
        container.register_instance("indicators_service", indicators_service)
        
        # Step 3: Resolve circular dependencies after all services are created
        logger.info("Resolving circular dependencies between services")
        indicators_service.initialize(data_service=data_service)
        
        # Try to import and register ML service if available
        try:
            from src.ml.ml_service import ml_service_factory
            ml_service = ml_service_factory(container)
            container.register_instance("ml_service", ml_service)
            logger.info("ML service registered successfully")
        except ImportError:
            # ML service is optional, log a warning but continue
            logger.warning("ML service not available, skipping initialization")
        
        # Register backfill service last since it depends on other services
        logger.info("Creating backfill_service")
        backfill_service = backfill_service_factory(container)
        container.register_instance("backfill_service", backfill_service)
        logger.info("Backfill service registered successfully")
        
        logger.info("Dependency container initialization complete")
    except Exception as e:
        logger.error(f"Error initializing container: {e}")
        raise
    
    # Return the initialized container
    return container