"""
Dependency Injection Container

This module provides a container for managing dependencies and services.
The DI container helps manage component lifecycles and improves testability.
"""

import logging
import psycopg2
import os
from typing import Dict, Any, Optional, Callable

from src.config import settings


class Container:
    """
    Dependency Injection Container
    
    Manages service instantiation and lifetime, centralizing dependency management.
    """
    
    def __init__(self):
        """Initialize the container"""
        self._instances = {}
        self._factories = {}
    
    def register_instance(self, key: str, instance: Any) -> None:
        """
        Register an instance
        
        Args:
            key: The name to register the instance under
            instance: The instance to register
        """
        self._instances[key] = instance
    
    def register_factory(self, key: str, factory: Callable[['Container'], Any]) -> None:
        """
        Register a factory function
        
        Args:
            key: The name to register the factory under
            factory: The factory function that creates the instance
        """
        self._factories[key] = factory
    
    def register_service(self, key: str, factory: Callable[['Container'], Any]) -> None:
        """
        Register a service factory and create the instance
        
        Args:
            key: The name to register the service under
            factory: The factory function that creates the service
        """
        instance = factory(self)
        self.register_instance(key, instance)
    
    def get(self, key: str) -> Any:
        """
        Get an instance by key
        
        Args:
            key: The key to look up
            
        Returns:
            The instance associated with the key
        
        Raises:
            KeyError: If the key is not registered
        """
        # Check if we already have an instance
        if key in self._instances:
            return self._instances[key]
        
        # Check if we have a factory
        if key in self._factories:
            # Create the instance
            instance = self._factories[key](self)
            
            # Save for future use
            self._instances[key] = instance
            
            return instance
        
        raise KeyError(f"No instance or factory registered for key: {key}")


# Create a global container instance
container = Container()


def setup_logging() -> logging.Logger:
    """
    Set up the application logger
    
    Returns:
        Configured logger instance
    """
    # Configure logging
    logger = logging.getLogger("crypto_platform")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # Create formatter
    formatter = logging.Formatter(settings.LOG_FORMAT)
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger


def create_db_connection() -> psycopg2.extensions.connection:
    """
    Create a database connection
    
    Returns:
        Active database connection
    
    Raises:
        Exception: If the connection cannot be established
    """
    logger = container.get("logger")
    
    try:
        # Try to connect using the DATABASE_URL environment variable
        if "DATABASE_URL" in os.environ:
            logger.info(f"Connecting to database using DATABASE_URL: {os.environ['DATABASE_URL'].split('@')[1]}")
            conn = psycopg2.connect(os.environ["DATABASE_URL"])
        else:
            # Fall back to the settings value
            logger.info("Connecting to database using settings.DATABASE_URL")
            conn = psycopg2.connect(settings.DATABASE_URL)
        
        # Configure the connection
        conn.autocommit = True
        
        logger.info("Successfully connected using DATABASE_URL")
        return conn
    
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise


def initialize_container() -> None:
    """Initialize the dependency container with core services"""
    # Register logger
    container.register_instance("logger", setup_logging())
    
    # Register database connection
    container.register_factory("db_connection", create_db_connection)