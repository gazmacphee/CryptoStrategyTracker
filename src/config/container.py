"""
Dependency Injection Container

This module provides a centralized dependency container to manage
application dependencies and facilitate testing through easy dependency mocking.
"""

import psycopg2
import logging
from typing import Dict, Any, Callable

from src.config import settings


class Container:
    """
    Container for managing application dependencies
    
    Implements a simple dependency injection container that manages
    service lifecycle and dependency resolution.
    """
    
    def __init__(self):
        """Initialize the container with empty registries"""
        self._services = {}
        self._factories = {}
        self._instances = {}
        self._initializing = set()
    
    def register_service(self, name: str, factory: Callable) -> None:
        """
        Register a service factory
        
        Args:
            name: Service name/identifier
            factory: Factory function that creates the service
        """
        self._factories[name] = factory
    
    def register_instance(self, name: str, instance: Any) -> None:
        """
        Register a pre-instantiated object
        
        Args:
            name: Service name/identifier
            instance: Instantiated object
        """
        self._instances[name] = instance
    
    def get(self, name: str) -> Any:
        """
        Get a service or instance
        
        Instantiates services on-demand and caches them for future use.
        
        Args:
            name: Service name/identifier
            
        Returns:
            Instantiated service or instance
        
        Raises:
            ValueError: If service isn't registered or a circular dependency is detected
        """
        # Check if already instantiated
        if name in self._instances:
            return self._instances[name]
        
        # Check if factory exists
        if name not in self._factories:
            raise ValueError(f"Service {name} is not registered")
        
        # Check for circular dependencies
        if name in self._initializing:
            raise ValueError(f"Circular dependency detected for {name}")
        
        # Track that we're initializing this service
        self._initializing.add(name)
        
        try:
            # Instantiate the service
            instance = self._factories[name](self)
            self._instances[name] = instance
            return instance
        finally:
            # Remove from initializing set
            self._initializing.remove(name)
    
    def reset(self) -> None:
        """
        Reset all instances (useful for testing)
        """
        self._instances = {}


# Create the container
container = Container()

# Register configuration
container.register_instance("config", settings)

# Register logging
logger = logging.getLogger("crypto_app")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(getattr(logging, settings.LOG_LEVEL))
container.register_instance("logger", logger)

# Database connection factory
def db_connection_factory(container) -> psycopg2.extensions.connection:
    """Create a database connection"""
    config = container.get("config")
    logger = container.get("logger")
    
    try:
        connection = psycopg2.connect(config.DATABASE_URL)
        connection.autocommit = True
        return connection
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise

container.register_service("db_connection", db_connection_factory)

# Add more service registrations here

def initialize_container():
    """
    Initialize the application container with all dependencies
    
    Registers all services needed for the application.
    This function should be called during application startup.
    """
    # This function will be extended as we add more services
    pass