"""
Main Application Module

This is the main entry point for the Crypto Trading Analysis Platform.
It initializes the application, sets up dependencies, and starts the UI.
"""

import os
import sys
import logging
import streamlit as st
from datetime import datetime, timedelta

# Add the parent directory to the path to enable imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import local modules from src package
from src.config.container import container, initialize_container
from src.config import settings
from src.data.database import create_tables
from src.services.backfill_service import BackfillService
from src.ui.dashboard import render_dashboard
from src.ui.data_browser import render_data_browser
from src.ui.trading import render_trading
from src.ui.analysis import render_analysis
from src.ui.ml_prediction import render_ml_predictions
from src.ui.settings_ui import render_settings


def setup_app():
    """Set up the application environment"""
    # Initialize the dependency container with all services
    # Use the returned container with all services registered
    global container
    container = initialize_container()
    
    # Make sure necessary directories exist
    os.makedirs('models', exist_ok=True)
    
    # Configure logging
    logger = container.get("logger")
    
    # Create database tables
    try:
        create_tables()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        st.error(f"Database initialization failed: {e}")


def run_app():
    """Run the Streamlit application"""
    # Set page config
    st.set_page_config(
        page_title="Crypto Trading Analysis Platform",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/crypto-trading-platform',
            'Report a bug': 'https://github.com/yourusername/crypto-trading-platform/issues',
            'About': "# Crypto Trading Analysis Platform\nA comprehensive platform for cryptocurrency trading analysis."
        }
    )
    
    # Get services
    logger = container.get("logger")
    data_service = container.get("data_service")
    backfill_service = container.get("backfill_service")
    
    # Custom theme from settings
    st.markdown(f"""
    <style>
        :root {{
            --primary-color: {settings.THEME['primary_color']};
            --background-color: {settings.THEME['background_color']};
            --secondary-background-color: {settings.THEME['secondary_background_color']};
            --text-color: {settings.THEME['text_color']};
            --font: {settings.THEME['font']};
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.sidebar.title("Crypto Trading Platform")
    
    # Navigation
    app_mode = st.sidebar.selectbox(
        "Navigation",
        ["Dashboard", "Data Browser", "Technical Analysis", "ML Predictions", "Trading", "Settings"],
        key="navigation"
    )
    
    # Check backfill status
    if backfill_service.lock_exists() or backfill_service.is_running:
        backfill_progress = backfill_service.get_backfill_progress()
        if 'progress' in backfill_progress:
            progress = backfill_progress['progress']
            
            # Show backfill progress
            st.sidebar.markdown("---")
            st.sidebar.subheader("Data Backfill Progress")
            
            progress_bar = st.sidebar.progress(progress['percentage'] / 100)
            st.sidebar.text(f"{progress['percentage']}% complete")
            st.sidebar.text(f"{progress['completed_tasks']}/{progress['total_tasks']} tasks")
            
            # Show time estimates
            if progress['elapsed_seconds'] > 0:
                hours, remainder = divmod(progress['elapsed_seconds'], 3600)
                minutes, seconds = divmod(remainder, 60)
                time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                st.sidebar.text(f"Elapsed time: {time_str}")
            
            if progress['estimated_remaining_seconds'] > 0:
                hours, remainder = divmod(progress['estimated_remaining_seconds'], 3600)
                minutes, seconds = divmod(remainder, 60)
                time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                st.sidebar.text(f"Estimated time remaining: {time_str}")
    
    # Render selected section
    if app_mode == "Dashboard":
        render_dashboard()
    elif app_mode == "Data Browser":
        render_data_browser()
    elif app_mode == "Technical Analysis":
        render_analysis()
    elif app_mode == "ML Predictions":
        render_ml_predictions()
    elif app_mode == "Trading":
        render_trading()
    elif app_mode == "Settings":
        render_settings()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2025 Crypto Trading Platform")
    

if __name__ == "__main__":
    # Setup application
    setup_app()
    
    # Run the Streamlit app
    run_app()