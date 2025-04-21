"""
Settings UI Module

Provides configuration controls for the application.
"""

import streamlit as st
import os
import json
from datetime import datetime, timedelta

from src.config.container import container
from src.config import settings


def render_settings():
    """Render the settings UI"""
    st.header("Application Settings")
    
    # Get services
    backfill_service = container.get("backfill_service")
    
    # Create tabs for different settings
    tabs = st.tabs([
        "Data Management", 
        "API Configuration", 
        "UI Settings", 
        "ML Configuration",
        "System Information"
    ])
    
    with tabs[0]:
        render_data_management_tab(backfill_service)
    
    with tabs[1]:
        render_api_configuration_tab()
    
    with tabs[2]:
        render_ui_settings_tab()
    
    with tabs[3]:
        render_ml_configuration_tab()
    
    with tabs[4]:
        render_system_information_tab()


def render_data_management_tab(backfill_service):
    """Render the data management settings tab"""
    st.subheader("Data Management")
    
    # Data backfill section
    st.markdown("### Data Backfill")
    st.write("Configure and run data backfill operations to ensure your database is up-to-date.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Symbols selection
        available_symbols = settings.AVAILABLE_SYMBOLS
        default_symbols = settings.DEFAULT_SYMBOLS
        
        selected_symbols = st.multiselect(
            "Select Symbols",
            options=available_symbols,
            default=default_symbols[:5],  # First 5 default symbols
            key="backfill_symbols"
        )
        
        # Intervals selection
        available_intervals = settings.AVAILABLE_INTERVALS
        default_intervals = settings.DEFAULT_INTERVALS
        
        selected_intervals = st.multiselect(
            "Select Intervals",
            options=available_intervals,
            default=default_intervals,
            key="backfill_intervals"
        )
        
        # Backfill options
        days_back = st.slider(
            "Days to Backfill",
            min_value=30,
            max_value=1095,  # 3 years
            value=180,
            step=30,
            help="Number of days of historical data to backfill"
        )
        
        max_workers = st.slider(
            "Maximum Workers",
            min_value=1,
            max_value=10,
            value=settings.MAX_DOWNLOAD_WORKERS,
            help="Maximum number of concurrent download workers"
        )
        
        update_indicators = st.checkbox(
            "Update Technical Indicators",
            value=True,
            help="Calculate and store technical indicators for downloaded data"
        )
        
        # Start backfill button
        if st.button("Start Full Backfill"):
            if not selected_symbols or not selected_intervals:
                st.error("Please select at least one symbol and one interval")
            else:
                with st.spinner("Starting backfill process..."):
                    # Call the backfill service
                    result = backfill_service.start_backfill(
                        symbols=selected_symbols,
                        intervals=selected_intervals,
                        days_back=days_back,
                        max_workers=max_workers,
                        update_indicators=update_indicators
                    )
                    
                    if result['status'] == 'started':
                        st.success(f"Backfill process started: {result['message']}")
                    elif result['status'] == 'already_running':
                        st.warning(f"Backfill is already running: {result['message']}")
                    elif result['status'] == 'locked':
                        st.error(f"Backfill is locked: {result['message']}")
                    else:
                        st.error(f"Error starting backfill: {result}")
    
    with col2:
        # Show backfill status
        st.markdown("### Backfill Status")
        
        if backfill_service.is_running or backfill_service.lock_exists():
            backfill_progress = backfill_service.get_backfill_progress()
            
            if 'progress' in backfill_progress:
                progress = backfill_progress['progress']
                
                # Show progress bar
                progress_bar = st.progress(progress['percentage'] / 100)
                
                # Show progress metrics
                st.metric("Progress", f"{progress['percentage']:.1f}%")
                st.metric("Completed Tasks", f"{progress['completed_tasks']}/{progress['total_tasks']}")
                
                # Show time estimates
                if progress['elapsed_seconds'] > 0:
                    hours, remainder = divmod(progress['elapsed_seconds'], 3600)
                    minutes, seconds = divmod(remainder, 60)
                    time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                    st.metric("Elapsed Time", time_str)
                
                if progress['estimated_remaining_seconds'] > 0:
                    hours, remainder = divmod(progress['estimated_remaining_seconds'], 3600)
                    minutes, seconds = divmod(remainder, 60)
                    time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                    st.metric("Estimated Time Remaining", time_str)
                
                # Stop button
                if st.button("Stop Backfill"):
                    result = backfill_service.stop_backfill()
                    st.warning(f"Stopping backfill: {result['message']}")
        else:
            st.info("No backfill process is currently running.")
            
            # Check if there are any completed backfill results
            if os.path.exists("backfill_results.json"):
                try:
                    with open("backfill_results.json", "r") as f:
                        results = json.load(f)
                    
                    st.markdown("### Last Backfill Results")
                    st.json(results)
                except Exception as e:
                    st.error(f"Error loading backfill results: {e}")
    
    # Data pruning section
    st.markdown("---")
    st.markdown("### Data Pruning")
    st.write("Remove old or unnecessary data to free up database space.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        prune_older_than = st.slider(
            "Prune Data Older Than (days)",
            min_value=30,
            max_value=1095,  # 3 years
            value=365,
            step=30,
            help="Remove data older than this many days"
        )
        
        prune_symbols = st.multiselect(
            "Symbols to Prune",
            options=available_symbols,
            default=[],
            help="Select symbols to prune (leave empty for all)"
        )
        
        prune_intervals = st.multiselect(
            "Intervals to Prune",
            options=available_intervals,
            default=[],
            help="Select intervals to prune (leave empty for all)"
        )
    
    with col2:
        # Add confirmation to avoid accidental data loss
        st.warning("Data pruning will permanently delete data from the database!")
        
        confirmation = st.text_input(
            "Type 'CONFIRM' to enable pruning",
            value="",
            help="This confirmation helps prevent accidental data deletion"
        )
        
        if st.button("Prune Data", disabled=(confirmation != "CONFIRM")):
            st.warning("This is a placeholder for data pruning functionality.")
            st.success(f"Data pruning initiated for {', '.join(prune_symbols) if prune_symbols else 'all symbols'}")


def render_api_configuration_tab():
    """Render the API configuration settings tab"""
    st.subheader("API Configuration")
    
    # Binance API section
    st.markdown("### Binance API")
    st.write("Configure Binance API credentials for real-time data access.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # API Key
        binance_api_key = st.text_input(
            "Binance API Key",
            value=os.environ.get("BINANCE_API_KEY", ""),
            type="password",
            help="Your Binance API Key (optional)"
        )
        
        # API Secret
        binance_api_secret = st.text_input(
            "Binance API Secret",
            value=os.environ.get("BINANCE_API_SECRET", ""),
            type="password",
            help="Your Binance API Secret (optional)"
        )
        
        # Save button
        if st.button("Save Binance API Credentials"):
            # This is a placeholder - in a real implementation, we'd securely save these
            st.warning("This is a placeholder for saving API credentials.")
            st.success("Binance API credentials saved successfully")
    
    with col2:
        # API Test
        st.markdown("### Test API Connection")
        
        if st.button("Test Binance API Connection"):
            st.warning("This is a placeholder for API connection testing.")
            
            # Simulate API test
            if binance_api_key and binance_api_secret:
                st.success("API connection successful! You have access to real-time data.")
            else:
                st.warning("No API credentials provided. The application will use historical data only.")
    
    # OpenAI API Section (for sentiment analysis)
    st.markdown("---")
    st.markdown("### OpenAI API")
    st.write("Configure OpenAI API credentials for sentiment analysis and news summarization.")
    
    openai_api_key = st.text_input(
        "OpenAI API Key",
        value=os.environ.get("OPENAI_API_KEY", ""),
        type="password",
        help="Your OpenAI API Key for sentiment analysis features"
    )
    
    if st.button("Save OpenAI API Key"):
        st.warning("This is a placeholder for saving OpenAI API credentials.")
        st.success("OpenAI API key saved successfully")


def render_ui_settings_tab():
    """Render the UI settings tab"""
    st.subheader("UI Settings")
    
    # Theme settings
    st.markdown("### Theme Settings")
    st.write("Customize the application appearance.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Color theme
        primary_color = st.color_picker(
            "Primary Color",
            value=settings.THEME["primary_color"],
            help="Main accent color for the application"
        )
        
        background_color = st.color_picker(
            "Background Color",
            value=settings.THEME["background_color"],
            help="Main background color"
        )
        
        secondary_background_color = st.color_picker(
            "Secondary Background Color",
            value=settings.THEME["secondary_background_color"],
            help="Secondary background color for cards and sections"
        )
        
        text_color = st.color_picker(
            "Text Color",
            value=settings.THEME["text_color"],
            help="Main text color"
        )
    
    with col2:
        # Font settings
        font_family = st.selectbox(
            "Font Family",
            options=["sans-serif", "serif", "monospace", "cursive", "fantasy"],
            index=0,
            help="Main font family for the application"
        )
        
        # Chart settings
        st.markdown("### Chart Settings")
        
        default_chart_height = st.slider(
            "Default Chart Height",
            min_value=300,
            max_value=1000,
            value=settings.DEFAULT_CHART_HEIGHT if hasattr(settings, 'DEFAULT_CHART_HEIGHT') else 500,
            step=50,
            help="Default height for charts in pixels"
        )
        
        show_volume_by_default = st.checkbox(
            "Show Volume by Default",
            value=True,
            help="Always show volume on price charts"
        )
    
    # Save settings button
    if st.button("Save UI Settings"):
        st.warning("This is a placeholder for saving UI settings.")
        st.success("UI settings saved successfully")
        
        # Show theme preview
        st.markdown("### Theme Preview")
        
        st.markdown(f"""
        <div style="
            padding: 20px; 
            background-color: {background_color}; 
            color: {text_color}; 
            font-family: {font_family}; 
            border-radius: 10px;
        ">
            <h3 style="color: {primary_color};">Theme Preview</h3>
            <p>This is how your theme will look.</p>
            <div style="
                padding: 15px; 
                background-color: {secondary_background_color}; 
                border-radius: 5px;
                margin-top: 10px;
            ">
                <p>This is a secondary background element.</p>
                <button style="
                    background-color: {primary_color}; 
                    color: white; 
                    border: none; 
                    padding: 8px 15px; 
                    border-radius: 5px;
                ">Sample Button</button>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_ml_configuration_tab():
    """Render the ML configuration settings tab"""
    st.subheader("Machine Learning Configuration")
    
    # ML model settings
    st.markdown("### Model Settings")
    st.write("Configure machine learning model parameters.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature selection
        st.markdown("#### Features")
        
        use_technical_indicators = st.checkbox(
            "Use Technical Indicators",
            value=True,
            help="Include technical indicators as model features"
        )
        
        use_price_patterns = st.checkbox(
            "Use Price Patterns",
            value=True,
            help="Include price patterns as model features"
        )
        
        use_sentiment = st.checkbox(
            "Use Sentiment Analysis",
            value=True,
            help="Include market sentiment as model features"
        )
        
        use_market_regime = st.checkbox(
            "Use Market Regime Detection",
            value=True,
            help="Include market regime classification as model features"
        )
        
        lookback_periods = st.slider(
            "Lookback Periods",
            min_value=5,
            max_value=100,
            value=20,
            help="Number of previous periods to use for prediction"
        )
    
    with col2:
        # Model parameters
        st.markdown("#### Model Parameters")
        
        model_type = st.selectbox(
            "Base Model Type",
            options=["RandomForest", "GradientBoosting", "LSTM", "XGBoost"],
            index=3,
            help="Base machine learning algorithm to use"
        )
        
        ensemble_size = st.slider(
            "Ensemble Size",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of models in the ensemble"
        )
        
        training_split = st.slider(
            "Training Split",
            min_value=0.5,
            max_value=0.9,
            value=0.8,
            step=0.05,
            help="Portion of data to use for training (vs testing)"
        )
        
        prediction_threshold = st.slider(
            "Prediction Threshold",
            min_value=0.5,
            max_value=0.9,
            value=0.6,
            step=0.05,
            help="Confidence threshold for predictions"
        )
    
    # Automated retraining
    st.markdown("---")
    st.markdown("### Automated Retraining")
    st.write("Configure automated model retraining.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        enable_auto_retraining = st.checkbox(
            "Enable Automated Retraining",
            value=True,
            help="Automatically retrain models on a schedule"
        )
        
        retraining_frequency = st.selectbox(
            "Retraining Frequency",
            options=["Daily", "Weekly", "Monthly"],
            index=1,
            help="How often to retrain models",
            disabled=not enable_auto_retraining
        )
        
        retraining_time = st.time_input(
            "Retraining Time",
            value=datetime.strptime("02:00", "%H:%M").time(),
            help="Time of day to run retraining",
            disabled=not enable_auto_retraining
        )
    
    with col2:
        st.markdown("#### Notification Settings")
        
        notify_on_retraining = st.checkbox(
            "Notify on Retraining",
            value=True,
            help="Send notification when retraining completes",
            disabled=not enable_auto_retraining
        )
        
        notify_on_performance_change = st.checkbox(
            "Notify on Performance Change",
            value=True,
            help="Send notification when model performance changes significantly",
            disabled=not enable_auto_retraining
        )
    
    # Save settings button
    if st.button("Save ML Settings"):
        st.warning("This is a placeholder for saving ML settings.")
        st.success("ML settings saved successfully")


def render_system_information_tab():
    """Render the system information tab"""
    st.subheader("System Information")
    
    # Application information
    st.markdown("### Application Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Application Version", "2.0.0")
        st.metric("Database Status", "Connected")
        st.metric("API Status", "Connected")
    
    with col2:
        st.metric("Uptime", "3 days, 7 hours")
        st.metric("Last Update", "2025-04-20 17:30:45")
        st.metric("System Load", "42%")
    
    # Database statistics
    st.markdown("---")
    st.markdown("### Database Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", "1,243,892")
    
    with col2:
        st.metric("Symbols Stored", "12")
    
    with col3:
        st.metric("Oldest Data", "2023-01-01")
    
    # System health
    st.markdown("---")
    st.markdown("### System Health")
    
    # Show system metrics as a chart
    # This is a placeholder for real system metrics
    days = 7
    dates = [datetime.now() - timedelta(days=i) for i in range(days)]
    dates.reverse()  # Reverse to get ascending dates
    
    # Sample CPU usage data
    cpu_usage = [35, 42, 39, 47, 52, 45, 41]
    memory_usage = [28, 30, 29, 33, 36, 32, 31]
    disk_usage = [52, 53, 55, 54, 56, 57, 58]
    
    # Create figure
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = go.Figure()
    
    # Add CPU usage
    fig.add_trace(go.Scatter(
        x=dates,
        y=cpu_usage,
        mode='lines+markers',
        name='CPU Usage (%)',
        line=dict(color='red', width=2)
    ))
    
    # Add memory usage
    fig.add_trace(go.Scatter(
        x=dates,
        y=memory_usage,
        mode='lines+markers',
        name='Memory Usage (%)',
        line=dict(color='blue', width=2)
    ))
    
    # Add disk usage
    fig.add_trace(go.Scatter(
        x=dates,
        y=disk_usage,
        mode='lines+markers',
        name='Disk Usage (%)',
        line=dict(color='green', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title='System Resources Usage',
        xaxis_title='Date',
        yaxis_title='Usage (%)',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Show chart
    st.plotly_chart(fig, use_container_width=True)
    
    # System logs
    st.markdown("---")
    st.markdown("### System Logs")
    
    log_level = st.selectbox(
        "Log Level",
        options=["INFO", "WARNING", "ERROR", "DEBUG"],
        index=0
    )
    
    log_lines = st.slider(
        "Number of Lines",
        min_value=10,
        max_value=1000,
        value=100,
        step=10
    )
    
    if st.button("View Logs"):
        st.warning("This is a placeholder for viewing system logs.")
        
        # Show sample logs
        st.code("""
2025-04-21 10:15:32 - INFO - Application started
2025-04-21 10:15:33 - INFO - Database connection established
2025-04-21 10:15:34 - INFO - API connection established
2025-04-21 10:17:45 - INFO - User logged in: admin
2025-04-21 10:25:12 - WARNING - API rate limit approaching: 80%
2025-04-21 11:05:23 - INFO - Backfill process started
2025-04-21 11:45:17 - INFO - Backfill process completed
2025-04-21 12:10:09 - ERROR - Failed to retrieve data for SOLUSDT: API timeout
2025-04-21 13:22:55 - INFO - ML model retraining started
2025-04-21 13:40:38 - INFO - ML model retraining completed
        """)