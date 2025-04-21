"""
Data Browser UI Module

Provides functionality to browse, inspect, and manage the historical data.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from src.config.container import container


def render_data_browser():
    """Render the data browser UI"""
    st.header("Data Browser")
    
    # Get services
    data_service = container.get("data_service")
    backfill_service = container.get("backfill_service")
    
    # Sidebar controls
    st.sidebar.subheader("Data Controls")
    
    # Get available symbols
    available_symbols = data_service.get_available_symbols()
    
    # Symbol selection
    selected_symbol = st.sidebar.selectbox(
        "Select Cryptocurrency",
        options=available_symbols,
        index=0 if "BTCUSDT" in available_symbols else 0,
        key="browser_symbol"
    )
    
    # Timeframe selection
    selected_interval = st.sidebar.selectbox(
        "Select Timeframe",
        options=["15m", "30m", "1h", "4h", "1d"],
        index=2,  # Default to 1h
        key="browser_interval"
    )
    
    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        days_back = st.number_input(
            "Days Back",
            min_value=1,
            max_value=365,
            value=30,
            key="browser_days_back"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            key="browser_end_date"
        )
    
    # Calculate start date
    start_date = end_date - timedelta(days=days_back)
    
    # Convert to datetime
    start_time = datetime.combine(start_date, datetime.min.time())
    end_time = datetime.combine(end_date, datetime.max.time())
    
    # Data actions
    st.sidebar.subheader("Data Actions")
    
    # Inspect data
    inspect_data = st.sidebar.button("Inspect Data")
    
    # Download new data
    start_backfill = st.sidebar.button("Start Backfill")
    
    # Check for gaps
    check_gaps = st.sidebar.button("Check for Gaps")
    
    # Main content area
    if inspect_data:
        with st.spinner(f"Loading data for {selected_symbol}/{selected_interval}..."):
            # Get historical data
            df = data_service.get_klines_data(
                symbol=selected_symbol,
                interval=selected_interval,
                start_time=start_time,
                end_time=end_time
            )
            
            if df.empty:
                st.error(f"No data available for {selected_symbol}/{selected_interval} in the selected time range")
                return
            
            # Display data statistics
            st.subheader("Data Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Candles", f"{len(df)}")
            
            with col2:
                # Calculate date range
                min_date = df['timestamp'].min()
                max_date = df['timestamp'].max()
                date_range = (max_date - min_date).days + 1
                
                st.metric("Date Range", f"{date_range} days")
            
            with col3:
                # Calculate completeness
                expected_candles = data_service._calculate_expected_candles(
                    selected_interval, min_date, max_date
                ) if not df.empty else 0
                
                completeness = (len(df) / expected_candles) * 100 if expected_candles > 0 else 0
                
                st.metric("Completeness", f"{completeness:.1f}%")
            
            with col4:
                # Calculate freshness
                if not df.empty:
                    latest_time = df['timestamp'].max()
                    now = datetime.now()
                    hours_old = (now - latest_time).total_seconds() / 3600
                    
                    st.metric("Data Freshness", f"{hours_old:.1f} hours old")
                else:
                    st.metric("Data Freshness", "N/A")
            
            # Display price chart
            st.subheader("Price Chart")
            
            fig = go.Figure()
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=selected_symbol
            ))
            
            # Update layout
            fig.update_layout(
                title=f"{selected_symbol} Price Chart ({selected_interval})",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=500,
                xaxis_rangeslider_visible=False
            )
            
            # Show chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Display data table
            st.subheader("Data Table")
            
            # Format the data for display
            display_df = df.copy()
            
            # Convert timestamp to readable format
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Show data table
            st.dataframe(display_df, use_container_width=True)
            
            # Export options
            st.subheader("Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export to CSV
                if st.button("Export to CSV"):
                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{selected_symbol}_{selected_interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                # Export to JSON
                if st.button("Export to JSON"):
                    json = display_df.to_json(orient="records")
                    st.download_button(
                        label="Download JSON",
                        data=json,
                        file_name=f"{selected_symbol}_{selected_interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
    
    # Handle backfill request
    if start_backfill:
        with st.spinner("Starting backfill process..."):
            # Start backfill
            result = backfill_service.start_backfill(
                symbols=[selected_symbol],
                intervals=[selected_interval],
                days_back=days_back
            )
            
            # Display result
            if result['status'] == 'started':
                st.success(f"Backfill process started: {result['message']}")
            elif result['status'] == 'already_running':
                st.warning(f"Backfill is already running: {result['message']}")
            elif result['status'] == 'locked':
                st.error(f"Backfill is locked: {result['message']}")
            else:
                st.error(f"Error starting backfill: {result['message']}")
            
            # Display progress information
            if result['status'] == 'started':
                progress = backfill_service.get_backfill_progress()
                
                if 'progress' in progress:
                    st.write(f"Completed tasks: {progress['progress']['completed_tasks']}/{progress['progress']['total_tasks']}")
                    st.write(f"Progress: {progress['progress']['percentage']:.1f}%")
    
    # Handle gap check request
    if check_gaps:
        with st.spinner(f"Checking for gaps in {selected_symbol}/{selected_interval} data..."):
            # Detect gaps
            gaps = data_service.detect_data_gaps(
                symbol=selected_symbol,
                interval=selected_interval,
                start_time=start_time,
                end_time=end_time
            )
            
            # Display gaps
            if not gaps:
                st.success(f"No gaps found in {selected_symbol}/{selected_interval} data")
            else:
                st.warning(f"Found {len(gaps)} gaps in {selected_symbol}/{selected_interval} data")
                
                # Create a dataframe for display
                gaps_df = pd.DataFrame([
                    {
                        'Start Time': gap['start_time'].strftime('%Y-%m-%d %H:%M'),
                        'End Time': gap['end_time'].strftime('%Y-%m-%d %H:%M'),
                        'Duration (hours)': (gap['end_time'] - gap['start_time']).total_seconds() / 3600,
                        'Missing Candles': gap['expected_candles']
                    }
                    for gap in gaps
                ])
                
                # Display gaps table
                st.dataframe(gaps_df, use_container_width=True)
                
                # Fill gaps option
                if st.button("Fill Gaps"):
                    with st.spinner("Filling gaps..."):
                        # Fill gaps
                        fill_result = data_service.fill_data_gaps(gaps)
                        
                        # Display results
                        if fill_result['filled_gaps'] > 0:
                            st.success(f"Filled {fill_result['filled_gaps']}/{fill_result['total_gaps']} gaps with {fill_result['total_candles_filled']} candles")
                        else:
                            st.error("Failed to fill any gaps")
                        
                        # Show any failed gaps
                        if fill_result['failed_gaps']:
                            st.warning(f"{len(fill_result['failed_gaps'])} gaps could not be filled")
                            
                            # Create a dataframe for display
                            failed_df = pd.DataFrame([
                                {
                                    'Start Time': gap['start_time'].strftime('%Y-%m-%d %H:%M'),
                                    'End Time': gap['end_time'].strftime('%Y-%m-%d %H:%M'),
                                    'Duration (hours)': (gap['end_time'] - gap['start_time']).total_seconds() / 3600,
                                    'Missing Candles': gap['expected_candles']
                                }
                                for gap in fill_result['failed_gaps']
                            ])
                            
                            # Display failed gaps table
                            st.dataframe(failed_df, use_container_width=True)
    
    # Display instructions if no action is selected
    if not inspect_data and not start_backfill and not check_gaps:
        st.info("""
        ## Data Browser Instructions
        
        This tool allows you to inspect historical price data, check for gaps, and perform backfill operations.
        
        ### To get started:
        1. Select a cryptocurrency and timeframe in the sidebar
        2. Choose a date range for analysis
        3. Click one of the action buttons:
           - **Inspect Data** - View and analyze historical data
           - **Start Backfill** - Download historical data for the selected symbol/interval
           - **Check for Gaps** - Identify and fix gaps in the historical data
        
        Use this tool to ensure your data is complete and up-to-date for accurate analysis.
        """)