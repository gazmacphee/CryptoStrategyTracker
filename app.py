import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
import database
from utils import calculate_portfolio_value, get_portfolio_performance_history, normalize_comparison_data
from datetime import datetime, timedelta
import os
import threading
import subprocess

# Import local modules
from database import create_tables, save_historical_data, get_historical_data
from binance_api import get_klines_data, get_available_symbols
from indicators import add_bollinger_bands, add_macd, add_rsi, add_ema, add_stochastic, add_atr, add_adx
from strategy import evaluate_buy_sell_signals, backtest_strategy, find_optimal_strategy
from utils import timeframe_to_seconds, timeframe_to_interval, get_timeframe_options, calculate_trade_statistics

# Set page config
st.set_page_config(
    page_title="Crypto Trading Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
create_tables()

# Helper functions for database status and auto-backfill
def get_last_update_time(symbol=None, interval=None):
    """Get the last update time for data in the database"""
    from database import get_db_connection
    
    conn = get_db_connection()
    cursor = conn.cursor()
    last_update = None
    
    try:
        if symbol and interval:
            # Get specific symbol/interval update time
            cursor.execute(
                "SELECT MAX(timestamp) FROM historical_data WHERE symbol = %s AND interval = %s",
                (symbol, interval)
            )
        else:
            # Get overall latest update
            cursor.execute("SELECT MAX(timestamp) FROM historical_data")
            
        result = cursor.fetchone()
        if result and result[0]:
            last_update = result[0]
    except Exception as e:
        print(f"Error getting last update time: {e}")
    finally:
        cursor.close()
        conn.close()
    
    return last_update

def get_database_stats():
    """Get statistics about database population"""
    from database import get_db_connection
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Initialize with zeros in case tables don't exist yet
    price_count = 0
    indicator_count = 0
    trade_count = 0
    symbol_interval_count = 0
    
    try:
        # Get count of records in each table
        cursor.execute("SELECT COUNT(*) FROM historical_data")
        price_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM technical_indicators")
        indicator_count = cursor.fetchone()[0]
        
        try:
            cursor.execute("SELECT COUNT(*) FROM trades")
            trade_count = cursor.fetchone()[0]
        except:
            # Trades table might not exist yet
            trade_count = 0
        
        # Get count of unique symbol/interval combinations
        cursor.execute("SELECT COUNT(*) FROM (SELECT DISTINCT symbol, interval FROM historical_data) AS temp")
        symbol_interval_count = cursor.fetchone()[0]
    except Exception as e:
        print(f"Error getting database stats: {e}")
    finally:
        cursor.close()
        conn.close()
    
    return {
        "price_count": price_count,
        "indicator_count": indicator_count,
        "trade_count": trade_count,
        "symbol_interval_count": symbol_interval_count
    }

def run_background_backfill(continuous=True, full=False, interval_minutes=15):
    """
    Run a background thread to backfill database without blocking the UI
    
    Args:
        continuous: Whether to run continuously in background or just once
        full: Whether to do a full backfill (more symbols, longer history)
        interval_minutes: How often to update in continuous mode (minutes)
    """
    # Create a file lock to prevent multiple instances
    import os, time
    lock_file = ".backfill_lock"
    
    # Check if a backfill process is already running
    if os.path.exists(lock_file):
        try:
            # Check if the lock file is stale (older than 30 minutes)
            lock_age = time.time() - os.path.getmtime(lock_file)
            if lock_age < 1800:  # 30 minutes in seconds
                print("A backfill process is already running. Skipping.")
                return None
            else:
                # Lock file is stale, remove it
                os.remove(lock_file)
                print("Removed stale lock file.")
        except Exception as e:
            print(f"Error checking lock file: {e}")
            return None
    
    # Create lock file
    try:
        with open(lock_file, 'w') as f:
            f.write(str(time.time()))
    except Exception as e:
        print(f"Error creating lock file: {e}")
        return None
        
    def backfill_thread():
        try:
            if continuous:
                # Run the continuous update function with specified interval
                args = ["python", "backfill_database.py", "--continuous", f"--interval={interval_minutes}"]
                if full:
                    args.append("--full")
                subprocess.run(args)
            else:
                # Just run a single backfill
                args = ["python", "backfill_database.py"]
                if full:
                    args.append("--full")
                subprocess.run(args)
        finally:
            # Always remove lock file when done
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                except Exception as e:
                    print(f"Error removing lock file: {e}")
    
    # Start backfill in separate thread
    thread = threading.Thread(target=backfill_thread)
    thread.daemon = True  # Thread will exit when main program exits
    thread.start()
    return thread

# Function to fetch data from Binance API or Database
def get_data(symbol, interval, lookback_days, start_date=None, end_date=None):
    """
    Fetch data for specified symbol and interval
    
    Args:
        symbol: Trading pair symbol (e.g., BTCUSDT)
        interval: Time interval for candles
        lookback_days: Default number of days to look back
        start_date: Optional specific start date (overrides lookback_days)
        end_date: Optional specific end date (defaults to now)
    """
    # Calculate date range if not explicitly provided
    if end_date is None:
        end_time = datetime.now()
    else:
        end_time = end_date
        
    if start_date is None:
        start_time = end_time - timedelta(days=lookback_days)
    else:
        start_time = start_date
    
    # Try to get data from database first
    df = get_historical_data(symbol, interval, start_time, end_time)
    
    # If not enough data in database, fetch from API
    if df.empty or len(df) < 100:
        df = get_klines_data(symbol, interval, start_time, end_time)
        if not df.empty:
            save_historical_data(df, symbol, interval)
    
    return df

# Cache wrapper for get_data function
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_data(symbol, interval, lookback_days, start_date=None, end_date=None):
    """Cached version of get_data to improve performance"""
    return get_data(symbol, interval, lookback_days, start_date, end_date)

def main():
    st.title("Cryptocurrency Trading Analysis Platform")
    
    # Create a sidebar tab selector
    tab_options = ["Analysis", "Portfolio", "Sentiment", "News Digest", "Trend Visualizer"]
    selected_tab = st.sidebar.radio("Navigation", tab_options)
    
    # Initialize session state for backfill tracking
    if 'backfill_started' not in st.session_state:
        st.session_state.backfill_started = False
        st.session_state.backfill_thread = None
    
    # Start background backfill automatically on first load if needed
    # Add a timestamp to track when the backfill was started
    if 'backfill_start_time' not in st.session_state:
        st.session_state.backfill_start_time = datetime.now()
    
    # Only start a new backfill process if one hasn't been started in the last 5 minutes
    current_time = datetime.now()
    time_diff = (current_time - st.session_state.backfill_start_time).total_seconds()
    
    # If it's the first time or if it's been more than 5 minutes since last attempt
    if not st.session_state.backfill_started or time_diff > 300:
        # Kill any existing backfill processes before starting a new one
        if st.session_state.backfill_started:
            try:
                # Find and kill any running backfill_database.py processes
                subprocess.run(["pkill", "-f", "backfill_database.py"], 
                              stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                time.sleep(1)  # Give it a second to clean up
            except Exception:
                pass  # Ignore errors with killing process
        
        # Start a new continuous background update with fixed 15 minute interval
        st.session_state.backfill_thread = run_background_backfill(
            continuous=True,  # Run continuously
            full=False,       # Start with quick update
            interval_minutes=15  # Always update every 15 minutes - do not change
        )
        st.session_state.backfill_started = True
        st.session_state.backfill_start_time = current_time
        st.info("Continuous database updates started in the background. The database will be regularly updated with the latest prices, indicators, and trading signals.")
    
    # Show database status in expandable section
    with st.expander("Database Status"):
        col1, col2, col3, col4 = st.columns(4)
        
        # Get database statistics
        stats = get_database_stats()
        
        # Display database population metrics
        col1.metric("Symbols/Intervals", stats["symbol_interval_count"])
        col2.metric("Price Records", f"{stats['price_count']:,}")
        col3.metric("Indicator Records", f"{stats['indicator_count']:,}")
        col4.metric("Trade Signals", f"{stats['trade_count']:,}")
        
        # Show last update time
        last_update = get_last_update_time()
        if last_update:
            st.info(f"Database last updated: {last_update.strftime('%Y-%m-%d %H:%M:%S')} GMT")
        else:
            st.warning("Database not yet populated. Initial backfill in progress...")
            
    # Display based on selected tab
    if selected_tab == "Analysis":
        # Sidebar for controls
        st.sidebar.header("Settings")
        
        # Get available symbols from Binance
        available_symbols = get_available_symbols()
        default_symbol = "BTCUSDT"
        if default_symbol not in available_symbols and available_symbols:
            default_symbol = available_symbols[0]
        
        symbol = st.sidebar.selectbox(
            "Select Cryptocurrency Pair",
            options=available_symbols,
            index=available_symbols.index(default_symbol) if default_symbol in available_symbols else 0
        )
        
        # Timeframe selection
    timeframe_options = get_timeframe_options()
    interval = st.sidebar.selectbox(
        "Select Timeframe",
        options=list(timeframe_options.keys()),
        index=5  # Default to 4h
    )
    binance_interval = timeframe_to_interval(interval)
    
    # Lookback period
    lookback_days = st.sidebar.slider(
        "Historical Data (days)",
        min_value=1,
        max_value=1095,  # 3 years
        value=30
    )
    
    # Indicators selection
    st.sidebar.header("Technical Indicators")
    show_bollinger = st.sidebar.checkbox("Bollinger Bands", value=True)
    show_macd = st.sidebar.checkbox("MACD", value=True)
    show_rsi = st.sidebar.checkbox("RSI", value=True)
    show_ema = st.sidebar.checkbox("EMA", value=True)
    
    # Auto-refresh option
    st.sidebar.header("Refresh Settings")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider(
        "Refresh Interval (seconds)",
        min_value=5,
        max_value=300,
        value=30
    )
    
    # Strategy settings
    st.sidebar.header("Strategy Settings")
    bb_threshold = st.sidebar.slider("Bollinger Band Threshold", 0.0, 1.0, 0.8)
    rsi_oversold = st.sidebar.slider("RSI Oversold Threshold", 10, 40, 30)
    rsi_overbought = st.sidebar.slider("RSI Overbought Threshold", 60, 90, 70)
    
    # Add date range selection for advanced users
    st.sidebar.header("Date Range")
    use_custom_dates = st.sidebar.checkbox("Use Custom Date Range", value=False)
    
    start_date = None
    end_date = None
    
    if use_custom_dates:
        # Calculate default dates
        default_end_date = datetime.now()
        default_start_date = default_end_date - timedelta(days=lookback_days)
        
        # Date range inputs
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=default_start_date, 
                                       max_value=datetime.now())
        with col2:
            end_date = st.date_input("End Date", value=default_end_date, 
                                     max_value=datetime.now())
        
        # Convert to datetime with time
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.max.time())
        
        # Ensure end date is not before start date
        if end_date < start_date:
            st.error("End date cannot be before start date")
            return
    
    # Load More Data button for when scrolling
    if st.sidebar.button("Load More Historical Data"):
        # If already using custom dates, extend them further back
        if use_custom_dates and start_date:
            # Extend lookback by 3x
            extended_start_date = start_date - timedelta(days=lookback_days*2)
            start_date = extended_start_date
        else:
            # Use 3x the normal lookback
            lookback_days = lookback_days * 3
        
        st.sidebar.success(f"Loading {lookback_days} days of historical data...")
    
    # Backfill Database button
    backfill_col1, backfill_col2 = st.sidebar.columns(2)
    
    # Quick update button
    if backfill_col1.button("Quick Update"):
        st.sidebar.info("Running quick update for recent data...")
        try:
            # Ensure subprocess module is used correctly
            process = subprocess.Popen(["python", "backfill_database.py"], 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE,
                                     text=True)
            st.sidebar.success("Quick update started in the background. Check the server logs for progress.")
        except Exception as e:
            st.sidebar.error(f"Error starting backfill process: {e}")
    
    # Full backfill button
    if backfill_col2.button("Full Backfill"):
        st.sidebar.info("Running full backfill to calculate and store optimized trading strategies for all available symbols...")
        try:
            # Ensure subprocess module is used correctly
            process = subprocess.Popen(["python", "backfill_database.py", "--full"], 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE,
                                     text=True)
            st.sidebar.success("Full backfill process started in the background. This may take several minutes to complete.")
        except Exception as e:
            st.sidebar.error(f"Error starting backfill process: {e}")
            
    # This code should only execute if we're in the Analysis tab
    if selected_tab == "Analysis":
        # Get data from API or database
        try:
            # Show last update time for this specific symbol/interval
            last_update_for_interval = get_last_update_time(symbol, binance_interval)
            if last_update_for_interval:
                st.info(f"Latest data for {symbol} ({interval}) from: {last_update_for_interval.strftime('%Y-%m-%d %H:%M:%S')} GMT")
            
            with st.spinner(f"Fetching {symbol} data..."):
                # Use the cached data function with appropriate parameters
                if use_custom_dates:
                    df = get_cached_data(symbol, binance_interval, lookback_days, start_date, end_date)
                    
                    # Format dates nicely for display
                    try:
                        start_date_str = start_date.strftime('%Y-%m-%d')
                        end_date_str = end_date.strftime('%Y-%m-%d')
                    except (AttributeError, TypeError):
                        # If dates aren't datetime objects, convert them
                        start_date_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
                        end_date_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')
                    
                    st.sidebar.info(f"Showing data from {start_date_str} to {end_date_str}")
                else:
                    df = get_cached_data(symbol, binance_interval, lookback_days)
                    
                    # Provide feedback about the date range being shown
                    if not df.empty:
                        try:
                            actual_start = df['timestamp'].min()
                            actual_end = df['timestamp'].max()
                            
                            # Make sure timestamp is a datetime object
                            if isinstance(actual_start, (str, int, float)):
                                actual_start = pd.to_datetime(actual_start)
                                
                            if isinstance(actual_end, (str, int, float)):
                                actual_end = pd.to_datetime(actual_end)
                                
                            start_date_str = actual_start.strftime('%Y-%m-%d')
                            end_date_str = actual_end.strftime('%Y-%m-%d')
                            
                            date_range_text = f"Showing data from {start_date_str} to {end_date_str}"
                            st.sidebar.info(date_range_text)
                        except (AttributeError, TypeError):
                            # If there's any issue with date formatting, just show a simpler message
                            st.sidebar.info(f"Loaded {len(df)} data points")
                
                if df.empty:
                    st.error("No data available for the selected pair and timeframe.")
                    return
                
                # Display the number of candles loaded
                st.sidebar.success(f"Loaded {len(df)} candles")
            
            # Calculate indicators
            try:
                # Import indicator functions again for proper scope
                from indicators import add_bollinger_bands, add_macd, add_rsi, add_ema
                
                if show_bollinger:
                    df = add_bollinger_bands(df)
                if show_macd:
                    df = add_macd(df)
                if show_rsi:
                    df = add_rsi(df)
                if show_ema:
                    df = add_ema(df, 9)
                    df = add_ema(df, 21)
                    df = add_ema(df, 50)
                    df = add_ema(df, 200)
            except Exception as e:
                st.error(f"Error calculating indicators: {str(e)}")
                import traceback
                st.text(traceback.format_exc())
                # Continue with available data
            
            # Evaluate buy/sell signals
            df = evaluate_buy_sell_signals(df, bb_threshold, rsi_oversold, rsi_overbought)
        except Exception as e:
            st.error(f"An error occurred in the Analysis tab: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
            return
    
        # Create main chart with candlesticks
        # Determine how many subplots we need based on selected indicators
        num_rows = 1  # Always have the main price chart
        
        # Add row for volume
        num_rows += 1
        
        # Add row for MACD if selected
        if show_macd:
            num_rows += 1
            
        # Add row for RSI if selected
        if show_rsi:
            num_rows += 1
        
        # Calculate row heights - give more space to price chart
        row_heights = [0.5]  # Main price chart gets 50%
        remaining_height = 0.5  # Remaining 50% divided among other indicators
        
        # Distribute remaining height among other rows
        for i in range(1, num_rows):
            row_heights.append(remaining_height / (num_rows - 1))
        
        # Create subplot titles
        subplot_titles = [f"{symbol} - {interval}"]
        subplot_titles.append("Volume")
        
        if show_macd:
            subplot_titles.append("MACD")
        
        if show_rsi:
            subplot_titles.append("RSI")
        
        fig = make_subplots(
            rows=num_rows, 
            cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            row_heights=row_heights,
            subplot_titles=subplot_titles
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Add volume bars
        colors = ['red' if row['open'] > row['close'] else 'green' for _, row in df.iterrows()]
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['volume'],
                marker_color=colors,
                name="Volume"
            ),
            row=2, col=1
        )
        
        # Add Bollinger Bands
        if show_bollinger and 'bb_upper' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['bb_upper'],
                    line=dict(color='rgba(173, 204, 255, 0.7)'),
                    name="BB Upper"
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['bb_middle'],
                    line=dict(color='rgba(173, 204, 255, 0.7)'),
                    name="BB Middle"
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['bb_lower'],
                    line=dict(color='rgba(173, 204, 255, 0.7)'),
                    fill='tonexty',
                    fillcolor='rgba(173, 204, 255, 0.1)',
                    name="BB Lower"
                ),
                row=1, col=1
            )
            
        # Add EMAs if selected
        if show_ema:
            ema_colors = {
                'ema_9': 'yellow',
                'ema_21': 'purple',
                'ema_50': 'blue',
                'ema_200': 'white'
            }
            
            for col, color in ema_colors.items():
                if col in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=df[col],
                            line=dict(color=color, width=1),
                            name=col.upper()
                        ),
                        row=1, col=1
                    )
        
        # Track which row we're on for indicators
        current_row = 3  # Start after price and volume
        
        # Add MACD to subplot (in its own row if selected)
        if show_macd and 'macd' in df.columns:
            macd_row = current_row
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['macd'],
                    line=dict(color='blue', width=1.5),
                    name="MACD"
                ),
                row=macd_row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['macd_signal'],
                    line=dict(color='red', width=1.5),
                    name="Signal"
                ),
                row=macd_row, col=1
            )
            
            colors = ['green' if val > 0 else 'red' for val in df['macd_histogram']]
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df['macd_histogram'],
                    marker_color=colors,
                    name="Histogram"
                ),
                row=macd_row, col=1
            )
            
            # Move to next row
            current_row += 1
        
        # RSI indicator (if selected)
        if show_rsi and 'rsi' in df.columns:
            # Always gets its own subplot
            rsi_row = current_row if show_macd else 3
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['rsi'],
                    line=dict(color='purple', width=1.5),
                    name="RSI"
                ),
                row=rsi_row, col=1
            )
            
            # Add overbought/oversold lines
            fig.add_trace(
                go.Scatter(
                    x=[df['timestamp'].iloc[0], df['timestamp'].iloc[-1]],
                    y=[rsi_overbought, rsi_overbought],
                    line=dict(color='red', width=1, dash='dash'),
                    name=f"Overbought ({rsi_overbought})"
                ),
                row=rsi_row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[df['timestamp'].iloc[0], df['timestamp'].iloc[-1]],
                    y=[rsi_oversold, rsi_oversold],
                    line=dict(color='green', width=1, dash='dash'),
                    name=f"Oversold ({rsi_oversold})"
                ),
                row=rsi_row, col=1
            )
        
        # Add buy/sell markers
        buy_signals = df[df['buy_signal']]
        sell_signals = df[df['sell_signal']]
        
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals['timestamp'],
                    y=buy_signals['low'] * 0.995,  # Place just below the candle
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green',
                        line=dict(width=2, color='darkgreen')
                    ),
                    name="Buy Signal"
                ),
                row=1, col=1
            )
        
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals['timestamp'],
                    y=sell_signals['high'] * 1.005,  # Place just above the candle
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    name="Sell Signal"
                ),
                row=1, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} Analysis",
            xaxis_rangeslider_visible=False,
            height=800,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=50, r=50, t=85, b=100),
            # Enable scrolling and zooming
            dragmode='zoom',  # Default drag mode is zoom
            hovermode='closest',
            # Allow more interaction options
            modebar=dict(
                orientation='v',
                bgcolor='rgba(0,0,0,0.2)',
                color='white',
                activecolor='rgba(0,191,255,0.7)'
            ),
            # Add zoom and pan buttons
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(
                            args=[{"dragmode": "zoom"}, {"annotations": {}}],
                            label="Zoom",
                            method="relayout"
                        ),
                        dict(
                            args=[{"dragmode": "pan"}, {"annotations": {}}],
                            label="Pan",
                            method="relayout"
                        )
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.05,
                    xanchor="left",
                    y=1.08,
                    yanchor="top"
                )
            ]
        )
        
        # Add grid to all axes
        for i in range(1, num_rows + 1):
            # Configure x-axis for each subplot
            fig.update_xaxes(
                row=i, col=1,
                gridcolor="rgba(255, 255, 255, 0.1)",
                showgrid=True,
                # Make the chart more interactive
                rangeslider_visible=False,
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikecolor="rgba(255, 255, 255, 0.5)",
                spikethickness=1
            )
            
            # Configure y-axis for each subplot
            fig.update_yaxes(
                row=i, col=1,
                gridcolor="rgba(255, 255, 255, 0.1)",
                showgrid=True,
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikecolor="rgba(255, 255, 255, 0.5)",
                spikethickness=1,
                # Disable abbreviated numbers like 'k' for thousands
                exponentformat='none',
                separatethousands=True
            )
        
        # Add dynamic events for chart relayouts (zooming/panning)
        fig.update_layout(
            # Add event listeners for range changes
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(
                            args=[{"dragmode": "zoom"}, {"annotations": {}}],
                            label="Zoom",
                            method="relayout"
                        ),
                        dict(
                            args=[{"dragmode": "pan"}, {"annotations": {}}],
                            label="Pan",
                            method="relayout"
                        ),
                        dict(
                            args=[{
                                "xaxis.autorange": True, 
                                "yaxis.autorange": True,
                                "xaxis.range": None,
                                "yaxis.range": None
                            }],
                            label="Reset Zoom",
                            method="relayout"
                        )
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.05,
                    xanchor="left",
                    y=1.08,
                    yanchor="top"
                )
            ]
        )
        
        # Add a message about data loading
        st.info("📈 **Chart Navigation**: Use the mouse wheel to zoom, drag to pan. To load more historical data, use the 'Load More Historical Data' button in the sidebar.")
        
        # Show chart
        chart_container = st.container()
        with chart_container:
            chart = st.plotly_chart(fig, use_container_width=True, config={
                'scrollZoom': True,  # Enable scroll to zoom
                'displayModeBar': True,  # Always show the mode bar
                'modeBarButtonsToAdd': ['select2d', 'lasso2d', 'resetScale2d', 'autoScale2d'],
                'modeBarButtonsToRemove': ['toggleSpikelines']
            })
            
        # Display current signals
        col1, col2, col3 = st.columns(3)
        
        # Get latest data point
        latest_data = df.iloc[-1]
        current_price = latest_data['close']
        
        with col1:
            st.metric(
                label=f"Current {symbol} Price",
                value=f"${current_price:.2f}",
                delta=f"{(current_price - df.iloc[-2]['close']):.2f}"
            )
        
        with col2:
            # Display buy signal if present
            if latest_data['buy_signal']:
                st.markdown(
                    "<div style='background-color: rgba(0, 255, 0, 0.3); padding: 20px; border-radius: 5px; text-align: center;'>"
                    "<h2>BUY SIGNAL</h2>"
                    f"<p>Price: ${current_price:.2f}</p>"
                    "</div>",
                    unsafe_allow_html=True
                )
        
        with col3:
            # Display sell signal if present
            if latest_data['sell_signal']:
                st.markdown(
                    "<div style='background-color: rgba(255, 0, 0, 0.3); padding: 20px; border-radius: 5px; text-align: center;'>"
                    "<h2>SELL SIGNAL</h2>"
                    f"<p>Price: ${current_price:.2f}</p>"
                    "</div>",
                    unsafe_allow_html=True
                )
        
        # Recent signals table
        st.subheader("Recent Signals")
        recent_buy_signals = buy_signals.iloc[-5:] if not buy_signals.empty else pd.DataFrame()
        recent_sell_signals = sell_signals.iloc[-5:] if not sell_signals.empty else pd.DataFrame()
        
        if not recent_buy_signals.empty or not recent_sell_signals.empty:
            signal_data = []
            
            for _, row in recent_buy_signals.iterrows():
                signal_data.append({
                    "Time": row['timestamp'],
                    "Type": "BUY",
                    "Price": f"${row['close']:.2f}",
                    "RSI": f"{row.get('rsi', 'N/A'):.1f}" if 'rsi' in row else 'N/A',
                    "MACD": f"{row.get('macd', 'N/A'):.6f}" if 'macd' in row else 'N/A'
                })
                
            for _, row in recent_sell_signals.iterrows():
                signal_data.append({
                    "Time": row['timestamp'],
                    "Type": "SELL",
                    "Price": f"${row['close']:.2f}",
                    "RSI": f"{row.get('rsi', 'N/A'):.1f}" if 'rsi' in row else 'N/A',
                    "MACD": f"{row.get('macd', 'N/A'):.6f}" if 'macd' in row else 'N/A'
                })
            
            # Sort signals by time (descending)
            signal_data = sorted(signal_data, key=lambda x: x["Time"], reverse=True)
            
            # Create DataFrame for display
            signal_df = pd.DataFrame(signal_data)
            st.dataframe(signal_df, use_container_width=True)
        else:
            st.info("No recent signals detected.")
        
        # Strategy Performance - Advanced Backtesting
        st.subheader("Strategy Performance")

        # Perform advanced backtesting using our strategy module
        with st.spinner("Running strategy backtesting..."):
            # Advanced backtest using the strategy module
            backtest_results = backtest_strategy(df)
            
            if backtest_results and backtest_results['metrics']['num_trades'] > 0:
                metrics = backtest_results['metrics']
                trades = backtest_results['trades']
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Trades", metrics['num_trades'])
                
                with col2:
                    st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                
                with col3:
                    st.metric("Avg Return per Trade", f"{metrics['avg_profit_pct']:.2f}%")
                
                with col4:
                    st.metric("Total Return", f"{metrics['total_return_pct']:.2f}%")
                
                # Recent trade history
                if len(trades) > 0:
                    sell_trades = [t for t in trades if t['type'] in ['SELL', 'SELL (EOT)']]
                    
                    if sell_trades:
                        st.subheader("Recent Trade History")
                        trade_history = []
                        
                        # Get only the most recent trades (up to 10)
                        recent_trades = sell_trades[-10:] if len(sell_trades) > 10 else sell_trades
                        
                        for trade in recent_trades:
                            # Find the corresponding buy trade
                            buy_price = 0
                            for t in trades:
                                if t['type'] == 'BUY' and t['timestamp'] < trade['timestamp']:
                                    buy_price = t['price']
                                    break
                            
                            trade_history.append({
                                "Entry Time": trade.get('entry_time', 'Unknown'),
                                "Exit Time": trade['timestamp'],
                                "Entry Price": f"${buy_price:.2f}",
                                "Exit Price": f"${trade['price']:.2f}",
                                "Return (%)": f"{trade.get('profit_pct', 0):.2f}%",
                                "Holding Time (hours)": f"{trade.get('holding_time', 0):.1f}",
                                "Result": "Win" if trade.get('profit_pct', 0) > 0 else "Loss"
                            })
                        
                        # Reverse to show most recent first
                        trade_history.reverse()
                        st.dataframe(trade_history, use_container_width=True)
                
                # Show equity curve
                backtest_df = backtest_results['backtest_df']
                if not backtest_df.empty and 'portfolio_value' in backtest_df.columns:
                    st.subheader("Equity Curve")
                    
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=backtest_df['timestamp'],
                            y=backtest_df['portfolio_value'],
                            mode='lines',
                            name="Portfolio Value",
                            line=dict(color='green', width=2)
                        )
                    )
                    
                    fig.update_layout(
                        title="Portfolio Performance Over Time",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        template="plotly_dark",
                        height=400,
                        margin=dict(l=50, r=50, t=50, b=50),
                        yaxis=dict(
                            exponentformat='none',
                            separatethousands=True
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.info("No completed trades to analyze. The strategy didn't generate any signals during this period or there wasn't enough data for backtesting.")
        
        # Strategy Optimization - Find the most profitable parameters
        # Moved outside of the if-block to always be available
        st.subheader("Strategy Optimization")
        
        with st.expander("Optimize Trading Strategy", expanded=True):
            # Create a container for the optimization controls
            optimization_container = st.container()
            
            # Use this container to show optimization results
            results_container = st.container()
            
            with optimization_container:
                # Add some explanation text
                st.write("""
                The optimization process will test multiple combinations of parameters to find the most profitable trading strategy.
                This process may take some time depending on the amount of data and parameter combinations.
                """)
                
                # Allow user to select which strategies to optimize
                st.write("### Strategy Selection")
                
                # Strategy selection
                strategy_cols = st.columns(3)
                
                # Column 1 - Bollinger Bands
                with strategy_cols[0]:
                    st.subheader("Bollinger Bands")
                    use_bb = st.checkbox("Enable Bollinger Bands", value=True)
                    
                    # Only show parameters if enabled
                    if use_bb:
                        bb_min = st.slider("BB Threshold Min", 0.0, 0.5, 0.1, 0.1)
                        bb_max = st.slider("BB Threshold Max", 0.1, 0.9, 0.5, 0.1)
                        bb_step = 0.1
                        bb_values = [round(x, 1) for x in np.arange(bb_min, bb_max + bb_step, bb_step)]
                        
                        # Bollinger Band window size
                        bb_window_options = [10, 20, 30, 40, 50]
                        bb_window = st.selectbox("BB Window Size", options=bb_window_options, index=1)
                    else:
                        # Default values if disabled
                        bb_values = [0.2]  # Use a neutral value
                        bb_window = 20
                            
                # Column 2 - RSI
                with strategy_cols[1]:
                    st.subheader("RSI")
                    use_rsi = st.checkbox("Enable RSI", value=True)
                    
                    # Only show parameters if enabled
                    if use_rsi:
                        rsi_o_min = st.slider("RSI Oversold Min", 10, 40, 20, 5)
                        rsi_o_max = st.slider("RSI Oversold Max", 20, 45, 35, 5)
                        rsi_o_step = 5
                        rsi_o_values = list(range(rsi_o_min, rsi_o_max + rsi_o_step, rsi_o_step))
                        
                        rsi_ob_min = st.slider("RSI Overbought Min", 55, 85, 65, 5)
                        rsi_ob_max = st.slider("RSI Overbought Max", 65, 90, 85, 5)
                        rsi_ob_step = 5
                        rsi_ob_values = list(range(rsi_ob_min, rsi_ob_max + rsi_ob_step, rsi_ob_step))
                        
                        # RSI window size
                        rsi_window_options = [7, 14, 21, 28]
                        rsi_window = st.selectbox("RSI Window", options=rsi_window_options, index=1)
                    else:
                        # Default values if disabled
                        rsi_o_values = [30]  # Default values that won't affect the strategy
                        rsi_ob_values = [70]
                        rsi_window = 14
                            
                # Column 3 - MACD
                with strategy_cols[2]:
                    st.subheader("MACD")
                    use_macd = st.checkbox("Enable MACD", value=True)
                    
                    # Only show parameters if enabled
                    if use_macd:
                        # MACD strategy type
                        macd_strategies = ['MACD Crossover', 'MACD Histogram', 'Both']
                        macd_strategy_type = st.radio("MACD Strategy Type", options=macd_strategies)
                        
                        if macd_strategy_type == 'Both':
                            selected_strategies = [True, False]
                        elif macd_strategy_type == 'MACD Crossover':
                            selected_strategies = [True]
                        else:
                            selected_strategies = [False]
                        
                        # MACD parameters
                        col1, col2 = st.columns(2)
                        with col1:
                            macd_fast_options = [8, 12, 16]
                            macd_fast = st.selectbox("MACD Fast Period", options=macd_fast_options, index=1)
                            
                        with col2:
                            macd_slow_options = [21, 26, 30]
                            macd_slow = st.selectbox("MACD Slow Period", options=macd_slow_options, index=1)
                        
                        macd_signal_options = [7, 9, 12]
                        macd_signal = st.selectbox("MACD Signal Period", options=macd_signal_options, index=1)
                    else:
                        # Default values if disabled
                        selected_strategies = [True]  # Default to crossover
                        macd_fast = 12
                        macd_slow = 26
                        macd_signal = 9
                            
                # Calculate combinations based on enabled strategies
                active_strategies = 0
                strategy_combinations = 1
                
                if use_bb:
                    active_strategies += 1
                    strategy_combinations *= len(bb_values)
                    
                if use_rsi:
                    active_strategies += 1
                    strategy_combinations *= len(rsi_o_values) * len(rsi_ob_values)
                    
                if use_macd:
                    active_strategies += 1
                    strategy_combinations *= len(selected_strategies)
                
                st.write("### Active Parameters")
                st.write(f"Strategies enabled: {active_strategies}")
                st.info(f"Total parameter combinations to test: {strategy_combinations}")
                        
                # Run optimization button
                if st.button("Run Optimization"):
                    # Define parameter ranges from user input
                    parameter_ranges = {}
                    
                    if use_bb:
                        parameter_ranges['bb_threshold'] = bb_values
                        
                    if use_rsi:
                        parameter_ranges['rsi_oversold'] = rsi_o_values
                        parameter_ranges['rsi_overbought'] = rsi_ob_values
                        
                    if use_macd:
                        parameter_ranges['use_macd_crossover'] = selected_strategies
                    
                    # Store fixed parameters
                    fixed_params = {
                        'use_bb': use_bb,
                        'use_rsi': use_rsi,
                        'use_macd': use_macd,
                        'bb_window': bb_window,
                        'rsi_window': rsi_window,
                        'macd_fast': macd_fast,
                        'macd_slow': macd_slow,
                        'macd_signal': macd_signal
                    }
                    
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Run optimization with progress updates
                    status_text.text("Starting optimization process...")
                    
                    # Create a placeholder for intermediate results
                    interim_results = st.empty()
                    
                    # Track number of combinations tested
                    combinations_tested = 0
                    
                    # Rather than passing the full optimization to find_optimal_strategy,
                    # we'll run each combination individually to show progress
                    results = []
                    base_df = df.copy()
                    
                    # Make sure all needed indicators are calculated
                    # Make sure we're accessing the indicator functions properly
                    # These are already imported at the top of the file
                    try:
                        if 'bb_lower' not in base_df.columns:
                            st.text("Adding Bollinger Bands...")
                            from indicators import add_bollinger_bands
                            base_df = add_bollinger_bands(base_df)
                        
                        if 'rsi' not in base_df.columns:
                            st.text("Adding RSI...")
                            from indicators import add_rsi
                            base_df = add_rsi(base_df)
                        
                        if 'macd' not in base_df.columns:
                            st.text("Adding MACD...")
                            from indicators import add_macd
                            base_df = add_macd(base_df)
                            
                    except Exception as e:
                        st.error(f"Error calculating indicators: {str(e)}")
                        import traceback
                        st.text(traceback.format_exc())
                            
                    # Create parameter combinations based on enabled strategies
                    param_combinations = []
                    
                    # Start with a base parameter set with single values
                    base_params = {
                        'bb_threshold': parameter_ranges.get('bb_threshold', [0.2])[0],
                        'rsi_oversold': parameter_ranges.get('rsi_oversold', [30])[0],
                        'rsi_overbought': parameter_ranges.get('rsi_overbought', [70])[0],
                        'use_macd_crossover': parameter_ranges.get('use_macd_crossover', [True])[0]
                    }
                            
                    # Function to add all combinations of a parameter to existing combinations
                    def add_parameter_combinations(combinations, param_name, values):
                        if not combinations:
                            return [{param_name: value} for value in values]
                        
                        new_combinations = []
                        for combo in combinations:
                            for value in values:
                                new_combo = combo.copy()
                                new_combo[param_name] = value
                                new_combinations.append(new_combo)
                        return new_combinations
                    
                    # Add enabled parameters to combinations
                    combinations = [{}]
                    
                    if 'bb_threshold' in parameter_ranges and use_bb:
                        combinations = add_parameter_combinations(combinations, 'bb_threshold', parameter_ranges['bb_threshold'])
                        
                    if 'rsi_oversold' in parameter_ranges and 'rsi_overbought' in parameter_ranges and use_rsi:
                        # First add oversold values
                        combinations = add_parameter_combinations(combinations, 'rsi_oversold', parameter_ranges['rsi_oversold'])
                        # Then add overbought values to each combination
                        combinations = add_parameter_combinations(combinations, 'rsi_overbought', parameter_ranges['rsi_overbought'])
                        
                    if 'use_macd_crossover' in parameter_ranges and use_macd:
                        combinations = add_parameter_combinations(combinations, 'use_macd_crossover', parameter_ranges['use_macd_crossover'])
                    
                    # Fill in defaults for any parameter that wasn't added
                    for combo in combinations:
                        for param, default_value in base_params.items():
                            if param not in combo:
                                combo[param] = default_value
                            
                            st.write(f"Testing {len(combinations)} parameter combinations")
                            
                            # Update total combinations for progress tracking
                            total_combinations = len(combinations)
                            
                            # Test each parameter combination
                            for params in combinations:
                                # Update status
                                bb_threshold = float(params.get('bb_threshold', 0.2))
                                rsi_oversold = int(params.get('rsi_oversold', 30))
                                rsi_overbought = int(params.get('rsi_overbought', 70))
                                use_macd_crossover = bool(params.get('use_macd_crossover', True))
                                
                                # Display parameter info
                                param_info = []
                                if use_bb:
                                    param_info.append(f"BB={bb_threshold:.1f}")
                                if use_rsi:
                                    param_info.append(f"RSI={rsi_oversold}/{rsi_overbought}")
                                if use_macd:
                                    macd_type = "Crossover" if use_macd_crossover else "Histogram"
                                    param_info.append(f"MACD={macd_type}")
                                
                                status_text.text(f"Testing parameters: {', '.join(param_info)}")
                                
                                # Generate signals and backtest
                                signals_df = evaluate_buy_sell_signals(
                                    base_df, 
                                    bb_threshold=bb_threshold, 
                                    rsi_oversold=rsi_oversold, 
                                    rsi_overbought=rsi_overbought,
                                    use_macd_crossover=use_macd_crossover,
                                    use_bb=fixed_params['use_bb'],
                                    use_rsi=fixed_params['use_rsi'],
                                    use_macd=fixed_params['use_macd'],
                                    bb_window=fixed_params['bb_window'],
                                    rsi_window=fixed_params['rsi_window'],
                                    macd_fast=fixed_params['macd_fast'],
                                    macd_slow=fixed_params['macd_slow'],
                                    macd_signal=fixed_params['macd_signal']
                                )
                                backtest_result = backtest_strategy(signals_df)
                                
                                if backtest_result and backtest_result['metrics']['num_trades'] > 0:
                                    results.append({
                                        'parameters': params,
                                        'metrics': backtest_result['metrics'],
                                        'trades': backtest_result['trades']
                                    })
                                    
                                    # Show interim best result
                                    if results:
                                        best_so_far = sorted(results, key=lambda x: x['metrics']['total_return_pct'], reverse=True)[0]
                                        
                                        # Format output based on active strategies
                                        best_param_details = []
                                        if use_bb:
                                            best_param_details.append(f"BB={best_so_far['parameters']['bb_threshold']:.2f}")
                                        if use_rsi:
                                            best_param_details.append(f"RSI Oversold={best_so_far['parameters']['rsi_oversold']}")
                                            best_param_details.append(f"RSI Overbought={best_so_far['parameters']['rsi_overbought']}")
                                        if use_macd:
                                            macd_type = "Crossover" if best_so_far['parameters'].get('use_macd_crossover', True) else "Histogram"
                                            best_param_details.append(f"MACD={macd_type}")
                                        
                                        best_param_details.append(f"Return: {best_so_far['metrics']['total_return_pct']:.2f}%")
                                        interim_results.info(f"Best strategy so far: {', '.join(best_param_details)}")
                                
                                # Update progress
                                combinations_tested += 1
                                progress_percentage = min(1.0, combinations_tested / len(combinations))
                                progress_bar.progress(progress_percentage)
                            
                            # Clear status
                            status_text.empty()
                            progress_bar.empty()
                            interim_results.empty()
                            
                            # Find best result
                            optimization_results = None
                            if results:
                                # Sort by total return
                                results.sort(key=lambda x: x['metrics']['total_return_pct'], reverse=True)
                                optimization_results = results[0]
                                st.success("Optimization completed!")
                                
                                # Display optimal parameters
                                st.write("### Optimal Strategy Parameters")
                                
                                params = optimization_results['parameters']
                                opt_metrics = optimization_results['metrics']
                                
                                # Determine how many active strategies we have
                                active_strategy_count = sum([use_bb, use_rsi, use_macd])
                                
                                # Create columns based on active strategies
                                param_cols = st.columns(active_strategy_count)
                                
                                # Counter for selecting the right column
                                col_idx = 0
                                
                                # Add metrics for each active strategy
                                if use_bb:
                                    with param_cols[col_idx]:
                                        st.metric("Bollinger Band Threshold", f"{params['bb_threshold']:.2f}")
                                    col_idx += 1
                                
                                if use_rsi:
                                    with param_cols[col_idx]:
                                        st.metric("RSI Levels", f"{params['rsi_oversold']}/{params['rsi_overbought']}")
                                    col_idx += 1
                                
                                if use_macd:
                                    with param_cols[col_idx]:
                                        macd_strategy = "MACD Crossover" if params.get('use_macd_crossover', True) else "MACD Histogram"
                                        st.metric("MACD Strategy", macd_strategy)
                                
                                # Performance metrics for optimal strategy
                                st.write("### Optimal Strategy Performance")
                                
                                m1, m2, m3, m4 = st.columns(4)
                                with m1:
                                    st.metric("Total Return", f"{opt_metrics['total_return_pct']:.2f}%")
                                with m2:
                                    st.metric("Number of Trades", opt_metrics['num_trades'])
                                with m3:
                                    st.metric("Win Rate", f"{opt_metrics['win_rate']:.1f}%")
                                with m4:
                                    st.metric("Avg Profit per Trade", f"{opt_metrics['avg_profit_pct']:.2f}%")
                                
                                # Display optimal trades
                                if 'trades' in optimization_results and len(optimization_results['trades']) > 0:
                                    st.write("### Profitable Trading Opportunities")
                                    
                                    # Filter to show only winning trades
                                    sell_trades = [t for t in optimization_results['trades'] 
                                                 if t['type'] in ['SELL', 'SELL (EOT)'] and t.get('profit_pct', 0) > 0]
                                    
                                    if sell_trades:
                                        # Sort by profitability
                                        sell_trades.sort(key=lambda x: x.get('profit_pct', 0), reverse=True)
                                        
                                        # Take top 10 most profitable trades
                                        top_trades = sell_trades[:10]
                                        
                                        trade_data = []
                                        for trade in top_trades:
                                            trade_data.append({
                                                "Exit Date": trade['timestamp'],
                                                "Price": f"${trade['price']:.2f}",
                                                "Profit (%)": f"{trade.get('profit_pct', 0):.2f}%",
                                                "Holding Period (hrs)": f"{trade.get('holding_time', 0):.1f}"
                                            })
                                        
                                        st.dataframe(trade_data, use_container_width=True)
                                    else:
                                        st.info("No profitable trades found in the optimal strategy.")
                                        
                                    # Plot the buy/sell points of the optimal strategy
                                    st.write("### Optimal Strategy Buy/Sell Points")
                                    
                                    # Create a new figure for the optimal strategy visualization
                                    fig = go.Figure()
                                    
                                    # Add candlestick chart
                                    fig.add_trace(
                                        go.Candlestick(
                                            x=df['timestamp'],
                                            open=df['open'],
                                            high=df['high'],
                                            low=df['low'],
                                            close=df['close'],
                                            name="Price"
                                        )
                                    )
                                    
                                    # Add buy points
                                    buy_trades = [t for t in optimization_results['trades'] if t['type'] == 'BUY']
                                    if buy_trades:
                                        buy_times = [t['timestamp'] for t in buy_trades]
                                        buy_prices = [t['price'] for t in buy_trades]
                                        
                                        fig.add_trace(
                                            go.Scatter(
                                                x=buy_times,
                                                y=buy_prices,
                                                mode='markers',
                                                marker=dict(
                                                    symbol='triangle-up',
                                                    size=15,
                                                    color='green',
                                                    line=dict(width=2, color='darkgreen')
                                                ),
                                                name="Optimal Buy"
                                            )
                                        )
                                    
                                    # Add sell points
                                    if sell_trades:
                                        sell_times = [t['timestamp'] for t in sell_trades]
                                        sell_prices = [t['price'] for t in sell_trades]
                                        
                                        fig.add_trace(
                                            go.Scatter(
                                                x=sell_times,
                                                y=sell_prices,
                                                mode='markers',
                                                marker=dict(
                                                    symbol='triangle-down',
                                                    size=15,
                                                    color='red',
                                                    line=dict(width=2, color='darkred')
                                                ),
                                                name="Optimal Sell"
                                            )
                                        )
                                    
                                    fig.update_layout(
                                        title="Optimal Strategy Entry/Exit Points",
                                        xaxis_title="Date",
                                        yaxis_title="Price ($)",
                                        template="plotly_dark",
                                        height=500,
                                        xaxis_rangeslider_visible=False,
                                        margin=dict(l=50, r=50, t=50, b=50),
                                        yaxis=dict(
                                            exponentformat='none',
                                            separatethousands=True
                                        )
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("Could not find an optimal strategy. Try adjusting parameter ranges or using a different dataset.")
                else:
                    st.info("No completed trades to analyze. The strategy didn't generate any signals during this period or there wasn't enough data for backtesting.")
        
        # Auto-refresh logic
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()

    # News Digest Tab
    elif selected_tab == "News Digest":
        try:
            from crypto_news import generate_personalized_digest, get_crypto_news
            import os
            
            st.header("AI-Curated Crypto News Digest")
            
            # Check for OpenAI API key for enhanced features
            OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
            if not OPENAI_API_KEY:
                st.warning("⚠️ For enhanced AI summarization and personalized recommendations, please add your OpenAI API key in the secrets.")
                st.info("The news digest will work with basic functionality without an API key.")
            
            # Sidebar for controls
            st.sidebar.header("News Settings")
            
            # Get portfolio data for personalization
            portfolio_df = database.get_portfolio()
            portfolio_symbols = list(portfolio_df['symbol'].unique()) if not portfolio_df.empty else []
            
            # Allow user to select additional interests beyond portfolio
            available_symbols = get_available_symbols()
            
            # Remove portfolio symbols from available symbols to avoid duplicates
            additional_symbols = [s for s in available_symbols if s not in portfolio_symbols]
            
            # Default selected interests (if no portfolio)
            if not portfolio_symbols:
                default_interests = ["BTCUSDT", "ETHUSDT"]
            else:
                default_interests = []
            
            interests = st.sidebar.multiselect(
                "Additional Interests",
                options=additional_symbols,
                default=default_interests
            )
            
            # Number of articles to display
            articles_count = st.sidebar.slider(
                "Number of Articles",
                min_value=3,
                max_value=20,
                value=8
            )
            
            # Days of news to include
            days_range = st.sidebar.slider(
                "News Timeframe (days)",
                min_value=1,
                max_value=14,
                value=5
            )
            
            # Sources to include (all by default)
            st.sidebar.subheader("News Sources")
            include_coindesk = st.sidebar.checkbox("CoinDesk", value=True)
            include_cryptonews = st.sidebar.checkbox("CryptoNews", value=True)
            include_cointelegraph = st.sidebar.checkbox("CoinTelegraph", value=True)
            
            # Refresh button
            if st.sidebar.button("Refresh News"):
                st.rerun()
            
            # Generate personalized digest
            with st.spinner("Generating your personalized news digest..."):
                # Combine portfolio symbols and additional interests
                all_interests = portfolio_symbols + interests
                
                # Get personalized digest
                digest = generate_personalized_digest(
                    portfolio_symbols=all_interests, 
                    max_articles=articles_count
                )
                
                # Filter articles by selected sources
                selected_sources = []
                if include_coindesk:
                    selected_sources.append("CoinDesk")
                if include_cryptonews:
                    selected_sources.append("CryptoNews")
                if include_cointelegraph:
                    selected_sources.append("CoinTelegraph")
                
                filtered_articles = [a for a in digest['articles'] 
                                    if a['source'] in selected_sources]
                
                # Handle case where no articles match the filter
                if not filtered_articles:
                    st.error("No articles match your selected sources. Please select at least one news source.")
                else:
                    # Replace the articles in the digest with filtered ones
                    digest['articles'] = filtered_articles[:articles_count]
            
            # Display personalized recommendations
            st.subheader("Personalized Recommendations")
            
            # Create three columns for recommendations
            if digest['recommendations']:
                cols = st.columns(min(3, len(digest['recommendations'])))
                
                for i, (col, rec) in enumerate(zip(cols, digest['recommendations'])):
                    with col:
                        # Check if rec is a dictionary with the expected keys
                        if isinstance(rec, dict) and 'title' in rec and 'details' in rec:
                            st.markdown(f"**{rec['title']}**")
                            st.markdown(f"_{rec['details']}_")
                        # Handle case where rec might be a string
                        elif isinstance(rec, str):
                            st.markdown(f"**Recommendation {i+1}**")
                            st.markdown(f"_{rec}_")
                        else:
                            st.markdown(f"**Recommendation {i+1}**")
                            st.markdown("_Unable to display this recommendation._")
            else:
                st.info("No personalized recommendations available.")
            
            # Display news articles
            st.subheader("Latest Cryptocurrency News")
            
            if not digest['articles']:
                st.warning("No news articles found matching your criteria.")
            else:
                # Create tabs for each article
                article_titles = [f"{a['source']}: {a['title'][:40]}..." for a in digest['articles']]
                tabs = st.tabs(article_titles)
                
                for i, (tab, article) in enumerate(zip(tabs, digest['articles'])):
                    with tab:
                        # Format date nicely with GMT/UTC indicator
                        date_str = article['date'].strftime("%Y-%m-%d %H:%M") + " GMT"
                        
                        # Display article metadata
                        st.markdown(f"**{article['title']}**")
                        st.markdown(f"*Source: {article['source']} | Date: {date_str}*")
                        
                        # Display sentiment with appropriate color
                        sentiment = article.get('sentiment', 0)
                        sentiment_color = "green" if sentiment > 0.05 else "red" if sentiment < -0.05 else "gray"
                        st.markdown(f"Sentiment: <span style='color:{sentiment_color}'>{sentiment:.2f}</span>", unsafe_allow_html=True)
                        
                        # Display summary if available, otherwise show content
                        if article.get('summary'):
                            st.markdown("### Summary")
                            st.markdown(article['summary'])
                            
                            # Option to view full article
                            with st.expander("View Full Article"):
                                st.markdown(article['content'])
                        else:
                            st.markdown(article['content'])
                        
                        # Link to original article
                        st.markdown(f"[Read full article on {article['source']}]({article['url']})")
            
            # Display focused news for specific coin (if not in portfolio/interests)
            st.subheader("Explore News by Cryptocurrency")
            
            # All available symbols for direct lookup
            coin_to_explore = st.selectbox(
                "Select Cryptocurrency",
                options=available_symbols,
                index=0
            )
            
            if coin_to_explore:
                coin_name = coin_to_explore.replace('USDT', '')
                
                # Check if this coin is already included in the digest
                if coin_to_explore in all_interests:
                    st.info(f"{coin_name} news is already included in your personalized digest above.")
                
                # Fetch specific news for this coin
                with st.spinner(f"Fetching latest news for {coin_name}..."):
                    coin_news = get_crypto_news(coin_name, max_articles=3, days_back=days_range)
                
                if coin_news:
                    for article in coin_news:
                        with st.expander(f"{article['title']} ({article['source']})"):
                            # Format date nicely with GMT/UTC indicator
                            date_str = article['date'].strftime("%Y-%m-%d %H:%M") + " GMT"
                            
                            # Display article metadata
                            st.markdown(f"*Source: {article['source']} | Date: {date_str}*")
                            
                            # Display sentiment
                            sentiment = article.get('sentiment', 0)
                            sentiment_color = "green" if sentiment > 0.05 else "red" if sentiment < -0.05 else "gray"
                            st.markdown(f"Sentiment: <span style='color:{sentiment_color}'>{sentiment:.2f}</span>", unsafe_allow_html=True)
                            
                            # Display content
                            if article.get('summary'):
                                st.markdown(article['summary'])
                            else:
                                st.markdown(article['content'][:300] + "...")
                            
                            # Link to original article
                            st.markdown(f"[Read full article on {article['source']}]({article['url']})")
                else:
                    st.info(f"No recent news found for {coin_name}.")
        except Exception as e:
            st.error(f"An error occurred in the News Digest tab: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
            
    # Sentiment Tab
    elif selected_tab == "Sentiment":
        try:
            from sentiment_scraper import get_combined_sentiment, get_sentiment_summary
            
            st.header("Cryptocurrency Sentiment Analysis")
            
            # Sidebar for controls
            st.sidebar.header("Sentiment Settings")
            
            # Symbol selection
            available_symbols = get_available_symbols()
            default_symbol = "BTCUSDT"
            if default_symbol not in available_symbols and available_symbols:
                default_symbol = available_symbols[0]
            
            symbol = st.sidebar.selectbox(
                "Select Cryptocurrency",
                options=available_symbols,
                index=available_symbols.index(default_symbol) if default_symbol in available_symbols else 0
            )
            
            # Date range for sentiment
            days_back = st.sidebar.slider(
                "Historical Data (days)",
                min_value=1,
                max_value=30,
                value=7
            )
            
            # Source types to include
            st.sidebar.subheader("Data Sources")
            include_news = st.sidebar.checkbox("News Media", value=True)
            include_social = st.sidebar.checkbox("Social Media", value=True)
            
            # Get sentiment data
            with st.spinner("Fetching sentiment data..."):
                sentiment_data = get_combined_sentiment(symbol, days_back)
                
                # Filter by selected sources
                if not include_news:
                    sentiment_data = sentiment_data[~sentiment_data['source'].isin(["CoinDesk", "CryptoNews", "CoinTelegraph"])]
                if not include_social:
                    sentiment_data = sentiment_data[~sentiment_data['source'].isin(["Twitter", "Reddit"])]
                
                # If we still have data after filtering
                if not sentiment_data.empty:
                    # Get summary metrics
                    summary = get_sentiment_summary(sentiment_data)
                    
                    # Dashboard layout with metrics and charts
                    col1, col2, col3 = st.columns(3)
                    
                    # Format the sentiment score for display
                    sentiment_score = summary['average_sentiment']
                    sentiment_color = "green" if sentiment_score > 0.05 else "red" if sentiment_score < -0.05 else "gray"
                    sentiment_label = "Positive" if sentiment_score > 0.05 else "Negative" if sentiment_score < -0.05 else "Neutral"
                    
                    with col1:
                        st.metric(
                            "Overall Sentiment", 
                            f"{sentiment_label} ({sentiment_score:.2f})",
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            "Sentiment Trend",
                            summary['sentiment_trend'].title()
                        )
                    
                    with col3:
                        st.metric(
                            "Volume Trend",
                            summary['volume_trend'].title()
                        )
                    
                    # Show sentiment over time chart
                    st.subheader("Sentiment Over Time")
                    
                    # Create a time series of sentiment
                    fig = go.Figure()
                    
                    # Group by source and timestamp to create time series
                    for source, group in sentiment_data.groupby('source'):
                        # Sort by timestamp
                        group = group.sort_values('timestamp')
                        
                        # Add line for this source
                        fig.add_trace(
                            go.Scatter(
                                x=group['timestamp'],
                                y=group['sentiment_score'],
                                mode='lines+markers',
                                name=source,
                                hovertemplate='%{y:.2f}'
                            )
                        )
                    
                    # Add a reference line at 0
                    fig.add_shape(
                        type="line",
                        x0=sentiment_data['timestamp'].min(),
                        y0=0,
                        x1=sentiment_data['timestamp'].max(),
                        y1=0,
                        line=dict(color="gray", width=1, dash="dash"),
                    )
                    
                    # Set axis labels and title
                    fig.update_layout(
                        title="Sentiment Score by Source Over Time",
                        xaxis_title="Date",
                        yaxis_title="Sentiment Score (-1 to 1)",
                        hovermode="x unified",
                        yaxis=dict(
                            exponentformat='none',
                            separatethousands=True
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Volume chart
                    st.subheader("Mention Volume by Source")
                    
                    # Group by source and date to show daily volumes
                    sentiment_data['date'] = sentiment_data['timestamp'].dt.date
                    volume_data = sentiment_data.groupby(['source', 'date'])['volume'].sum().reset_index()
                    
                    # Create stacked bar chart for volumes
                    fig2 = go.Figure()
                    
                    for source, group in volume_data.groupby('source'):
                        fig2.add_trace(
                            go.Bar(
                                x=group['date'],
                                y=group['volume'],
                                name=source
                            )
                        )
                    
                    # Stack the bars
                    fig2.update_layout(
                        barmode='stack',
                        title="Daily Mention Volume by Source",
                        xaxis_title="Date",
                        yaxis_title="Volume (mentions)",
                        hovermode="x unified",
                        yaxis=dict(
                            exponentformat='none',
                            separatethousands=True
                        )
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Source analysis
                    st.subheader("Source Analysis")
                    
                    source_data = []
                    for source, details in summary['source_breakdown'].items():
                        source_data.append({
                            "Source": source,
                            "Average Sentiment": f"{details['average_sentiment']:.2f}",
                            "Total Volume": details['volume'],
                            "Sentiment": "Positive" if details['average_sentiment'] > 0.05 else 
                                        "Negative" if details['average_sentiment'] < -0.05 else "Neutral"
                        })
                    
                    source_df = pd.DataFrame(source_data)
                    st.dataframe(source_df, use_container_width=True)
                    
                    # Sentiment word cloud (placeholder)
                    st.subheader("Common Topics")
                    st.info("This feature will show a word cloud of common topics mentioned alongside " + 
                           f"{symbol.replace('USDT', '')}. Web scraping integration in progress.")
                    
                else:
                    st.warning(f"No sentiment data available for {symbol} with the selected filters.")
                    st.info("Please try a different cryptocurrency or adjust your filter settings.")
        except Exception as e:
            st.error(f"An error occurred in the Sentiment tab: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
    
    # Trend Visualizer Tab
    elif selected_tab == "Trend Visualizer":
        st.header("Animated Crypto Trend Visualizer")
        
        # Create a tool for visualizing crypto trends with animated emojis
        try:
            # Sidebar controls for the visualizer
            st.sidebar.header("Visualizer Settings")
            
            # Get available symbols from Binance
            available_symbols = get_available_symbols()
            default_symbol = "BTCUSDT"
            if default_symbol not in available_symbols and available_symbols:
                default_symbol = available_symbols[0]
            
            # Allow selection of multiple cryptos to visualize
            selected_cryptos = st.sidebar.multiselect(
                "Select Cryptocurrencies",
                options=available_symbols,
                default=[default_symbol, "ETHUSDT"] if "ETHUSDT" in available_symbols else [default_symbol]
            )
            
            # Timeframe selection for trend analysis
            trend_timeframe = st.sidebar.selectbox(
                "Trend Timeframe",
                options=["1h", "4h", "1d", "1w"],
                index=1  # Default to 4h
            )
            
            # Lookback period
            trend_lookback = st.sidebar.slider(
                "Lookback Period (days)",
                min_value=1,
                max_value=30,
                value=7
            )
            
            # Animation speed
            animation_speed = st.sidebar.slider(
                "Animation Speed",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1
            )
            
            # Check if any cryptos were selected
            if not selected_cryptos:
                st.warning("Please select at least one cryptocurrency to visualize.")
            else:
                # Main content area
                st.subheader("Crypto Price Trends with Emoji Indicators")
                
                # Create columns for displaying each selected crypto
                if len(selected_cryptos) > 0:
                    # Set up grid layout based on number of selected cryptos
                    if len(selected_cryptos) == 1:
                        cols = st.columns(1)
                    elif len(selected_cryptos) == 2:
                        cols = st.columns(2)
                    else:
                        cols = st.columns(min(3, len(selected_cryptos)))
                    
                    # Dictionary to store data for each crypto
                    crypto_data = {}
                    
                    # Get data for each selected crypto
                    with st.spinner("Fetching cryptocurrency data..."):
                        for symbol in selected_cryptos:
                            # Convert timeframe to interval format
                            interval = trend_timeframe
                            
                            # Calculate date range
                            end_time = datetime.now()
                            start_time = end_time - timedelta(days=trend_lookback)
                            
                            # Get data from database or API
                            df = get_cached_data(symbol, interval, trend_lookback)
                            
                            if not df.empty:
                                # Calculate percentage change and other metrics
                                df['pct_change'] = df['close'].pct_change() * 100
                                df['cumulative_return'] = (1 + df['pct_change']/100).cumprod() - 1
                                df['sma_5'] = df['close'].rolling(window=5).mean()
                                
                                # Store data
                                crypto_data[symbol] = df
                    
                    # Display each crypto in its column with animated emoji
                    for i, symbol in enumerate(selected_cryptos):
                        col_idx = i % len(cols)
                        with cols[col_idx]:
                            if symbol in crypto_data and not crypto_data[symbol].empty:
                                df = crypto_data[symbol]
                                
                                # Get latest price data
                                latest_price = df['close'].iloc[-1]
                                prev_price = df['close'].iloc[-2] if len(df) > 1 else latest_price
                                pct_change_24h = ((latest_price - prev_price) / prev_price) * 100
                                
                                # Determine emoji based on price movement
                                if pct_change_24h > 5:
                                    emoji = "🚀"  # Rocket for big gains
                                    color = "green"
                                    trend_text = "Strong Bullish"
                                elif pct_change_24h > 2:
                                    emoji = "📈"  # Chart up for gains
                                    color = "green"
                                    trend_text = "Bullish"
                                elif pct_change_24h >= 0:
                                    emoji = "✅"  # Green check for small gains
                                    color = "green"
                                    trend_text = "Slightly Bullish"
                                elif pct_change_24h > -2:
                                    emoji = "⚠️"  # Warning for small loss
                                    color = "orange"
                                    trend_text = "Slightly Bearish"
                                elif pct_change_24h > -5:
                                    emoji = "📉"  # Chart down for losses
                                    color = "red"
                                    trend_text = "Bearish"
                                else:
                                    emoji = "💥"  # Explosion for big losses
                                    color = "red"
                                    trend_text = "Strong Bearish"
                                
                                # Create static styling for emoji and text
                                st.markdown(f"""
                                <style>
                                    .emoji-container {{
                                        font-size: 3em;
                                        text-align: center;
                                    }}
                                    .crypto-info {{
                                        text-align: center;
                                    }}
                                    .price-change-{color} {{
                                        color: {color};
                                        font-weight: bold;
                                    }}
                                </style>
                                """, unsafe_allow_html=True)
                                
                                # Display cryptocurrency info with emoji
                                st.markdown(f"<div class='crypto-info'><h3>{symbol[:-4] if symbol.endswith('USDT') else symbol}</h3></div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='emoji-container'>{emoji}</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='crypto-info'>Current Price: ${latest_price:.2f}</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='crypto-info'>24h Change: <span class='price-change-{color}'>{pct_change_24h:.2f}%</span></div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='crypto-info'>Trend: <span class='price-change-{color}'>{trend_text}</span></div>", unsafe_allow_html=True)
                                
                                # Create simple trend line chart
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=df['timestamp'],
                                    y=df['close'],
                                    mode='lines',
                                    line=dict(color=color, width=2),
                                    name='Price'
                                ))
                                fig.update_layout(
                                    height=200,
                                    margin=dict(l=0, r=0, t=0, b=0),
                                    showlegend=False,
                                    xaxis=dict(
                                        showgrid=False,
                                        zeroline=False,
                                        showticklabels=False
                                    ),
                                    yaxis=dict(
                                        showgrid=False,
                                        zeroline=False,
                                        showticklabels=False
                                    )
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                            else:
                                st.error(f"No data available for {symbol}")
                
                # Add global trend insights section
                st.subheader("Market Trend Insights")
                if len(crypto_data) > 0:
                    # Calculate overall market sentiment
                    positive_count = sum(1 for symbol in crypto_data if not crypto_data[symbol].empty and 
                                      crypto_data[symbol]['pct_change'].mean() > 0)
                    total_count = len(crypto_data)
                    market_sentiment = positive_count / total_count if total_count > 0 else 0
                    
                    # Display sentiment gauge
                    sentiment_fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=market_sentiment * 100,
                        title={'text': "Market Sentiment"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "red"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "green"}
                            ],
                        }
                    ))
                    sentiment_fig.update_layout(height=300)
                    st.plotly_chart(sentiment_fig, use_container_width=True)
                    
                    # Market trend summary
                    if market_sentiment > 0.7:
                        st.success("🌟 **Bullish Market**: Most cryptocurrencies are showing positive trends. Consider looking for buying opportunities.")
                    elif market_sentiment > 0.3:
                        st.info("⚖️ **Mixed Market**: The market shows mixed trends. Exercise caution and research before making decisions.")
                    else:
                        st.error("🔴 **Bearish Market**: Most cryptocurrencies are trending downward. Consider defensive strategies.")
                    
                    # Correlation heatmap if multiple cryptos selected
                    if len(crypto_data) > 1:
                        st.subheader("Price Correlation")
                        # Prepare correlation data
                        corr_data = pd.DataFrame()
                        for symbol, df in crypto_data.items():
                            if not df.empty:
                                corr_data[symbol.replace('USDT', '')] = df['close']
                        
                        if not corr_data.empty:
                            corr_matrix = corr_data.corr()
                            
                            # Create correlation heatmap
                            heatmap_fig = go.Figure(data=go.Heatmap(
                                z=corr_matrix.values,
                                x=corr_matrix.columns,
                                y=corr_matrix.columns,
                                colorscale='Viridis',
                                zmin=-1, zmax=1
                            ))
                            heatmap_fig.update_layout(
                                title='Cryptocurrency Price Correlation',
                                height=500
                            )
                            st.plotly_chart(heatmap_fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"An error occurred in the Trend Visualizer tab: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
    
    # Portfolio Tab
    elif selected_tab == "Portfolio":
        try:
            # Portfolio Management Section
            st.header("Cryptocurrency Portfolio Tracker")
            
            # Initialize session state for portfolio management
            if 'portfolio_add_form_submitted' not in st.session_state:
                st.session_state.portfolio_add_form_submitted = False
            
            # Create tabs for portfolio views
            portfolio_tabs = st.tabs(["Portfolio Overview", "Add/Edit Holdings", "Performance Analysis"])
            
            # Get current portfolio data
            portfolio_df = database.get_portfolio()
            
            # Get current prices
            available_symbols = get_available_symbols()
            current_prices = {}
            
            with st.spinner("Fetching current crypto prices..."):
                try:
                    for symbol in available_symbols:
                        # Get most recent price from database
                        price_data = get_data(symbol, "1h", 1)
                        if not price_data.empty:
                            current_prices[symbol] = float(price_data['close'].iloc[-1])
                except Exception as e:
                    st.warning(f"Could not fetch all current prices: {e}")
            
            # Initialize portfolio_data as an empty dict with necessary structure
            portfolio_data = {
                'total_value': 0.0,
                'total_cost': 0.0,
                'total_profit_loss': 0.0,
                'total_profit_loss_percent': 0.0,
                'assets': []
            }
            
            # Tab 1: Portfolio Overview
            with portfolio_tabs[0]:
                if portfolio_df.empty:
                    st.info("Your portfolio is empty. Go to the 'Add/Edit Holdings' tab to add crypto assets.")
                else:
                    # Calculate portfolio metrics
                    portfolio_data = calculate_portfolio_value(portfolio_df, current_prices)
                    
                    # Display portfolio overview
                    st.subheader("Portfolio Summary")
                    
                    # Top-level metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Value", f"${portfolio_data['total_value']:,.2f}")
                    col2.metric("Total Cost", f"${portfolio_data['total_cost']:,.2f}")
                    col3.metric("Total Profit/Loss", f"${portfolio_data['total_profit_loss']:,.2f}")
                    col4.metric("Return", f"{portfolio_data['total_profit_loss_percent']:,.2f}%")
                
                # Portfolio allocation chart
                if portfolio_data['assets']:
                    st.subheader("Portfolio Allocation")
                    
                    # Prepare data for pie chart
                    labels = [asset['symbol'] for asset in portfolio_data['assets']]
                    values = [asset['current_value'] for asset in portfolio_data['assets']]
                    
                    # Create pie chart
                    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
                    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)
                
                # Holdings table
                st.subheader("Current Holdings")
                holdings_data = []
                
                for asset in portfolio_data['assets']:
                    holdings_data.append({
                        "ID": asset['id'],
                        "Symbol": asset['symbol'],
                        "Quantity": f"{asset['quantity']:,.8f}",
                        "Purchase Price": f"${asset['purchase_price']:,.2f}",
                        "Current Price": f"${asset['current_price']:,.2f}",
                        "Value": f"${asset['current_value']:,.2f}",
                        "Cost": f"${asset['cost_basis']:,.2f}",
                        "Profit/Loss": f"${asset['profit_loss']:,.2f}",
                        "Return": f"{asset['profit_loss_percent']:,.2f}%",
                        "Days Held": asset['days_held']
                    })
                
                st.dataframe(holdings_data, use_container_width=True)
        
            # Tab 2: Add/Edit Holdings
            with portfolio_tabs[1]:
                st.subheader("Add New Cryptocurrency")
                
                with st.form("add_crypto_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        symbol = st.selectbox("Symbol", options=available_symbols)
                        quantity = st.number_input("Quantity", min_value=0.0, format="%.8f")
                    
                    with col2:
                        purchase_price = st.number_input("Purchase Price ($)", min_value=0.0, format="%.2f")
                        purchase_date = st.date_input("Purchase Date", value=datetime.now())
                    
                    notes = st.text_area("Notes (optional)")
                    
                    submitted = st.form_submit_button("Add to Portfolio")
                    
                    if submitted:
                        try:
                            # Convert purchase date to datetime
                            purchase_datetime = datetime.combine(purchase_date, datetime.min.time())
                            
                            # Add to database
                            result = database.add_portfolio_item(
                                symbol, 
                                quantity, 
                                purchase_price, 
                                purchase_datetime, 
                                notes
                            )
                            
                            if result:
                                st.success(f"Added {quantity} {symbol} to your portfolio!")
                                st.session_state.portfolio_add_form_submitted = True
                                # Force a rerun to refresh the data
                                st.rerun()
                            else:
                                st.error("Failed to add item to portfolio.")
                        except Exception as e:
                            st.error(f"Error adding portfolio item: {e}")
                
                # Display current holdings for editing/deletion
                if not portfolio_df.empty:
                    st.subheader("Edit/Delete Holdings")
                
                    for _, row in portfolio_df.iterrows():
                        with st.expander(f"{row['symbol']} - {row['quantity']} (Purchased: {row['purchase_date'].strftime('%Y-%m-%d')})"):
                            col1, col2, col3 = st.columns([2, 2, 1])
                        
                            with col1:
                                new_quantity = st.number_input(
                                    "Update Quantity", 
                                    min_value=0.0, 
                                    value=float(row['quantity']), 
                                    key=f"qty_{row['id']}",
                                    format="%.8f"
                                )
                                
                                new_price = st.number_input(
                                    "Update Purchase Price", 
                                    min_value=0.0, 
                                    value=float(row['purchase_price']), 
                                    key=f"price_{row['id']}",
                                    format="%.2f"
                                )
                            
                            with col2:
                                new_date = st.date_input(
                                    "Update Purchase Date", 
                                    value=row['purchase_date'], 
                                    key=f"date_{row['id']}"
                                )
                                
                                new_notes = st.text_area(
                                    "Update Notes", 
                                    value=row.get('notes', ''), 
                                    key=f"notes_{row['id']}"
                                )
                            
                            with col3:
                                update_button = st.button("Update", key=f"update_{row['id']}")
                                delete_button = st.button("Delete", key=f"delete_{row['id']}")
                            
                            if update_button:
                                try:
                                    # Convert date to datetime
                                    update_datetime = datetime.combine(new_date, datetime.min.time())
                                    
                                    # Update in database
                                    result = database.update_portfolio_item(
                                        row['id'], 
                                        new_quantity, 
                                        new_price, 
                                        update_datetime, 
                                        new_notes
                                    )
                                    
                                    if result:
                                        st.success(f"Updated {row['symbol']} in your portfolio!")
                                        # Force a rerun to refresh the data
                                        st.rerun()
                                    else:
                                        st.error("Failed to update portfolio item.")
                                except Exception as e:
                                    st.error(f"Error updating portfolio item: {e}")
                            
                            if delete_button:
                                try:
                                    result = database.delete_portfolio_item(row['id'])
                                    
                                    if result:
                                        st.success(f"Deleted {row['symbol']} from your portfolio!")
                                        # Force a rerun to refresh the data
                                        st.rerun()
                                    else:
                                        st.error("Failed to delete portfolio item.")
                                except Exception as e:
                                    st.error(f"Error deleting portfolio item: {e}")
        
            # Tab 3: Performance Analysis
            with portfolio_tabs[2]:
                st.subheader("Portfolio Performance")
            
                if portfolio_df.empty:
                    st.info("Add assets to your portfolio to see performance analysis.")
                else:
                    # Time period selector
                    time_periods = {
                        "1w": "1 Week",
                        "1m": "1 Month", 
                        "3m": "3 Months", 
                        "6m": "6 Months",
                        "1y": "1 Year", 
                        "all": "All Time"
                    }
                
                    selected_period = st.selectbox("Select Time Period", options=list(time_periods.keys()), format_func=lambda x: time_periods[x])
                    
                    # Calculate date range
                    end_date = datetime.now()
                    
                    if selected_period == "1w":
                        start_date = end_date - timedelta(days=7)
                    elif selected_period == "1m":
                        start_date = end_date - timedelta(days=30)
                    elif selected_period == "3m":
                        start_date = end_date - timedelta(days=90)
                    elif selected_period == "6m":
                        start_date = end_date - timedelta(days=180)
                    elif selected_period == "1y":
                        start_date = end_date - timedelta(days=365)
                    else:
                        # All time - use earliest purchase date
                        start_date = portfolio_df['purchase_date'].min()
                    
                    # Get historical price data for each symbol in portfolio
                    symbol_price_history = {}
                    
                    with st.spinner("Fetching historical price data..."):
                        for symbol in portfolio_df['symbol'].unique():
                            try:
                                # Get data from database
                                interval = "1d"  # Daily data is best for portfolio tracking
                                price_df = get_data(symbol, interval, (end_date - start_date).days, start_date, end_date)
                                
                                if not price_df.empty:
                                    symbol_price_history[symbol] = price_df
                            except Exception as e:
                                st.warning(f"Could not fetch history for {symbol}: {e}")
                    
                    # Calculate portfolio performance over time
                    portfolio_performance = get_portfolio_performance_history(portfolio_df, symbol_price_history)
                    
                    # Get benchmark data (Bitcoin as the default benchmark)
                    benchmark_symbol = "BTCUSDT"
                    benchmark_interval = "1d"
                    
                    benchmark_df = get_data(benchmark_symbol, benchmark_interval, (end_date - start_date).days, start_date, end_date)
                    
                    if not benchmark_df.empty:
                        # Rename 'close' to 'value' for consistency
                        benchmark_df = benchmark_df.rename(columns={'close': 'value'})
                        
                        # Calculate normalized comparison data
                        comparison_data = normalize_comparison_data(portfolio_performance, benchmark_df)
                        
                        # Portfolio performance metrics
                        if not portfolio_performance.empty and len(portfolio_performance) > 1:
                            start_value = portfolio_performance['value'].iloc[0]
                            end_value = portfolio_performance['value'].iloc[-1]
                            portfolio_change_pct = ((end_value / start_value) - 1) * 100 if start_value > 0 else 0
                            
                            # Benchmark metrics
                            benchmark_start = benchmark_df['value'].iloc[0]
                            benchmark_end = benchmark_df['value'].iloc[-1]
                            benchmark_change_pct = ((benchmark_end / benchmark_start) - 1) * 100 if benchmark_start > 0 else 0
                            
                            # Display metrics
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Portfolio Performance", f"{portfolio_change_pct:.2f}%")
                            col2.metric("BTC Performance", f"{benchmark_change_pct:.2f}%")
                            col3.metric("Difference", f"{(portfolio_change_pct - benchmark_change_pct):.2f}%")
                            
                            # Create performance comparison chart
                            st.subheader("Performance Comparison")
                            
                            # Extract normalized data
                            portfolio_norm = comparison_data['portfolio']
                            benchmark_norm = comparison_data['benchmark']
                            
                            # Create traces
                            fig = go.Figure()
                            
                            if portfolio_norm:
                                timestamps = [entry['timestamp'] for entry in portfolio_norm]
                                values = [entry['normalized'] for entry in portfolio_norm]
                                
                                fig.add_trace(go.Scatter(
                                    x=timestamps,
                                    y=values,
                                    mode='lines',
                                    name='Your Portfolio',
                                    line=dict(color='#5DADEC', width=2)
                                ))
                            
                            if benchmark_norm:
                                timestamps = [entry['timestamp'] for entry in benchmark_norm]
                                values = [entry['normalized'] for entry in benchmark_norm]
                                
                                fig.add_trace(go.Scatter(
                                    x=timestamps,
                                    y=values,
                                    mode='lines',
                                    name='Bitcoin (BTC)',
                                    line=dict(color='#FF9900', width=2, dash='dot')
                                ))
                            
                            # Update layout
                            fig.update_layout(
                                title="Normalized Performance (Starting Value = 100)",
                                xaxis_title="Date",
                                yaxis_title="Value",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                margin=dict(l=20, r=20, t=60, b=20),
                                height=500,
                                yaxis=dict(
                                    exponentformat='none',
                                    separatethousands=True
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Not enough performance data for the selected time period. Try extending the time range or add more historical assets.")
                    else:
                        st.warning(f"Could not fetch benchmark data for {benchmark_symbol}. Try a different time period.")

        except Exception as e:
            st.error(f"An error occurred in Portfolio tab: {str(e)}")
            import traceback
            st.text(traceback.format_exc())

if __name__ == "__main__":
    main()
