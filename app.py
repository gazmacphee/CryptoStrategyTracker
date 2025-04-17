import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
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
        cursor.execute("SELECT COUNT(DISTINCT symbol, interval) FROM historical_data")
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

def run_background_backfill():
    """Run a background thread to backfill database without blocking the UI"""
    def backfill_thread():
        subprocess.run(["python", "backfill_database.py"])
    
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
    
    # Initialize session state for backfill tracking
    if 'backfill_started' not in st.session_state:
        st.session_state.backfill_started = False
        st.session_state.backfill_thread = None
    
    # Start background backfill automatically on first load
    if not st.session_state.backfill_started:
        st.session_state.backfill_thread = run_background_backfill()
        st.session_state.backfill_started = True
        st.info("Initial database backfill started in the background. This will populate your database with historical data, indicators, and trading signals.")
    
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
            st.info(f"Database last updated: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.warning("Database not yet populated. Initial backfill in progress...")
    
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
    if st.sidebar.button("Backfill Database with Optimized Strategies"):
        st.sidebar.info("Running backfill to calculate and store optimized trading strategies...")
        import subprocess
        try:
            process = subprocess.Popen(["python", "backfill_database.py"], 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE,
                                      text=True)
            # Display first update 
            st.sidebar.success("Backfill process started in the background. Check the server logs for progress.")
        except Exception as e:
            st.sidebar.error(f"Error starting backfill process: {e}")
            
    # Get data from API or database
    try:
        # Show last update time for this specific symbol/interval
        last_update_for_interval = get_last_update_time(symbol, binance_interval)
        if last_update_for_interval:
            st.info(f"Latest data for {symbol} ({interval}) from: {last_update_for_interval.strftime('%Y-%m-%d %H:%M:%S')}")
        
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
                spikethickness=1
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

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
