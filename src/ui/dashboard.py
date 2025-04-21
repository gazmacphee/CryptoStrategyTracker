"""
Dashboard UI Module

Renders the main dashboard with key metrics, performance indicators,
and market overview.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from src.config.container import container


def render_dashboard():
    """Render the main dashboard"""
    st.header("Crypto Trading Dashboard")
    
    # Get services
    data_service = container.get("data_service")
    
    # Get available symbols
    available_symbols = data_service.get_available_symbols()
    
    # Dashboard configuration
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Market Overview")
        
        # Symbol selection
        selected_symbol = st.selectbox(
            "Select Cryptocurrency",
            options=available_symbols,
            index=0 if "BTCUSDT" in available_symbols else 0,
            key="dashboard_symbol"
        )
        
        # Timeframe selection
        selected_interval = st.selectbox(
            "Select Timeframe",
            options=["15m", "1h", "4h", "1d"],
            index=2,  # Default to 4h
            key="dashboard_interval"
        )
        
        # Time range
        time_range = st.selectbox(
            "Time Range",
            options=["Last 24 hours", "Last 7 days", "Last 30 days", "Last 90 days"],
            index=1,
            key="dashboard_time_range"
        )
        
        # Convert time range to actual dates
        end_time = datetime.now()
        if time_range == "Last 24 hours":
            start_time = end_time - timedelta(days=1)
        elif time_range == "Last 7 days":
            start_time = end_time - timedelta(days=7)
        elif time_range == "Last 30 days":
            start_time = end_time - timedelta(days=30)
        else:  # Last 90 days
            start_time = end_time - timedelta(days=90)
        
        # Refresh button
        refresh = st.button("Refresh Data")
    
    # Load data
    with st.spinner("Loading data..."):
        # Get historical data
        df = data_service.get_klines_data(
            symbol=selected_symbol,
            interval=selected_interval,
            start_time=start_time,
            end_time=end_time
        )
        
        if df.empty:
            st.error(f"No data available for {selected_symbol}/{selected_interval}")
            return
        
        # Show main chart in the larger column
        with col2:
            # Current price and change
            latest_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2] if len(df) > 1 else latest_price
            price_change = latest_price - prev_price
            price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
            
            # Display current price with change
            st.metric(
                label=f"{selected_symbol} Price",
                value=f"${latest_price:.2f}",
                delta=f"{price_change_pct:.2f}%",
                delta_color="normal" if price_change >= 0 else "inverse"
            )
            
            # Create price chart
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
    
    # Additional dashboard sections
    st.markdown("---")
    
    # Market statistics
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Calculate volume metrics
            latest_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].mean()
            volume_ratio = latest_volume / avg_volume if avg_volume != 0 else 0
            
            st.metric(
                label="Volume",
                value=f"{latest_volume:.1f}",
                delta=f"{(volume_ratio-1)*100:.1f}% vs avg",
                delta_color="normal" if volume_ratio >= 1 else "inverse"
            )
        
        with col2:
            # Calculate volatility (using ATR proxy)
            recent_high = df['high'].iloc[-10:].max()
            recent_low = df['low'].iloc[-10:].min()
            price_range = (recent_high - recent_low) / recent_low if recent_low != 0 else 0
            
            st.metric(
                label="Volatility",
                value=f"{price_range*100:.1f}%",
                delta=None
            )
        
        with col3:
            # Calculate price range
            max_price = df['high'].max()
            min_price = df['low'].min()
            
            st.metric(
                label="Price Range",
                value=f"${min_price:.2f} - ${max_price:.2f}",
                delta=None
            )
        
        with col4:
            # Calculate performance vs. beginning of period
            first_price = df['open'].iloc[0]
            performance = (latest_price - first_price) / first_price if first_price != 0 else 0
            
            st.metric(
                label=f"Period Performance",
                value=f"{performance*100:.2f}%",
                delta=None,
                delta_color="normal" if performance >= 0 else "inverse"
            )
    
    # Market summary
    st.markdown("---")
    st.subheader("Market Summary")
    
    # Get top symbols
    top_symbols = available_symbols[:5]  # Get first 5 symbols
    
    # Create placeholders for data we'll load
    market_data = []
    
    # Load data for each symbol
    with st.spinner("Loading market data..."):
        for symbol in top_symbols:
            try:
                # Get last 24h of data
                yesterday = datetime.now() - timedelta(days=1)
                df = data_service.get_klines_data(
                    symbol=symbol,
                    interval="1h",
                    start_time=yesterday
                )
                
                if not df.empty:
                    # Calculate metrics
                    latest_price = df['close'].iloc[-1]
                    open_price = df['open'].iloc[0]
                    change_24h = (latest_price - open_price) / open_price if open_price != 0 else 0
                    high_24h = df['high'].max()
                    low_24h = df['low'].min()
                    volume_24h = df['volume'].sum()
                    
                    market_data.append({
                        'Symbol': symbol,
                        'Price': latest_price,
                        'Change 24h': change_24h * 100,  # as percentage
                        'High 24h': high_24h,
                        'Low 24h': low_24h,
                        'Volume 24h': volume_24h
                    })
            except Exception as e:
                # Log the error but continue
                st.error(f"Error loading data for {symbol}: {e}")
    
    # Display market data
    if market_data:
        market_df = pd.DataFrame(market_data)
        
        # Format columns
        market_df['Price'] = market_df['Price'].map('${:.2f}'.format)
        market_df['Change 24h'] = market_df['Change 24h'].map('{:.2f}%'.format)
        market_df['High 24h'] = market_df['High 24h'].map('${:.2f}'.format)
        market_df['Low 24h'] = market_df['Low 24h'].map('${:.2f}'.format)
        market_df['Volume 24h'] = market_df['Volume 24h'].map('{:.1f}'.format)
        
        # Display as a table
        st.dataframe(market_df, use_container_width=True)
    else:
        st.warning("No market data available. Please check your data connection or backfill process.")