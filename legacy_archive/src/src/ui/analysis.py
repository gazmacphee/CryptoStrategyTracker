"""
Technical Analysis UI Module

Provides tools for analyzing cryptocurrency price data using technical analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from src.config.container import container


def render_analysis():
    """Render the technical analysis UI"""
    st.header("Technical Analysis")
    
    # Get services
    data_service = container.get("data_service")
    indicators_service = container.get("indicators_service")
    
    # Sidebar for analysis controls
    st.sidebar.header("Analysis Settings")
    
    # Get available symbols
    available_symbols = data_service.get_available_symbols()
    default_symbol = "BTCUSDT"
    if default_symbol not in available_symbols and available_symbols:
        default_symbol = available_symbols[0]
    
    # Symbol selection
    symbol = st.sidebar.selectbox(
        "Select Cryptocurrency",
        options=available_symbols,
        index=available_symbols.index(default_symbol) if default_symbol in available_symbols else 0,
        key="analysis_symbol"
    )
    
    # Timeframe selection
    interval = st.sidebar.selectbox(
        "Select Timeframe",
        options=["15m", "30m", "1h", "4h", "1d"],
        index=3,  # Default to 4h
        key="analysis_interval"
    )
    
    # Date range
    days_back = st.sidebar.slider(
        "Days to Analyze",
        min_value=1,
        max_value=90,
        value=30,
        help="Number of days to include in analysis"
    )
    
    # Calculate date range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)
    
    # Select indicators
    st.sidebar.subheader("Technical Indicators")
    
    show_bollinger = st.sidebar.checkbox("Bollinger Bands", value=True)
    show_rsi = st.sidebar.checkbox("RSI", value=True)
    show_macd = st.sidebar.checkbox("MACD", value=True)
    show_ema = st.sidebar.checkbox("EMA", value=True)
    show_volume = st.sidebar.checkbox("Volume", value=True)
    
    # Analyze button
    analyze = st.sidebar.button("Analyze")
    
    # Main content
    if analyze:
        with st.spinner(f"Analyzing {symbol}/{interval}..."):
            # Get historical data
            df = data_service.get_klines_data(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time
            )
            
            if df.empty:
                st.error(f"No data available for {symbol}/{interval} in the selected time range")
                return
            
            # Calculate indicators
            with_indicators = indicators_service.calculate_all_indicators(df)
            
            # Create subplots
            fig = make_subplots(
                rows=3, 
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.6, 0.2, 0.2],
                specs=[
                    [{"type": "candlestick"}],
                    [{"type": "scatter"}],
                    [{"type": "bar"}]
                ],
                subplot_titles=("Price", "Indicators", "Volume")
            )
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name=symbol
                ),
                row=1, col=1
            )
            
            # Add indicators
            if show_bollinger and 'bb_upper' in with_indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=with_indicators['timestamp'],
                        y=with_indicators['bb_upper'],
                        name="BB Upper",
                        line=dict(color='rgba(250, 0, 0, 0.7)')
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=with_indicators['timestamp'],
                        y=with_indicators['bb_middle'],
                        name="BB Middle",
                        line=dict(color='rgba(0, 0, 250, 0.7)')
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=with_indicators['timestamp'],
                        y=with_indicators['bb_lower'],
                        name="BB Lower",
                        line=dict(color='rgba(0, 250, 0, 0.7)')
                    ),
                    row=1, col=1
                )
            
            if show_ema and 'ema_9' in with_indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=with_indicators['timestamp'],
                        y=with_indicators['ema_9'],
                        name="EMA 9",
                        line=dict(color='rgba(255, 165, 0, 0.7)')
                    ),
                    row=1, col=1
                )
            
            if show_ema and 'ema_21' in with_indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=with_indicators['timestamp'],
                        y=with_indicators['ema_21'],
                        name="EMA 21",
                        line=dict(color='rgba(148, 0, 211, 0.7)')
                    ),
                    row=1, col=1
                )
            
            if show_ema and 'ema_50' in with_indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=with_indicators['timestamp'],
                        y=with_indicators['ema_50'],
                        name="EMA 50",
                        line=dict(color='rgba(128, 128, 128, 0.7)')
                    ),
                    row=1, col=1
                )
            
            # Add RSI
            if show_rsi and 'rsi' in with_indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=with_indicators['timestamp'],
                        y=with_indicators['rsi'],
                        name="RSI"
                    ),
                    row=2, col=1
                )
                
                # Add RSI overbought/oversold lines
                fig.add_trace(
                    go.Scatter(
                        x=[with_indicators['timestamp'].iloc[0], with_indicators['timestamp'].iloc[-1]],
                        y=[70, 70],
                        name="Overbought",
                        line=dict(color='red', dash='dash')
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[with_indicators['timestamp'].iloc[0], with_indicators['timestamp'].iloc[-1]],
                        y=[30, 30],
                        name="Oversold",
                        line=dict(color='green', dash='dash')
                    ),
                    row=2, col=1
                )
            
            # Add MACD
            if show_macd and 'macd' in with_indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=with_indicators['timestamp'],
                        y=with_indicators['macd'],
                        name="MACD",
                        line=dict(color='blue')
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=with_indicators['timestamp'],
                        y=with_indicators['macd_signal'],
                        name="Signal",
                        line=dict(color='red')
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=with_indicators['timestamp'],
                        y=with_indicators['macd_histogram'],
                        name="Histogram",
                        marker_color='green'
                    ),
                    row=2, col=1
                )
            
            # Add volume
            if show_volume:
                fig.add_trace(
                    go.Bar(
                        x=df['timestamp'],
                        y=df['volume'],
                        name="Volume",
                        marker_color='rgba(0, 0, 255, 0.5)'
                    ),
                    row=3, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=f"{symbol} Technical Analysis ({interval})",
                height=800,
                xaxis_rangeslider_visible=False,
                xaxis3_title="Date",
                yaxis_title="Price (USD)",
                yaxis2_title="Oscillators",
                yaxis3_title="Volume"
            )
            
            # Show chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical Analysis Insights
            st.subheader("Technical Analysis Insights")
            
            # Calculate last values
            last_row = with_indicators.iloc[-1]
            
            # Create columns for insights
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Price Action")
                last_close = last_row['close']
                prev_close = with_indicators.iloc[-2]['close'] if len(with_indicators) > 1 else last_close
                price_change = (last_close - prev_close) / prev_close * 100
                
                st.metric(
                    "Current Price",
                    f"${last_close:.2f}",
                    delta=f"{price_change:.2f}%",
                    delta_color="normal" if price_change >= 0 else "inverse"
                )
                
                if 'ema_50' in with_indicators.columns:
                    ema_50 = last_row['ema_50']
                    ema_signal = "Above EMA 50 (Bullish)" if last_close > ema_50 else "Below EMA 50 (Bearish)"
                    ema_color = "green" if last_close > ema_50 else "red"
                    
                    st.markdown(f"**EMA Signal:** <span style='color:{ema_color}'>{ema_signal}</span>", unsafe_allow_html=True)
                
                if 'bb_upper' in with_indicators.columns and 'bb_lower' in with_indicators.columns:
                    bb_upper = last_row['bb_upper']
                    bb_lower = last_row['bb_lower']
                    
                    if last_close > bb_upper:
                        bb_signal = "Above Upper Band (Overbought)"
                        bb_color = "red"
                    elif last_close < bb_lower:
                        bb_signal = "Below Lower Band (Oversold)"
                        bb_color = "green"
                    else:
                        bb_signal = "Within Bands (Neutral)"
                        bb_color = "gray"
                    
                    st.markdown(f"**Bollinger Bands:** <span style='color:{bb_color}'>{bb_signal}</span>", unsafe_allow_html=True)
            
            with col2:
                st.subheader("Momentum")
                
                if 'rsi' in with_indicators.columns:
                    rsi = last_row['rsi']
                    
                    if rsi > 70:
                        rsi_signal = "Overbought"
                        rsi_color = "red"
                    elif rsi < 30:
                        rsi_signal = "Oversold"
                        rsi_color = "green"
                    else:
                        rsi_signal = "Neutral"
                        rsi_color = "gray"
                    
                    st.metric(
                        "RSI (14)",
                        f"{rsi:.1f}",
                        delta=rsi_signal,
                        delta_color="normal" if rsi_color == "green" else "inverse" if rsi_color == "red" else "off"
                    )
                
                if 'macd' in with_indicators.columns and 'macd_signal' in with_indicators.columns:
                    macd = last_row['macd']
                    macd_signal = last_row['macd_signal']
                    
                    prev_macd = with_indicators.iloc[-2]['macd'] if len(with_indicators) > 1 else macd
                    prev_macd_signal = with_indicators.iloc[-2]['macd_signal'] if len(with_indicators) > 1 else macd_signal
                    
                    macd_cross_up = prev_macd < prev_macd_signal and macd > macd_signal
                    macd_cross_down = prev_macd > prev_macd_signal and macd < macd_signal
                    
                    if macd_cross_up:
                        macd_text = "Bullish Crossover"
                        macd_color = "green"
                    elif macd_cross_down:
                        macd_text = "Bearish Crossover"
                        macd_color = "red"
                    elif macd > macd_signal:
                        macd_text = "Bullish"
                        macd_color = "green"
                    else:
                        macd_text = "Bearish"
                        macd_color = "red"
                    
                    st.markdown(f"**MACD Signal:** <span style='color:{macd_color}'>{macd_text}</span>", unsafe_allow_html=True)
            
            with col3:
                st.subheader("Overall Signal")
                
                # Count bullish signals
                bullish_signals = 0
                bearish_signals = 0
                
                # EMA
                if 'ema_50' in with_indicators.columns:
                    if last_close > last_row['ema_50']:
                        bullish_signals += 1
                    else:
                        bearish_signals += 1
                
                # Bollinger Bands
                if 'bb_upper' in with_indicators.columns and 'bb_lower' in with_indicators.columns:
                    if last_close < last_row['bb_lower']:
                        bullish_signals += 1  # Oversold
                    elif last_close > last_row['bb_upper']:
                        bearish_signals += 1  # Overbought
                
                # RSI
                if 'rsi' in with_indicators.columns:
                    if last_row['rsi'] < 30:
                        bullish_signals += 1
                    elif last_row['rsi'] > 70:
                        bearish_signals += 1
                
                # MACD
                if 'macd' in with_indicators.columns and 'macd_signal' in with_indicators.columns:
                    if last_row['macd'] > last_row['macd_signal']:
                        bullish_signals += 1
                    else:
                        bearish_signals += 1
                
                # Overall signal
                total_signals = bullish_signals + bearish_signals
                if total_signals > 0:
                    bullish_pct = (bullish_signals / total_signals) * 100
                    
                    if bullish_pct >= 75:
                        signal = "Strong Buy"
                        color = "green"
                    elif bullish_pct >= 50:
                        signal = "Buy"
                        color = "lightgreen"
                    elif bullish_pct >= 25:
                        signal = "Sell"
                        color = "lightcoral"
                    else:
                        signal = "Strong Sell"
                        color = "red"
                    
                    st.markdown(f"<h3 style='color:{color}'>{signal}</h3>", unsafe_allow_html=True)
                    st.markdown(f"**Bullish Signals:** {bullish_signals}/{total_signals}")
                    st.markdown(f"**Bearish Signals:** {bearish_signals}/{total_signals}")
                else:
                    st.markdown("**Insufficient signals to determine trend**")
    
    else:
        # Show instructions when no analysis is running
        st.info("""
        ## Technical Analysis Tools
        
        This tool provides advanced technical analysis capabilities for cryptocurrency traders.
        
        ### Available Indicators:
        - **Bollinger Bands** - Identify overbought/oversold conditions
        - **RSI (Relative Strength Index)** - Measure momentum
        - **MACD (Moving Average Convergence Divergence)** - Identify trend changes
        - **EMA (Exponential Moving Average)** - Track trend direction
        - **Volume Analysis** - Confirm price movements
        
        ### To get started:
        1. Select a cryptocurrency and timeframe in the sidebar
        2. Choose which indicators to display
        3. Click "Analyze" to generate the technical analysis
        
        The analysis includes both visualizations and actionable trading insights.
        """)