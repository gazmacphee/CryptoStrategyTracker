import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os

# Import local modules
from database import create_tables, save_historical_data, get_historical_data
from binance_api import get_klines_data, get_available_symbols
from indicators import add_bollinger_bands, add_macd, add_rsi, add_ema
from strategy import evaluate_buy_sell_signals
from utils import timeframe_to_seconds, timeframe_to_interval, get_timeframe_options

# Set page config
st.set_page_config(
    page_title="Crypto Trading Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
create_tables()

# Function to fetch data from Binance API or Database
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_data(symbol, interval, lookback_days):
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)
    
    # Try to get data from database first
    df = get_historical_data(symbol, interval, start_time, end_time)
    
    # If not enough data in database, fetch from API
    if df.empty or len(df) < 100:
        df = get_klines_data(symbol, interval, start_time, end_time)
        if not df.empty:
            save_historical_data(df, symbol, interval)
    
    return df

def main():
    st.title("Cryptocurrency Trading Analysis Platform")
    
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
        index=3  # Default to 1h
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
    
    # Get data from API or database
    try:
        with st.spinner(f"Fetching {symbol} data..."):
            df = get_data(symbol, binance_interval, lookback_days)
            
            if df.empty:
                st.error("No data available for the selected pair and timeframe.")
                return
            
            # Calculate indicators
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
            
            # Evaluate buy/sell signals
            df = evaluate_buy_sell_signals(df, bb_threshold, rsi_oversold, rsi_overbought)
    
        # Create main chart with candlesticks
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{symbol} - {interval}", "Volume")
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
        
        # Add MACD to subplot
        if show_macd and 'macd' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['macd'],
                    line=dict(color='blue', width=1.5),
                    name="MACD"
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['macd_signal'],
                    line=dict(color='red', width=1.5),
                    name="Signal"
                ),
                row=2, col=1
            )
            
            colors = ['green' if val > 0 else 'red' for val in df['macd_histogram']]
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df['macd_histogram'],
                    marker_color=colors,
                    name="Histogram"
                ),
                row=2, col=1
            )
        
        # RSI indicator (if selected)
        if show_rsi and 'rsi' in df.columns:
            # Replace MACD with RSI if selected
            if not show_macd:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['rsi'],
                        line=dict(color='purple', width=1.5),
                        name="RSI"
                    ),
                    row=2, col=1
                )
                
                # Add overbought/oversold lines
                fig.add_trace(
                    go.Scatter(
                        x=[df['timestamp'].iloc[0], df['timestamp'].iloc[-1]],
                        y=[rsi_overbought, rsi_overbought],
                        line=dict(color='red', width=1, dash='dash'),
                        name=f"Overbought ({rsi_overbought})"
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[df['timestamp'].iloc[0], df['timestamp'].iloc[-1]],
                        y=[rsi_oversold, rsi_oversold],
                        line=dict(color='green', width=1, dash='dash'),
                        name=f"Oversold ({rsi_oversold})"
                    ),
                    row=2, col=1
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
        )
        
        fig.update_xaxes(
            gridcolor="rgba(255, 255, 255, 0.1)",
            showgrid=True
        )
        
        fig.update_yaxes(
            gridcolor="rgba(255, 255, 255, 0.1)",
            showgrid=True
        )
        
        # Show chart
        chart_container = st.container()
        with chart_container:
            chart = st.plotly_chart(fig, use_container_width=True)
            
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
        
        # Strategy Performance
        st.subheader("Strategy Performance")
        
        # Simple backtest calculation
        if not df.empty and len(df) > 1:
            buy_prices = []
            sell_prices = []
            current_position = False
            
            for i, row in df.iterrows():
                if row['buy_signal'] and not current_position:
                    buy_prices.append(row['close'])
                    current_position = True
                elif row['sell_signal'] and current_position:
                    sell_prices.append(row['close'])
                    current_position = False
            
            # Make sure we have equal buy/sell pairs
            min_trades = min(len(buy_prices), len(sell_prices))
            buy_prices = buy_prices[:min_trades]
            sell_prices = sell_prices[:min_trades]
            
            # Calculate performance
            if min_trades > 0:
                trade_results = [(sell - buy) / buy * 100 for buy, sell in zip(buy_prices, sell_prices)]
                winning_trades = sum(1 for res in trade_results if res > 0)
                losing_trades = sum(1 for res in trade_results if res <= 0)
                
                # Calculate metrics
                total_return = sum(trade_results)
                avg_return = sum(trade_results) / len(trade_results) if trade_results else 0
                win_rate = winning_trades / min_trades * 100 if min_trades > 0 else 0
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Trades", min_trades)
                
                with col2:
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                
                with col3:
                    st.metric("Avg Return per Trade", f"{avg_return:.2f}%")
                
                with col4:
                    st.metric("Total Return", f"{total_return:.2f}%")
                
                # Recent trade history
                if min_trades > 0:
                    st.subheader("Recent Trade History")
                    trade_history = []
                    
                    for i in range(min(min_trades, 10)):
                        trade_history.append({
                            "Entry Price": f"${buy_prices[i]:.2f}",
                            "Exit Price": f"${sell_prices[i]:.2f}",
                            "Return (%)": f"{trade_results[i]:.2f}%",
                            "Result": "Win" if trade_results[i] > 0 else "Loss"
                        })
                    
                    # Reverse to show most recent first
                    trade_history.reverse()
                    st.dataframe(trade_history, use_container_width=True)
            else:
                st.info("No completed trades to analyze.")
        
        # Auto-refresh logic
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
