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

                # Strategy Optimization - Find the most profitable parameters
                st.subheader("Strategy Optimization")
                
                with st.expander("Optimize Trading Strategy", expanded=True):
                    if st.button("Run Optimization"):
                        with st.spinner("Finding optimal trading parameters..."):
                            # Define parameter ranges to test
                            parameter_ranges = {
                                'bb_threshold': [0.1, 0.2, 0.3, 0.4, 0.5],
                                'rsi_oversold': [20, 25, 30, 35],
                                'rsi_overbought': [65, 70, 75, 80, 85]
                            }
                            
                            # Run optimization
                            optimization_results = find_optimal_strategy(df, parameter_ranges)
                            
                            if optimization_results:
                                st.success("Optimization completed!")
                                
                                # Display optimal parameters
                                st.write("### Optimal Strategy Parameters")
                                
                                params = optimization_results['parameters']
                                opt_metrics = optimization_results['metrics']
                                
                                # Create side-by-side columns
                                p1, p2, p3 = st.columns(3)
                                with p1:
                                    st.metric("Bollinger Band Threshold", f"{params['bb_threshold']:.2f}")
                                with p2:
                                    st.metric("RSI Oversold", f"{params['rsi_oversold']}")
                                with p3:
                                    st.metric("RSI Overbought", f"{params['rsi_overbought']}")
                                
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
