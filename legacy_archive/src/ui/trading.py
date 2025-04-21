"""
Trading UI Module

Provides tools for simulating and tracking cryptocurrency trades.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from src.config.container import container


def render_trading():
    """Render the trading UI"""
    st.header("Trading Dashboard")
    
    # Create tabs for different trading features
    tabs = st.tabs(["Portfolio", "Trade Simulator", "Performance"])
    
    with tabs[0]:
        render_portfolio_tab()
    
    with tabs[1]:
        render_trade_simulator_tab()
    
    with tabs[2]:
        render_performance_tab()


def render_portfolio_tab():
    """Render the portfolio management tab"""
    st.subheader("Portfolio Management")
    
    # Get services
    data_service = container.get("data_service")
    
    # Display portfolio summary
    st.write("Track your cryptocurrency holdings and monitor their performance over time.")
    
    # Portfolio management options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Add Position")
        
        # Get available symbols
        available_symbols = data_service.get_available_symbols()
        
        # Symbol selection
        symbol = st.selectbox(
            "Select Cryptocurrency",
            options=available_symbols,
            key="portfolio_symbol"
        )
        
        # Amount
        quantity = st.number_input(
            "Quantity",
            min_value=0.0,
            value=1.0,
            step=0.01,
            key="portfolio_quantity"
        )
        
        # Purchase price
        purchase_price = st.number_input(
            "Purchase Price (USD)",
            min_value=0.0,
            value=0.0,
            step=0.01,
            key="portfolio_price"
        )
        
        # Purchase date
        purchase_date = st.date_input(
            "Purchase Date",
            value=datetime.now(),
            key="portfolio_date"
        )
        
        # Notes
        notes = st.text_area(
            "Notes",
            value="",
            key="portfolio_notes"
        )
        
        # Add button
        if st.button("Add to Portfolio"):
            st.warning("This is a placeholder for adding a portfolio item.")
            st.success(f"Added {quantity} {symbol} to portfolio")
    
    with col2:
        st.markdown("#### Portfolio Summary")
        
        # Sample portfolio data
        portfolio_data = [
            {"Symbol": "BTC", "Quantity": 0.5, "Purchase Price": 50000, "Current Price": 55000, "Profit/Loss": 2500, "Profit %": 10.0},
            {"Symbol": "ETH", "Quantity": 5.0, "Purchase Price": 3000, "Current Price": 3200, "Profit/Loss": 1000, "Profit %": 6.67},
            {"Symbol": "SOL", "Quantity": 20.0, "Purchase Price": 150, "Current Price": 140, "Profit/Loss": -200, "Profit %": -6.67}
        ]
        
        # Show portfolio table
        st.dataframe(pd.DataFrame(portfolio_data), use_container_width=True)
        
        # Calculate totals
        total_investment = sum(item["Quantity"] * item["Purchase Price"] for item in portfolio_data)
        total_value = sum(item["Quantity"] * item["Current Price"] for item in portfolio_data)
        total_profit = total_value - total_investment
        total_profit_pct = (total_profit / total_investment) * 100 if total_investment > 0 else 0
        
        # Show portfolio metrics
        st.metric(
            "Total Portfolio Value",
            f"${total_value:,.2f}",
            delta=f"${total_profit:,.2f} ({total_profit_pct:.2f}%)",
            delta_color="normal" if total_profit >= 0 else "inverse"
        )


def render_trade_simulator_tab():
    """Render the trade simulator tab"""
    st.subheader("Trade Simulator")
    
    # Get services
    data_service = container.get("data_service")
    
    # Display trade simulator description
    st.write("Simulate cryptocurrency trades and track their performance over time.")
    
    # Trade simulator controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### New Trade")
        
        # Get available symbols
        available_symbols = data_service.get_available_symbols()
        
        # Symbol selection
        symbol = st.selectbox(
            "Select Cryptocurrency",
            options=available_symbols,
            key="trade_symbol"
        )
        
        # Trade type
        trade_type = st.selectbox(
            "Trade Type",
            options=["Buy", "Sell"],
            key="trade_type"
        )
        
        # Timeframe selection
        interval = st.selectbox(
            "Select Timeframe",
            options=["15m", "30m", "1h", "4h", "1d"],
            index=3,  # Default to 4h
            key="trade_interval"
        )
        
        # Trade strategy
        strategy = st.selectbox(
            "Trading Strategy",
            options=["Manual", "ML-Based", "Technical Analysis"],
            key="trade_strategy"
        )
        
        # Quantity
        quantity = st.number_input(
            "Quantity",
            min_value=0.0,
            value=1.0,
            step=0.01,
            key="trade_quantity"
        )
        
        # Entry price
        entry_price = st.number_input(
            "Entry Price (USD)",
            min_value=0.0,
            value=0.0,
            step=0.01,
            key="trade_entry_price"
        )
        
        # Entry time
        entry_time = st.date_input(
            "Entry Date",
            value=datetime.now(),
            key="trade_entry_date"
        )
        
        # Exit price (optional for open trades)
        exit_price = st.number_input(
            "Exit Price (USD, optional)",
            min_value=0.0,
            value=0.0,
            step=0.01,
            key="trade_exit_price"
        )
        
        # Exit time (optional for open trades)
        exit_time = st.date_input(
            "Exit Date (optional)",
            value=None,
            key="trade_exit_date"
        )
        
        # Add button
        if st.button("Save Trade"):
            st.warning("This is a placeholder for saving a trade.")
            st.success(f"Saved {trade_type} trade for {quantity} {symbol}")
    
    with col2:
        st.markdown("#### Recent Trades")
        
        # Sample trade data
        trades_data = [
            {"Symbol": "BTC", "Type": "Buy", "Entry Price": 50000, "Exit Price": 55000, "Profit/Loss": 5000, "Profit %": 10.0, "Status": "Closed"},
            {"Symbol": "ETH", "Type": "Buy", "Entry Price": 3000, "Exit Price": None, "Profit/Loss": None, "Profit %": None, "Status": "Open"},
            {"Symbol": "SOL", "Type": "Sell", "Entry Price": 150, "Exit Price": 140, "Profit/Loss": 200, "Profit %": 6.67, "Status": "Closed"}
        ]
        
        # Show trades table
        st.dataframe(pd.DataFrame(trades_data), use_container_width=True)
        
        # Trade performance metrics
        st.markdown("#### Trade Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Win Rate",
                "66.7%",
                delta=None
            )
        
        with col2:
            st.metric(
                "Average Profit/Loss",
                "+8.34%",
                delta=None
            )


def render_performance_tab():
    """Render the performance tracking tab"""
    st.subheader("Performance Tracking")
    
    # Display performance tracking description
    st.write("Track the performance of your trading strategies over time.")
    
    # Performance tracking controls
    time_period = st.selectbox(
        "Time Period",
        options=["Last 7 days", "Last 30 days", "Last 90 days", "Last 365 days", "All time"],
        index=1,
        key="performance_period"
    )
    
    # Performance metrics
    st.markdown("#### Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Trades",
            "24",
            delta=None
        )
    
    with col2:
        st.metric(
            "Win Rate",
            "67%",
            delta="+7% vs previous",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            "Total Profit/Loss",
            "+$12,450",
            delta="+22% vs previous",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            "Average Trade",
            "+5.2%",
            delta="+0.8% vs previous",
            delta_color="normal"
        )
    
    # Performance chart
    st.markdown("#### Performance Over Time")
    
    # Create sample data for performance chart
    days = 30
    dates = [datetime.now() - timedelta(days=i) for i in range(days)]
    dates.reverse()  # Reverse to get ascending dates
    
    # Sample cumulative portfolio value
    value_start = 10000
    daily_changes = [1 + (0.01 * (i % 5 - 2)) for i in range(days)]
    values = [value_start]
    
    for change in daily_changes:
        values.append(values[-1] * change)
    
    values = values[1:]  # Remove the initial value
    
    # Create figure
    fig = go.Figure()
    
    # Add line chart
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='green' if values[-1] > values[0] else 'red', width=2)
    ))
    
    # Add benchmark (e.g., BTC)
    btc_changes = [1 + (0.007 * (i % 7 - 3)) for i in range(days)]
    btc_values = [value_start]
    
    for change in btc_changes:
        btc_values.append(btc_values[-1] * change)
    
    btc_values = btc_values[1:]  # Remove the initial value
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=btc_values,
        mode='lines',
        name='BTC (Benchmark)',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title='Portfolio Performance vs Benchmark',
        xaxis_title='Date',
        yaxis_title='Value (USD)',
        height=500,
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
    
    # Trade distribution
    st.markdown("#### Trade Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**By Cryptocurrency**")
        
        # Sample data
        assets = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
        trades = [10, 7, 4, 2, 1]
        
        # Create figure
        fig1 = go.Figure(data=[go.Pie(
            labels=assets,
            values=trades,
            hole=.3
        )])
        
        # Update layout
        fig1.update_layout(
            title='Trade Count by Asset',
            height=400
        )
        
        # Show chart
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("**By Strategy**")
        
        # Sample data
        strategies = ["Manual", "Technical Analysis", "ML-Based"]
        trade_counts = [12, 8, 4]
        
        # Create figure
        fig2 = go.Figure(data=[go.Pie(
            labels=strategies,
            values=trade_counts,
            hole=.3
        )])
        
        # Update layout
        fig2.update_layout(
            title='Trade Count by Strategy',
            height=400
        )
        
        # Show chart
        st.plotly_chart(fig2, use_container_width=True)