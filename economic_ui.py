"""
Economic Indicators UI

This module provides UI components for displaying and analyzing economic indicators
such as US Dollar Index (DXY) and global liquidity metrics in relation to cryptocurrency prices.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Use data utilities module to avoid circular imports
from data_utilities import get_data

from economic_indicators import (
    get_dxy_data, get_liquidity_data, calculate_correlation,
    update_economic_indicators, create_economic_indicator_tables,
    GLOBAL_LIQUIDITY_INDICATORS
)

def render_economic_indicators_tab(crypto_symbol="BTCUSDT"):
    """
    Render the Economic Indicators tab in the UI
    
    Args:
        crypto_symbol: The cryptocurrency symbol to analyze
    """
    st.header("Economic Indicators Analysis")
    
    # Initialize economic indicators tables if needed
    if st.button("Initialize/Update Economic Data", help="Create or update economic indicator data"):
        with st.spinner("Initializing economic indicators..."):
            create_economic_indicator_tables()
            update_success = update_economic_indicators()
            if update_success:
                st.success("Economic indicators initialized and updated successfully!")
            else:
                st.error("There was an error updating economic indicators. Check logs for details.")
    
    # Time period selection
    col1, col2 = st.columns(2)
    with col1:
        period_options = {
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365,
            "3 Years": 1095,
            "5 Years": 1825
        }
        selected_period = st.selectbox("Select Time Period", list(period_options.keys()), index=3)
        days = period_options[selected_period]
    
    with col2:
        end_date = st.date_input("End Date", datetime.now())
        start_date = end_date - timedelta(days=days)
    
    # Convert to datetime
    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.min.time())
    
    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["US Dollar Index", "Global Liquidity", "Correlation Analysis"])
    
    with tab1:
        render_dxy_analysis(crypto_symbol, start_date, end_date)
    
    with tab2:
        render_liquidity_analysis(crypto_symbol, start_date, end_date)
    
    with tab3:
        render_correlation_analysis(crypto_symbol, start_date, end_date)

def render_dxy_analysis(crypto_symbol, start_date, end_date):
    """Render US Dollar Index analysis"""
    st.subheader("US Dollar Index (DXY) Analysis")
    
    # Fetch DXY data
    with st.spinner("Fetching Dollar Index data..."):
        dxy_df = get_dxy_data(start_date, end_date)
    
    if dxy_df.empty:
        st.warning("No Dollar Index data available for the selected period. Try initializing the data or selecting a different period.")
        return
    
    # Get crypto data for comparison
    with st.spinner(f"Fetching {crypto_symbol} data..."):
        crypto_df = get_data(
            crypto_symbol, 
            interval="1d",  # Daily data for comparison with DXY
            start_date=start_date,
            end_date=end_date
        )
    
    if crypto_df is None or crypto_df.empty:
        st.warning(f"No {crypto_symbol} data available for the selected period.")
        
        # Still show DXY data even if crypto data is not available
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dxy_df['timestamp'],
            y=dxy_df['close'],
            mode='lines',
            name='US Dollar Index',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title=f"US Dollar Index (DXY) - {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            xaxis_title="Date",
            yaxis_title="Index Value",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data statistics - convert to float to avoid decimal.Decimal issues
        st.subheader("Dollar Index Statistics")
        st.write(f"Data Points: {len(dxy_df)}")
        st.write(f"Average: {float(dxy_df['close'].astype(float).mean()):.2f}")
        st.write(f"Range: {float(dxy_df['close'].astype(float).min()):.2f} - {float(dxy_df['close'].astype(float).max()):.2f}")
        
        return
    
    # Create dual-axis chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add DXY line
    fig.add_trace(
        go.Scatter(
            x=dxy_df['timestamp'],
            y=dxy_df['close'],
            mode='lines',
            name='US Dollar Index',
            line=dict(color='green')
        ),
        secondary_y=False
    )
    
    # Add crypto price line
    fig.add_trace(
        go.Scatter(
            x=crypto_df['timestamp'],
            y=crypto_df['close'],
            mode='lines',
            name=f'{crypto_symbol} Price',
            line=dict(color='blue')
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=f"{crypto_symbol} vs US Dollar Index (DXY) - {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        xaxis_title="Date",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    # Update axes titles
    fig.update_yaxes(title_text="DXY Value", secondary_y=False)
    fig.update_yaxes(title_text=f"{crypto_symbol} Price (USD)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate and display correlation
    try:
        # Merge dataframes on date
        merged_df = pd.merge(
            dxy_df[['timestamp', 'close']].rename(columns={'close': 'dxy_close'}),
            crypto_df[['timestamp', 'close']].rename(columns={'close': 'crypto_close'}),
            on='timestamp'
        )
        
        if not merged_df.empty:
            correlation = merged_df['dxy_close'].corr(merged_df['crypto_close'])
            
            # Display correlation
            st.subheader("Correlation Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Correlation Coefficient", f"{correlation:.4f}")
                st.write("Interpretation:")
                if correlation > 0.7:
                    st.write("Strong positive correlation")
                elif correlation > 0.3:
                    st.write("Moderate positive correlation")
                elif correlation > -0.3:
                    st.write("Weak or no correlation")
                elif correlation > -0.7:
                    st.write("Moderate negative correlation")
                else:
                    st.write("Strong negative correlation")
            
            with col2:
                st.write("Data points used:", len(merged_df))
                st.write("Period:", f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                
                # Time range filter for viewing specific periods
                st.write("**Insight**: Correlation often changes during different market conditions.")
    except Exception as e:
        st.error(f"Error calculating correlation: {e}")

def render_liquidity_analysis(crypto_symbol, start_date, end_date):
    """Render Global Liquidity analysis"""
    st.subheader("Global Liquidity Analysis")
    
    # Select liquidity indicator
    indicator_options = {
        "M2 Money Supply": "M2",
        "Federal Reserve Balance Sheet": "WALCL",
        "Monetary Base": "BOGMBASE"
    }
    
    selected_indicator = st.selectbox(
        "Select Liquidity Indicator", 
        list(indicator_options.keys())
    )
    
    indicator_code = indicator_options[selected_indicator]
    
    # Fetch liquidity data
    with st.spinner(f"Fetching {selected_indicator} data..."):
        liquidity_df = get_liquidity_data(indicator_code, start_date, end_date)
    
    if liquidity_df.empty:
        st.warning(f"No {selected_indicator} data available for the selected period. Try initializing the data or selecting a different period.")
        
        # Provide information about the indicator
        st.subheader("About this Indicator")
        if indicator_code == "M2":
            st.write("M2 Money Supply includes cash, checking deposits, and easily convertible near money. It is a measure of the money supply that includes all elements of M1 plus 'near money'.")
        elif indicator_code == "WALCL":
            st.write("Federal Reserve Balance Sheet (Total Assets) shows the total assets held by the Federal Reserve. Increases in the balance sheet typically occur during periods of Quantitative Easing.")
        elif indicator_code == "BOGMBASE":
            st.write("Monetary Base is the sum of currency in circulation and reserve balances (deposits held by banks with the Federal Reserve).")
        
        return
    
    # Get crypto data for comparison
    with st.spinner(f"Fetching {crypto_symbol} data..."):
        crypto_df = get_data(
            crypto_symbol, 
            interval="1d",  # Daily data for comparison
            start_date=start_date,
            end_date=end_date
        )
    
    if crypto_df is None or crypto_df.empty:
        st.warning(f"No {crypto_symbol} data available for the selected period.")
        
        # Still show liquidity data even if crypto data is not available
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=liquidity_df['timestamp'],
            y=liquidity_df['value'],
            mode='lines',
            name=selected_indicator,
            line=dict(color='purple')
        ))
        
        fig.update_layout(
            title=f"{selected_indicator} - {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            xaxis_title="Date",
            yaxis_title="Value (in millions of USD)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data statistics - convert to float to avoid decimal.Decimal issues
        st.subheader(f"{selected_indicator} Statistics")
        st.write(f"Data Points: {len(liquidity_df)}")
        st.write(f"Average: ${float(liquidity_df['value'].astype(float).mean()):,.2f} million")
        st.write(f"Range: ${float(liquidity_df['value'].astype(float).min()):,.2f} - ${float(liquidity_df['value'].astype(float).max()):,.2f} million")
        
        # Provide information about the indicator
        st.subheader("About this Indicator")
        if indicator_code == "M2":
            st.write("M2 Money Supply includes cash, checking deposits, and easily convertible near money. It is a measure of the money supply that includes all elements of M1 plus 'near money'.")
        elif indicator_code == "WALCL":
            st.write("Federal Reserve Balance Sheet (Total Assets) shows the total assets held by the Federal Reserve. Increases in the balance sheet typically occur during periods of Quantitative Easing.")
        elif indicator_code == "BOGMBASE":
            st.write("Monetary Base is the sum of currency in circulation and reserve balances (deposits held by banks with the Federal Reserve).")
        
        return
    
    # Create dual-axis chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add liquidity line
    fig.add_trace(
        go.Scatter(
            x=liquidity_df['timestamp'],
            y=liquidity_df['value'],
            mode='lines',
            name=selected_indicator,
            line=dict(color='purple')
        ),
        secondary_y=False
    )
    
    # Add crypto price line
    fig.add_trace(
        go.Scatter(
            x=crypto_df['timestamp'],
            y=crypto_df['close'],
            mode='lines',
            name=f'{crypto_symbol} Price',
            line=dict(color='blue')
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=f"{crypto_symbol} vs {selected_indicator} - {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        xaxis_title="Date",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    # Update axes titles
    fig.update_yaxes(title_text=f"{selected_indicator} (millions USD)", secondary_y=False)
    fig.update_yaxes(title_text=f"{crypto_symbol} Price (USD)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate and display correlation
    try:
        # Prepare data for correlation
        # First resample crypto data to monthly (to match liquidity data frequency)
        crypto_df['timestamp'] = pd.to_datetime(crypto_df['timestamp'])
        crypto_df.set_index('timestamp', inplace=True)
        monthly_crypto = crypto_df['close'].resample('M').mean().reset_index()
        
        # Ensure liquidity_df timestamp is datetime
        liquidity_df['timestamp'] = pd.to_datetime(liquidity_df['timestamp'])
        
        # Merge on month
        monthly_crypto['month'] = monthly_crypto['timestamp'].dt.to_period('M')
        liquidity_df['month'] = liquidity_df['timestamp'].dt.to_period('M')
        
        merged_df = pd.merge(
            monthly_crypto[['month', 'close']].rename(columns={'close': 'crypto_close'}),
            liquidity_df[['month', 'value']].rename(columns={'value': 'liquidity_value'}),
            on='month'
        )
        
        if not merged_df.empty and len(merged_df) >= 3:  # Need at least a few data points
            # Convert to float to avoid Decimal type issues
            correlation = merged_df['liquidity_value'].astype(float).corr(merged_df['crypto_close'].astype(float))
            
            # Display correlation
            st.subheader("Correlation Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Correlation Coefficient", f"{correlation:.4f}")
                st.write("Interpretation:")
                if correlation > 0.7:
                    st.write("Strong positive correlation")
                elif correlation > 0.3:
                    st.write("Moderate positive correlation")
                elif correlation > -0.3:
                    st.write("Weak or no correlation")
                elif correlation > -0.7:
                    st.write("Moderate negative correlation")
                else:
                    st.write("Strong negative correlation")
            
            with col2:
                st.write("Data points used:", len(merged_df))
                st.write("Period:", f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                st.write("Note: Data was resampled to monthly frequency for comparison")
        else:
            st.warning("Not enough overlapping data points to calculate correlation")
    except Exception as e:
        st.error(f"Error calculating correlation: {e}")
    
    # Provide information about the indicator
    st.subheader("About this Indicator")
    if indicator_code == "M2":
        st.write("M2 Money Supply includes cash, checking deposits, and easily convertible near money. It is a measure of the money supply that includes all elements of M1 plus 'near money'.")
        st.write("Historical correlation with crypto: Often positive, as increased money supply may lead to inflation concerns and drive investment in alternative assets.")
    elif indicator_code == "WALCL":
        st.write("Federal Reserve Balance Sheet (Total Assets) shows the total assets held by the Federal Reserve. Increases in the balance sheet typically occur during periods of Quantitative Easing.")
        st.write("Historical correlation with crypto: Generally positive during QE periods, as increased liquidity tends to flow into risk assets including cryptocurrencies.")
    elif indicator_code == "BOGMBASE":
        st.write("Monetary Base is the sum of currency in circulation and reserve balances (deposits held by banks with the Federal Reserve).")
        st.write("Historical correlation with crypto: Can signal changes in monetary policy that may impact cryptocurrency values.")

def render_correlation_analysis(crypto_symbol, start_date, end_date):
    """Render correlation analysis between economic indicators and cryptocurrency"""
    st.subheader("Multi-Indicator Correlation Analysis")
    
    # Fetch DXY data
    with st.spinner("Fetching economic indicators..."):
        dxy_df = get_dxy_data(start_date, end_date)
        
        # Fetch liquidity data
        liquidity_data = {}
        for name, code in GLOBAL_LIQUIDITY_INDICATORS.items():
            liquidity_data[name] = get_liquidity_data(code, start_date, end_date)
    
    # Get crypto data
    with st.spinner(f"Fetching {crypto_symbol} data..."):
        crypto_df = get_data(
            crypto_symbol, 
            interval="1d",
            start_date=start_date,
            end_date=end_date
        )
    
    if crypto_df is None or crypto_df.empty:
        st.warning(f"No {crypto_symbol} data available for the selected period.")
        return
    
    # Create correlation matrix
    correlations = {}
    
    # DXY correlation
    if not dxy_df.empty:
        dxy_corr = calculate_correlation(dxy_df, crypto_df)
        if dxy_corr is not None:
            correlations["US Dollar Index (DXY)"] = dxy_corr
    
    # Liquidity correlations
    for name, df in liquidity_data.items():
        if not df.empty:
            liq_corr = calculate_correlation(df, crypto_df, resample='M')
            if liq_corr is not None:
                correlations[name] = liq_corr
    
    if not correlations:
        st.warning("No correlation data could be calculated for the selected period.")
        return
    
    # Create correlation chart
    fig = go.Figure()
    
    for name, corr in correlations.items():
        fig.add_trace(go.Bar(
            x=[name],
            y=[corr],
            name=name
        ))
    
    fig.update_layout(
        title=f"{crypto_symbol} Correlation with Economic Indicators",
        xaxis_title="Economic Indicator",
        yaxis_title="Correlation Coefficient",
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show correlation table
    st.subheader("Correlation Summary")
    
    # Create a DataFrame for better display
    corr_df = pd.DataFrame({
        'Indicator': list(correlations.keys()),
        'Correlation': list(correlations.values())
    })
    
    # Sort by absolute correlation
    corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False).drop('Abs_Correlation', axis=1)
    
    # Add interpretation
    def interpret_correlation(corr):
        if corr > 0.7:
            return "Strong positive"
        elif corr > 0.3:
            return "Moderate positive"
        elif corr > -0.3:
            return "Weak or none"
        elif corr > -0.7:
            return "Moderate negative"
        else:
            return "Strong negative"
    
    corr_df['Interpretation'] = corr_df['Correlation'].apply(interpret_correlation)
    
    # Format correlation values
    corr_df['Correlation'] = corr_df['Correlation'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(corr_df)
    
    # Add explanatory text
    st.subheader("Understanding the Analysis")
    st.write("""
    This correlation analysis examines the relationship between various economic indicators and cryptocurrency prices. 
    A positive correlation suggests that the indicator and cryptocurrency prices tend to move in the same direction, 
    while a negative correlation indicates they move in opposite directions.
    
    Key insights:
    - DXY is often negatively correlated with crypto (dollar strength = crypto weakness)
    - Liquidity metrics typically show positive correlation with crypto markets
    - Correlations may change dramatically during different market regimes
    """)