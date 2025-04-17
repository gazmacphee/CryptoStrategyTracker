from datetime import datetime, timedelta

def timeframe_to_seconds(timeframe):
    """Convert timeframe string to seconds"""
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    
    if unit == 'm':
        return value * 60
    elif unit == 'h':
        return value * 3600
    elif unit == 'd':
        return value * 86400
    elif unit == 'w':
        return value * 604800
    else:
        return 3600  # Default to 1 hour

def timeframe_to_interval(timeframe):
    """Convert timeframe to Binance interval format"""
    # Map of common timeframes to Binance interval format
    timeframe_map = {
        '1m': '1m',
        '3m': '3m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
        '1h': '1h',
        '2h': '2h',
        '4h': '4h',
        '6h': '6h',
        '8h': '8h',
        '12h': '12h',
        '1d': '1d',
        '3d': '3d',
        '1w': '1w',
        '1M': '1M'
    }
    
    return timeframe_map.get(timeframe, '1h')

def get_timeframe_options():
    """Return dictionary of timeframe options for UI"""
    return {
        '1m': '1 Minute',
        '5m': '5 Minutes',
        '15m': '15 Minutes',
        '30m': '30 Minutes',
        '1h': '1 Hour',
        '4h': '4 Hours',
        '1d': '1 Day',
        '1w': '1 Week',
    }

def calculate_trade_statistics(trades):
    """Calculate statistics from a list of trades"""
    if not trades:
        return {
            'num_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0
        }
    
    # Filter completed trades
    completed_trades = [t for t in trades if t.get('profit_pct') is not None]
    
    if not completed_trades:
        return {
            'num_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0
        }
    
    # Calculate statistics
    num_trades = len(completed_trades)
    winning_trades = [t for t in completed_trades if t['profit_pct'] > 0]
    losing_trades = [t for t in completed_trades if t['profit_pct'] <= 0]
    
    win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0
    
    # Calculate profit metrics
    avg_profit = sum(t['profit_pct'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(t['profit_pct'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
    
    total_profit = sum(t['profit_pct'] for t in winning_trades) if winning_trades else 0
    total_loss = abs(sum(t['profit_pct'] for t in losing_trades)) if losing_trades else 0
    
    profit_factor = total_profit / total_loss if total_loss > 0 else 0 if total_profit == 0 else float('inf')
    
    # Calculate drawdown
    equity_curve = []
    current_value = 100  # Start with 100 units
    
    for trade in completed_trades:
        current_value *= (1 + trade['profit_pct'] / 100)
        equity_curve.append(current_value)
    
    if equity_curve:
        # Calculate maximum drawdown
        max_value = equity_curve[0]
        max_drawdown = 0
        
        for value in equity_curve:
            max_value = max(max_value, value)
            drawdown = (max_value - value) / max_value * 100
            max_drawdown = max(max_drawdown, drawdown)
    else:
        max_drawdown = 0
    
    return {
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown
    }

def calculate_portfolio_value(portfolio_df, price_data):
    """
    Calculate current portfolio value based on latest price data
    
    Args:
        portfolio_df: DataFrame with portfolio holdings (symbol, quantity, purchase_price, purchase_date)
        price_data: Dictionary with latest prices {'BTCUSDT': 60000.0, 'ETHUSDT': 2500.0, ...}
    
    Returns:
        Dictionary with portfolio metrics
    """
    import pandas as pd
    from datetime import datetime
    
    if portfolio_df.empty:
        return {
            "total_value": 0,
            "total_cost": 0,
            "total_profit_loss": 0,
            "total_profit_loss_percent": 0,
            "assets": []
        }
    
    # Calculate current value and profit/loss for each holding
    results = []
    total_value = 0
    total_cost = 0
    
    for _, row in portfolio_df.iterrows():
        symbol = row['symbol']
        quantity = float(row['quantity'])
        purchase_price = float(row['purchase_price'])
        purchase_date = row['purchase_date']
        
        # Current price (default to purchase price if not available)
        current_price = price_data.get(symbol, purchase_price)
        
        # Calculate values
        cost_basis = quantity * purchase_price
        current_value = quantity * current_price
        profit_loss = current_value - cost_basis
        profit_loss_percent = (profit_loss / cost_basis) * 100 if cost_basis > 0 else 0
        
        # Days held
        days_held = (datetime.now() - purchase_date).days
        
        # Annualized return
        if days_held > 0:
            annualized_return = ((1 + (profit_loss_percent / 100)) ** (365 / days_held) - 1) * 100
        else:
            annualized_return = 0
        
        # Add to totals
        total_value += current_value
        total_cost += cost_basis
        
        # Add to results
        results.append({
            "id": row.get('id'),
            "symbol": symbol,
            "quantity": quantity,
            "purchase_price": purchase_price,
            "purchase_date": purchase_date,
            "current_price": current_price,
            "cost_basis": cost_basis,
            "current_value": current_value,
            "profit_loss": profit_loss,
            "profit_loss_percent": profit_loss_percent,
            "days_held": days_held,
            "annualized_return": annualized_return,
            "notes": row.get('notes', '')
        })
    
    # Calculate portfolio totals
    total_profit_loss = total_value - total_cost
    total_profit_loss_percent = (total_profit_loss / total_cost) * 100 if total_cost > 0 else 0
    
    return {
        "total_value": total_value,
        "total_cost": total_cost,
        "total_profit_loss": total_profit_loss,
        "total_profit_loss_percent": total_profit_loss_percent,
        "assets": results
    }

def get_portfolio_performance_history(portfolio_df, symbol_price_history):
    """
    Calculate historical portfolio performance over time
    
    Args:
        portfolio_df: DataFrame with portfolio holdings (symbol, quantity, purchase_price, purchase_date)
        symbol_price_history: Dictionary with price history per symbol {'BTCUSDT': dataframe, 'ETHUSDT': dataframe, ...}
        
    Returns:
        DataFrame with portfolio value over time
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    if portfolio_df.empty:
        return pd.DataFrame(columns=['timestamp', 'value'])
    
    # Get unique dates across all price histories
    all_dates = set()
    for symbol, history_df in symbol_price_history.items():
        if not history_df.empty:
            all_dates.update(history_df['timestamp'].tolist())
    
    # Convert to sorted list
    all_dates = sorted(all_dates)
    
    if not all_dates:
        return pd.DataFrame(columns=['timestamp', 'value'])
    
    # Create results dataframe
    results = pd.DataFrame({'timestamp': all_dates})
    results['value'] = 0
    
    # For each portfolio item, calculate value at each date
    for _, asset in portfolio_df.iterrows():
        symbol = asset['symbol']
        quantity = float(asset['quantity'])
        purchase_date = asset['purchase_date']
        
        # Get price history for this symbol
        history_df = symbol_price_history.get(symbol)
        
        if history_df is not None and not history_df.empty:
            # Filter to dates after purchase
            history_df = history_df[history_df['timestamp'] >= purchase_date]
            
            # Merge with results
            if not history_df.empty:
                value_df = pd.DataFrame({
                    'timestamp': history_df['timestamp'],
                    f'value_{symbol}': history_df['close'] * quantity
                })
                
                results = pd.merge(results, value_df, on='timestamp', how='left')
                results[f'value_{symbol}'] = results[f'value_{symbol}'].fillna(0)
                results['value'] += results[f'value_{symbol}']
    
    # Drop any value columns except the total
    value_cols = [col for col in results.columns if col.startswith('value_')]
    results = results.drop(columns=value_cols)
    
    # Sort by timestamp
    results = results.sort_values('timestamp').reset_index(drop=True)
    
    return results

def get_benchmark_performance(benchmark_df, start_date, end_date=None):
    """
    Calculate benchmark performance over a specific time period
    
    Args:
        benchmark_df: DataFrame with benchmark data (timestamp, value)
        start_date: Start date for performance calculation
        end_date: End date for performance calculation (defaults to latest date in data)
    
    Returns:
        Dictionary with benchmark performance metrics
    """
    import pandas as pd
    
    if benchmark_df.empty:
        return {
            "start_value": 0,
            "end_value": 0,
            "percent_change": 0,
            "values": []
        }
    
    # Filter data to requested date range
    if end_date is None:
        end_date = benchmark_df['timestamp'].max()
    
    filtered_df = benchmark_df[(benchmark_df['timestamp'] >= start_date) & 
                               (benchmark_df['timestamp'] <= end_date)]
    
    if filtered_df.empty:
        return {
            "start_value": 0,
            "end_value": 0,
            "percent_change": 0,
            "values": []
        }
    
    # Get start and end values
    start_value = filtered_df.iloc[0]['value']
    end_value = filtered_df.iloc[-1]['value']
    
    # Calculate performance
    percent_change = ((end_value / start_value) - 1) * 100 if start_value > 0 else 0
    
    # Prepare data points for charting
    values = filtered_df.to_dict('records')
    
    return {
        "start_value": start_value,
        "end_value": end_value,
        "percent_change": percent_change,
        "values": values
    }

def normalize_comparison_data(portfolio_data, benchmark_data):
    """
    Normalize portfolio and benchmark data for comparison charting
    
    Args:
        portfolio_data: DataFrame with portfolio value over time (timestamp, value)
        benchmark_data: DataFrame with benchmark value over time (timestamp, value)
    
    Returns:
        Dictionary with normalized values for both datasets
    """
    import pandas as pd
    
    if (portfolio_data.empty if hasattr(portfolio_data, 'empty') else not portfolio_data) or \
       (benchmark_data.empty if hasattr(benchmark_data, 'empty') else not benchmark_data):
        return {
            "portfolio": [],
            "benchmark": []
        }
    
    # Convert to DataFrame if needed
    if isinstance(portfolio_data, list):
        portfolio_df = pd.DataFrame(portfolio_data)
    else:
        portfolio_df = portfolio_data
        
    if isinstance(benchmark_data, list):
        benchmark_df = pd.DataFrame(benchmark_data)
    else:
        benchmark_df = benchmark_data
    
    # Get initial values
    portfolio_start = portfolio_df.iloc[0]['value'] if not portfolio_df.empty else 1
    benchmark_start = benchmark_df.iloc[0]['value'] if not benchmark_df.empty else 1
    
    # Normalize data (starting at 100)
    if not portfolio_df.empty and 'value' in portfolio_df.columns:
        portfolio_df['normalized'] = (portfolio_df['value'] / portfolio_start) * 100
        
    if not benchmark_df.empty and 'value' in benchmark_df.columns:
        benchmark_df['normalized'] = (benchmark_df['value'] / benchmark_start) * 100
    
    return {
        "portfolio": portfolio_df.to_dict('records') if not portfolio_df.empty else [],
        "benchmark": benchmark_df.to_dict('records') if not benchmark_df.empty else []
    }
