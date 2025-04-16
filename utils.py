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
