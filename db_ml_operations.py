"""
ML Database Operations
This module provides functions to save and retrieve ML-related data from the database.
These include: ML predictions, ML model performance metrics, detected patterns, and news with sentiment analysis.
"""

import psycopg2
import pandas as pd
from datetime import datetime
from psycopg2 import extras

# Import the database connection function
from database import get_db_connection, execute_sql_to_df


def save_news_article(symbol, title, content, source, published_at, url=None, author=None, sentiment_score=None, relevance_score=None):
    """Save a news article to the database"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        query = """
        INSERT INTO news_data 
        (symbol, title, content, source, url, author, published_at, sentiment_score, relevance_score)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (source, title, published_at) 
        DO UPDATE SET
            symbol = EXCLUDED.symbol,
            content = EXCLUDED.content,
            url = EXCLUDED.url,
            author = EXCLUDED.author,
            sentiment_score = EXCLUDED.sentiment_score,
            relevance_score = EXCLUDED.relevance_score,
            created_at = CURRENT_TIMESTAMP
        RETURNING id
        """
        
        cur.execute(query, (
            symbol, 
            title, 
            content, 
            source, 
            url, 
            author, 
            published_at,
            sentiment_score,
            relevance_score
        ))
        news_id = cur.fetchone()[0]
        conn.commit()
        
        return news_id
    except psycopg2.Error as e:
        print(f"Error saving news article: {e}")
        return False
    finally:
        if conn:
            conn.close()


def get_news_data(symbol=None, sources=None, start_time=None, end_time=None, limit=100):
    """Get news data from database"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        # Base query
        query = """
        SELECT symbol, title, content, source, url, author, published_at, sentiment_score, relevance_score
        FROM news_data
        WHERE 1=1
        """
        
        params = []
        
        # Add symbol filter if specified
        if symbol:
            query += " AND symbol = %s"
            params.append(symbol)
        
        # Add source filter if specified
        if sources:
            if isinstance(sources, str):
                sources = [sources]
            placeholders = ', '.join(['%s'] * len(sources))
            query += f" AND source IN ({placeholders})"
            params.extend(sources)
        
        # Add time range if specified
        if start_time:
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            query += " AND published_at >= %s"
            params.append(start_time)
        
        if end_time:
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            query += " AND published_at <= %s"
            params.append(end_time)
        
        # Add order by and limit
        query += " ORDER BY published_at DESC LIMIT %s"
        params.append(limit)
        
        # Use custom function instead of pandas read_sql_query to avoid SQLAlchemy dependency
        df = execute_sql_to_df(query, conn, params=tuple(params))
        
        return df
    except psycopg2.Error as e:
        print(f"Error fetching news data: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()


def save_ml_prediction(symbol, interval, prediction_timestamp, target_timestamp, model_name, 
                       predicted_price, predicted_change_pct, confidence_score, 
                       features_used=None, prediction_type='price'):
    """Save a machine learning prediction to the database"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        # Convert features_used to JSON if it's a dictionary
        if features_used is None:
            features_used = {}
            
        query = """
        INSERT INTO ml_predictions 
        (symbol, interval, prediction_timestamp, target_timestamp, model_name, 
         predicted_price, predicted_change_pct, confidence_score, features_used, prediction_type)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, interval, prediction_timestamp, target_timestamp, model_name) 
        DO UPDATE SET
            predicted_price = EXCLUDED.predicted_price,
            predicted_change_pct = EXCLUDED.predicted_change_pct,
            confidence_score = EXCLUDED.confidence_score,
            features_used = EXCLUDED.features_used,
            prediction_type = EXCLUDED.prediction_type,
            created_at = CURRENT_TIMESTAMP
        RETURNING id
        """
        
        cur.execute(query, (
            symbol,
            interval,
            prediction_timestamp,
            target_timestamp,
            model_name,
            predicted_price,
            predicted_change_pct,
            confidence_score,
            extras.Json(features_used),
            prediction_type
        ))
        prediction_id = cur.fetchone()[0]
        conn.commit()
        
        return prediction_id
    except psycopg2.Error as e:
        print(f"Error saving ML prediction: {e}")
        return False
    finally:
        if conn:
            conn.close()


def get_ml_predictions(symbol, interval, start_time=None, end_time=None, model_name=None, limit=100):
    """Get ML predictions from database"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        # Base query
        query = """
        SELECT symbol, interval, prediction_timestamp, target_timestamp, model_name,
               predicted_price, predicted_change_pct, confidence_score, features_used,
               prediction_type, created_at
        FROM ml_predictions
        WHERE symbol = %s
        AND interval = %s
        """
        
        params = [symbol, interval]
        
        # Add time range if specified
        if start_time:
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            query += " AND prediction_timestamp >= %s"
            params.append(start_time)
        
        if end_time:
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            query += " AND prediction_timestamp <= %s"
            params.append(end_time)
        
        # Add model name filter if specified
        if model_name:
            query += " AND model_name = %s"
            params.append(model_name)
        
        # Add order by and limit
        query += " ORDER BY prediction_timestamp DESC LIMIT %s"
        params.append(limit)
        
        # Use custom function instead of pandas read_sql_query to avoid SQLAlchemy dependency
        df = execute_sql_to_df(query, conn, params=tuple(params))
        
        return df
    except psycopg2.Error as e:
        print(f"Error fetching ML predictions: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()


def save_ml_model_performance(model_name, symbol, interval, training_timestamp, accuracy, precision_score, 
                             recall, f1_score, mse, mae, training_params=None):
    """Save machine learning model performance metrics"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        # Convert training_params to JSON if it's a dictionary
        if training_params is None:
            training_params = {}
            
        query = """
        INSERT INTO ml_model_performance 
        (model_name, symbol, interval, training_timestamp, accuracy, precision, recall, 
         f1_score, mse, mae, training_params)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (model_name, symbol, interval, training_timestamp) 
        DO UPDATE SET
            accuracy = EXCLUDED.accuracy,
            precision = EXCLUDED.precision,
            recall = EXCLUDED.recall,
            f1_score = EXCLUDED.f1_score,
            mse = EXCLUDED.mse,
            mae = EXCLUDED.mae,
            training_params = EXCLUDED.training_params,
            created_at = CURRENT_TIMESTAMP
        RETURNING id
        """
        
        cur.execute(query, (
            model_name,
            symbol,
            interval,
            training_timestamp,
            accuracy,
            precision_score,
            recall,
            f1_score,
            mse,
            mae,
            extras.Json(training_params)
        ))
        performance_id = cur.fetchone()[0]
        conn.commit()
        
        return performance_id
    except psycopg2.Error as e:
        print(f"Error saving ML model performance: {e}")
        return False
    finally:
        if conn:
            conn.close()


def save_detected_pattern(symbol, interval, timestamp, pattern_type, pattern_strength, 
                         expected_outcome, confidence_score, description=None, detection_timestamp=None):
    """Save a detected pattern to the database"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        # Set detection timestamp to now if not provided
        if detection_timestamp is None:
            detection_timestamp = datetime.now()
            
        query = """
        INSERT INTO detected_patterns 
        (symbol, interval, timestamp, detection_timestamp, pattern_type, pattern_strength, 
         expected_outcome, confidence_score, description)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, interval, timestamp, pattern_type) 
        DO UPDATE SET
            detection_timestamp = EXCLUDED.detection_timestamp,
            pattern_strength = EXCLUDED.pattern_strength,
            expected_outcome = EXCLUDED.expected_outcome,
            confidence_score = EXCLUDED.confidence_score,
            description = EXCLUDED.description,
            created_at = CURRENT_TIMESTAMP
        RETURNING id
        """
        
        cur.execute(query, (
            symbol,
            interval,
            timestamp,
            detection_timestamp,
            pattern_type,
            pattern_strength,
            expected_outcome,
            confidence_score,
            description
        ))
        pattern_id = cur.fetchone()[0]
        conn.commit()
        
        return pattern_id
    except psycopg2.Error as e:
        print(f"Error saving detected pattern: {e}")
        return False
    finally:
        if conn:
            conn.close()


def get_detected_patterns(symbol=None, interval=None, pattern_types=None, min_strength=0.0, 
                         start_time=None, end_time=None, limit=100):
    """Get detected patterns from database"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        # Base query
        query = """
        SELECT symbol, interval, timestamp, detection_timestamp, pattern_type, 
               pattern_strength, expected_outcome, confidence_score, description
        FROM detected_patterns
        WHERE pattern_strength >= %s
        """
        
        params = [min_strength]
        
        # Add symbol and interval filters if specified
        if symbol:
            query += " AND symbol = %s"
            params.append(symbol)
            
        if interval:
            query += " AND interval = %s"
            params.append(interval)
        
        # Add pattern type filter if specified
        if pattern_types:
            if isinstance(pattern_types, str):
                pattern_types = [pattern_types]
            placeholders = ', '.join(['%s'] * len(pattern_types))
            query += f" AND pattern_type IN ({placeholders})"
            params.extend(pattern_types)
        
        # Add time range if specified
        if start_time:
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            query += " AND timestamp >= %s"
            params.append(start_time)
        
        if end_time:
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            query += " AND timestamp <= %s"
            params.append(end_time)
        
        # Add order by and limit
        query += " ORDER BY timestamp DESC LIMIT %s"
        params.append(limit)
        
        # Use custom function instead of pandas read_sql_query to avoid SQLAlchemy dependency
        df = execute_sql_to_df(query, conn, params=tuple(params))
        
        return df
    except psycopg2.Error as e:
        print(f"Error fetching detected patterns: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()