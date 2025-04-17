import psycopg2
import os
from datetime import datetime, timedelta

# Database connection function
def get_db_connection():
    """Create a database connection"""
    try:
        # Use environment variables
        db_url = os.getenv("DATABASE_URL", "")
        
        # Connect to database
        if db_url:
            conn = psycopg2.connect(db_url)
        else:
            conn = None
            print("No DATABASE_URL provided")
        
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def count_records():
    """Count records in each database table"""
    conn = get_db_connection()
    if not conn:
        print("Could not connect to database")
        return
    
    try:
        cur = conn.cursor()
        
        # Get list of tables
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        
        tables = cur.fetchall()
        
        print("Database Record Counts:")
        print("========================")
        
        for table in tables:
            table_name = table[0]
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cur.fetchone()[0]
            print(f"{table_name}: {count} records")
            
            # For historical_prices and technical_indicators, show count by symbol
            if table_name in ['historical_prices', 'technical_indicators']:
                cur.execute(f"""
                    SELECT symbol, interval, COUNT(*) 
                    FROM {table_name} 
                    GROUP BY symbol, interval
                """)
                symbol_counts = cur.fetchall()
                for symbol, interval, count in symbol_counts:
                    print(f"  - {symbol} ({interval}): {count} records")
        
        # Check for indicators with no price data
        print("\nChecking for data integrity issues:")
        cur.execute("""
            SELECT ti.symbol, ti.interval, COUNT(ti.*)
            FROM technical_indicators ti
            LEFT JOIN historical_data hp 
                ON ti.symbol = hp.symbol 
                AND ti.interval = hp.interval 
                AND ti.timestamp = hp.timestamp
            WHERE hp.timestamp IS NULL
            GROUP BY ti.symbol, ti.interval
        """)
        
        orphan_indicators = cur.fetchall()
        if orphan_indicators:
            print("Found indicators with no matching price data:")
            for symbol, interval, count in orphan_indicators:
                print(f"  - {symbol} ({interval}): {count} records")
        else:
            print("No indicators found without matching price data.")
        
        # Check for price data with no indicators
        cur.execute("""
            SELECT hp.symbol, hp.interval, COUNT(hp.*)
            FROM historical_data hp
            LEFT JOIN technical_indicators ti 
                ON hp.symbol = ti.symbol 
                AND hp.interval = ti.interval 
                AND hp.timestamp = ti.timestamp
            WHERE ti.timestamp IS NULL
            GROUP BY hp.symbol, hp.interval
        """)
        
        prices_without_indicators = cur.fetchall()
        if prices_without_indicators:
            print("Found price data without indicators:")
            for symbol, interval, count in prices_without_indicators:
                print(f"  - {symbol} ({interval}): {count} records")
        else:
            print("No price data found without matching indicators.")
            
    except Exception as e:
        print(f"Error counting records: {e}")
    finally:
        cur.close()
        conn.close()

def manually_add_indicator_records():
    """Manually add a few indicator records for testing"""
    conn = get_db_connection()
    if not conn:
        print("Could not connect to database")
        return
    
    try:
        cur = conn.cursor()
        
        # First check if we have price data
        cur.execute("SELECT COUNT(*) FROM historical_data WHERE symbol = 'BTCUSDT' AND interval = '4h'")
        price_count = cur.fetchone()[0]
        
        if price_count == 0:
            print("No price data found for BTCUSDT (4h), adding sample price data first")
            
            # Add a few sample price records
            now = datetime.now()
            timestamps = [now - timedelta(hours=i*4) for i in range(10)]
            
            for timestamp in timestamps:
                cur.execute("""
                    INSERT INTO historical_data 
                    (symbol, interval, timestamp, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, interval, timestamp) DO NOTHING
                """, ('BTCUSDT', '4h', timestamp, 50000, 51000, 49000, 50500, 1000))
            
            print(f"Added {len(timestamps)} sample price records")
            
        # Get some timestamp values from the price data
        cur.execute("""
            SELECT timestamp 
            FROM historical_data 
            WHERE symbol = 'BTCUSDT' AND interval = '4h'
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        
        timestamps = [row[0] for row in cur.fetchall()]
        
        if not timestamps:
            print("No timestamps found in price data")
            return
            
        print(f"Adding indicator records for {len(timestamps)} timestamps")
        
        # Add indicator records for these timestamps
        for timestamp in timestamps:
            cur.execute("""
                INSERT INTO technical_indicators 
                (symbol, interval, timestamp, rsi, macd, macd_signal, macd_histogram, 
                 bb_upper, bb_middle, bb_lower, bb_percent, ema_9, buy_signal, sell_signal)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, interval, timestamp) DO UPDATE SET
                rsi = EXCLUDED.rsi,
                macd = EXCLUDED.macd,
                buy_signal = EXCLUDED.buy_signal,
                sell_signal = EXCLUDED.sell_signal
            """, (
                'BTCUSDT', '4h', timestamp, 
                45.5, 100, 95, 5,  # RSI, MACD values
                52000, 50000, 48000, 0.5,  # Bollinger Bands
                50200,  # EMA
                timestamp.hour % 12 == 0,  # Buy signal every 12 hours
                timestamp.hour % 8 == 0   # Sell signal every 8 hours
            ))
            
        conn.commit()
        print(f"Successfully added {len(timestamps)} indicator records to the database")
        
    except Exception as e:
        conn.rollback()
        print(f"Error adding indicator records: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    count_records()
    # Uncomment to manually add indicator records
    # manually_add_indicator_records()
    # count_records()  # Show counts after adding records