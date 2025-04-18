"""
Local PostgreSQL database checker and setup script.
This script checks if the crypto database exists on the local PostgreSQL server,
creates it if it doesn't exist, and ensures all required tables are created.
"""

import psycopg2
import sys
import os

# Local PostgreSQL connection parameters
DB_USER = "postgres"
DB_PASS = "2212"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "crypto"

def check_db_exists():
    """Check if the crypto database exists on the local PostgreSQL server"""
    try:
        # Connect to the default 'postgres' database to check if our database exists
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database="postgres",  # Connect to default database to check other databases
            user=DB_USER,
            password=DB_PASS
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if the crypto database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
        exists = cursor.fetchone()
        
        if not exists:
            print(f"Database '{DB_NAME}' does not exist. Creating it...")
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            print(f"Database '{DB_NAME}' created successfully.")
        else:
            print(f"Database '{DB_NAME}' already exists.")
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error checking database existence: {e}")
        return False

def create_tables():
    """Create all required tables if they don't exist"""
    try:
        # Connect to the crypto database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Create historical_data table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS historical_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            interval VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            close NUMERIC NOT NULL,
            volume NUMERIC NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, interval, timestamp)
        );
        """)
        
        # Create index for faster queries
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_historical_data_symbol_interval_timestamp 
        ON historical_data(symbol, interval, timestamp);
        """)
        
        # Create technical_indicators table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS technical_indicators (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            interval VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            rsi NUMERIC,
            macd NUMERIC,
            macd_signal NUMERIC,
            macd_histogram NUMERIC,
            bb_upper NUMERIC,
            bb_middle NUMERIC,
            bb_lower NUMERIC,
            bb_percent NUMERIC,
            ema_9 NUMERIC,
            ema_21 NUMERIC,
            ema_50 NUMERIC,
            ema_200 NUMERIC,
            buy_signal BOOLEAN DEFAULT FALSE,
            sell_signal BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, interval, timestamp)
        );
        """)
        
        # Create index for indicators table
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_indicators_symbol_interval_timestamp 
        ON technical_indicators(symbol, interval, timestamp);
        """)
        
        # Create trades table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            interval VARCHAR(10) NOT NULL,
            type VARCHAR(10) NOT NULL,  -- BUY, SELL, SELL (EOT)
            entry_timestamp TIMESTAMP,
            exit_timestamp TIMESTAMP,
            entry_price NUMERIC,
            exit_price NUMERIC,
            quantity NUMERIC,
            profit NUMERIC,
            profit_pct NUMERIC,
            holding_time_hours NUMERIC,
            trade_status VARCHAR(10) NOT NULL, -- OPEN, CLOSED
            strategy_params JSONB,  -- Stores the strategy parameters used
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Create index for trades table
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_trades_symbol_interval 
        ON trades(symbol, interval);
        """)
        
        # Create portfolio table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            quantity NUMERIC NOT NULL,
            purchase_price NUMERIC NOT NULL,
            purchase_date TIMESTAMP NOT NULL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, purchase_date)
        );
        """)
        
        # Create index for portfolio table
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_portfolio_symbol 
        ON portfolio(symbol);
        """)
        
        # Create benchmarks table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS benchmarks (
            id SERIAL PRIMARY KEY,
            name VARCHAR(50) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            value NUMERIC NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name, timestamp)
        );
        """)
        
        # Create index for benchmarks table
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_benchmarks_name_timestamp 
        ON benchmarks(name, timestamp);
        """)
        
        # Create sentiment_data table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            source VARCHAR(50) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            sentiment_score NUMERIC NOT NULL,
            volume INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, source, timestamp)
        );
        """)
        
        # Create index for sentiment data
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_source_timestamp
        ON sentiment_data(symbol, source, timestamp);
        """)
        
        print("All tables and indexes created successfully")
        
        # Grant permissions to postgres user
        cursor.execute(f"GRANT ALL PRIVILEGES ON DATABASE {DB_NAME} TO {DB_USER}")
        cursor.execute(f"GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO {DB_USER}")
        cursor.execute(f"GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO {DB_USER}")
        print(f"Granted all privileges to user '{DB_USER}'")
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating tables: {e}")
        return False

def update_env_file():
    """Update the .env file with the local PostgreSQL connection information"""
    try:
        # Create a PostgreSQL connection string
        db_url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        
        # Check if .env file exists
        if os.path.exists(".env"):
            # Read the current .env file
            with open(".env", "r") as f:
                env_lines = f.readlines()
            
            # Update DATABASE_URL and PostgreSQL settings
            updated_lines = []
            db_url_updated = False
            pg_settings_updated = {
                "PGHOST": False,
                "PGPORT": False,
                "PGUSER": False,
                "PGPASSWORD": False,
                "PGDATABASE": False
            }
            
            for line in env_lines:
                if line.startswith("DATABASE_URL="):
                    updated_lines.append(f"DATABASE_URL={db_url}\n")
                    db_url_updated = True
                elif line.startswith("PGHOST="):
                    updated_lines.append(f"PGHOST={DB_HOST}\n")
                    pg_settings_updated["PGHOST"] = True
                elif line.startswith("PGPORT="):
                    updated_lines.append(f"PGPORT={DB_PORT}\n")
                    pg_settings_updated["PGPORT"] = True
                elif line.startswith("PGUSER="):
                    updated_lines.append(f"PGUSER={DB_USER}\n")
                    pg_settings_updated["PGUSER"] = True
                elif line.startswith("PGPASSWORD="):
                    updated_lines.append(f"PGPASSWORD={DB_PASS}\n")
                    pg_settings_updated["PGPASSWORD"] = True
                elif line.startswith("PGDATABASE="):
                    updated_lines.append(f"PGDATABASE={DB_NAME}\n")
                    pg_settings_updated["PGDATABASE"] = True
                else:
                    updated_lines.append(line)
            
            # Add any missing settings
            if not db_url_updated:
                updated_lines.append(f"DATABASE_URL={db_url}\n")
            
            for key, updated in pg_settings_updated.items():
                if not updated:
                    if key == "PGHOST":
                        updated_lines.append(f"PGHOST={DB_HOST}\n")
                    elif key == "PGPORT":
                        updated_lines.append(f"PGPORT={DB_PORT}\n")
                    elif key == "PGUSER":
                        updated_lines.append(f"PGUSER={DB_USER}\n")
                    elif key == "PGPASSWORD":
                        updated_lines.append(f"PGPASSWORD={DB_PASS}\n")
                    elif key == "PGDATABASE":
                        updated_lines.append(f"PGDATABASE={DB_NAME}\n")
            
            # Write the updated .env file
            with open(".env", "w") as f:
                f.writelines(updated_lines)
        else:
            # Create a new .env file
            with open(".env", "w") as f:
                f.write(f"# Database configuration\n")
                f.write(f"PGHOST={DB_HOST}\n")
                f.write(f"PGPORT={DB_PORT}\n")
                f.write(f"PGUSER={DB_USER}\n")
                f.write(f"PGPASSWORD={DB_PASS}\n")
                f.write(f"PGDATABASE={DB_NAME}\n")
                f.write(f"DATABASE_URL={db_url}\n\n")
                f.write(f"# Binance API credentials (optional but recommended for real data)\n")
                f.write(f"BINANCE_API_KEY=your_binance_api_key\n")
                f.write(f"BINANCE_API_SECRET=your_binance_api_secret\n\n")
                f.write(f"# OpenAI API key (required for news digest and sentiment analysis features)\n")
                f.write(f"OPENAI_API_KEY=your_openai_api_key\n\n")
                f.write(f"# Application settings\n")
                f.write(f"RESET_DB=false  # Set to true to reset database on startup\n")
                f.write(f"BACKFILL_ON_START=true  # Set to true to start data backfill on startup\n")
        
        print(".env file updated with local PostgreSQL connection information")
        return True
    except Exception as e:
        print(f"Error updating .env file: {e}")
        return False

def check_tables():
    """Check if all required tables exist in the database"""
    try:
        # Connect to the crypto database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # List of required tables
        required_tables = [
            "historical_data",
            "technical_indicators",
            "trades",
            "portfolio",
            "benchmarks",
            "sentiment_data"
        ]
        
        # Check each table
        for table in required_tables:
            cursor.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table}')")
            exists = cursor.fetchone()[0]
            if exists:
                print(f"Table '{table}' exists")
            else:
                print(f"Table '{table}' does not exist")
                return False
        
        print("All required tables exist")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error checking tables: {e}")
        return False

def main():
    print("Checking local PostgreSQL database setup...")
    
    # Step 1: Check if database exists, create if not
    if not check_db_exists():
        print("Failed to check/create database. Please ensure PostgreSQL is running.")
        return
    
    # Step 2: Create all required tables
    if not create_tables():
        print("Failed to create tables. Please check PostgreSQL connection.")
        return
    
    # Step 3: Check if all tables exist
    if not check_tables():
        print("Not all tables were created successfully.")
        return
    
    # Step 4: Update the .env file with local connection info
    if not update_env_file():
        print("Failed to update .env file.")
        return
    
    print("\nLocal PostgreSQL database setup is complete!")
    print(f"You can now connect to: postgresql://{DB_USER}:***@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    print("\nThe connection details have been updated in your .env file.")
    print("You can now run the application with: streamlit run app.py")

if __name__ == "__main__":
    main()