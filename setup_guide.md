# Local Setup Guide for Crypto Trading Analysis Platform

## Prerequisites

1. **Python 3.8 or higher**
   - Download from [python.org](https://www.python.org/downloads/)

2. **PostgreSQL Database**
   - Download from [postgresql.org](https://www.postgresql.org/download/)
   - Create a database named `crypto_trading`
   - Note your database credentials (username, password, host, port)

3. **Git** (optional, for cloning the repository)
   - Download from [git-scm.com](https://git-scm.com/downloads)

## Installation Steps

1. **Clone or Download the Project**
   ```bash
   git clone <repository-url>
   # OR download and extract the ZIP file
   ```

2. **Create a Virtual Environment** (recommended)
   ```bash
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   
   Create a `.env` file in the project root with:
   ```
   DATABASE_URL=postgresql://username:password@localhost:5432/crypto_trading
   PGUSER=username
   PGPASSWORD=password
   PGHOST=localhost
   PGPORT=5432
   PGDATABASE=crypto_trading
   
   # Optional: Add your Binance API keys for live data
   BINANCE_API_KEY=your_binance_api_key
   BINANCE_API_SECRET=your_binance_api_secret
   ```

## Running the Application

1. **Initialize Database** (first time only)
   ```bash
   python database.py
   ```

2. **Start the Application**
   ```bash
   streamlit run app.py
   ```

3. **Access the Application**
   - Open your browser and go to: http://localhost:8501

## Troubleshooting

- **Database Connection Issues**: Ensure PostgreSQL is running and credentials are correct
- **Package Import Errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`
- **Binance API Restrictions**: If you're in a restricted region, the app will fall back to alternative data sources

## Auto-Updating Data

The application will automatically start backfilling data when launched. You can also:

1. Click the "Backfill Database" button to manually trigger a full data refresh
2. Set the auto-refresh toggle to have the data update at your chosen interval