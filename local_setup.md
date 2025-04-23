# Local Setup Instructions

This document provides instructions for setting up and running the Cryptocurrency Trading Analysis Platform on your local machine.

## Prerequisites

1. Python 3.8 or higher
2. PostgreSQL database
3. Required Python packages (listed in `dependencies.txt`)

## Step 1: Install Required Python Packages

```bash
# Install from the dependencies.txt file
pip install -r dependencies.txt

# Alternatively, install packages individually:
pip install nltk numpy openai pandas pandas-ta plotly psycopg2-binary python-dotenv requests scikit-learn streamlit trafilatura joblib sqlalchemy toml
```

## Step 2: Set Up the Database

1. Create a PostgreSQL database for the application
2. Update the `.env` file with your database credentials

`.env` file example:
```
# Database configuration
PGHOST=localhost
PGPORT=5432
PGUSER=your_username
PGPASSWORD=your_password
PGDATABASE=crypto_db
DATABASE_URL=postgresql://your_username:your_password@localhost:5432/crypto_db

# Binance API credentials (optional)
BINANCE_API_KEY=
BINANCE_API_SECRET=

# Application settings
RESET_DATABASE=true

# OpenAI API key (if using sentiment analysis)
OPENAI_API_KEY=your_openai_api_key
```

## Step 3: Run the Application

There are two ways to start the application:

### Option 1: Using the local starter script (recommended)

```bash
python start_local.py
```

This script will:
1. Load environment variables from your `.env` file
2. Verify that the necessary variables are set
3. Start the application

### Option 2: Using the main script directly

```bash
python reset_and_start.py
```

Note that this method requires that the environment variables are already properly set in your environment.

## Step 4: Access the Application

The application will be available at:
```
http://localhost:5000
```

## Windows-Specific Setup

For Windows users, we provide a helper script to simplify installation of the required Node.js package for data downloading:

1. Make sure Node.js is installed on your system. You can download it from the [Node.js website](https://nodejs.org/).
2. Run the `setup_windows.bat` script by double-clicking it in File Explorer.
3. The script will check for Node.js, create the required files, and install the binance-historical-data package.

If you encounter the error `binance-historical-data command not found in PATH` or similar, don't worry - the application will automatically fall back to direct downloads from Binance Data Vision.

## Troubleshooting

### Database Connection Issues

If you see an error like "DATABASE_URL environment variable is not set", make sure:
1. Your `.env` file is correctly formatted
2. You're using the `start_local.py` script which properly loads the `.env` file
3. Your DATABASE_URL follows the correct format: `postgresql://username:password@host:port/database_name`

### Data Loading Issues

The application will start backfilling cryptocurrency data in the background. This process may take some time depending on your internet connection and the amount of data being fetched.

### Windows npm Package Issues

If you're experiencing issues with binance-historical-data on Windows:

1. Run the `setup_windows.bat` script to install the package locally
2. Check if Node.js is properly installed and in your PATH
3. If problems persist, don't worry - the application will automatically use direct downloads instead

### PostgreSQL Installation

If you don't have PostgreSQL installed:

- **Windows**: Download the installer from the [PostgreSQL website](https://www.postgresql.org/download/windows/)
- **macOS**: Use Homebrew: `brew install postgresql`
- **Linux**: Use your distribution's package manager, e.g., `sudo apt install postgresql` for Ubuntu/Debian

After installation, create a database for the application:
```bash
createdb crypto_db
```

## Advanced Configuration

For advanced configuration options, refer to the `project_architecture.md` file.