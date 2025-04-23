@echo off
echo ======================================================
echo CryptoApp Windows Startup Script
echo ======================================================
echo.

REM Check if Python is installed
python --version >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python not found in PATH. Please install Python and ensure it's in your PATH.
    pause
    exit /b 1
)

REM Create .env file if it doesn't exist
if not exist .env (
    echo Warning: .env file not found. Running setup_local_database.ps1...
    
    if exist setup_local_database.ps1 (
        powershell -ExecutionPolicy Bypass -File setup_local_database.ps1
    ) else (
        echo setup_local_database.ps1 not found. Creating basic .env file...
        (
            echo # Database configuration
            echo PGHOST=localhost
            echo PGPORT=5432
            echo PGUSER=postgres
            echo PGPASSWORD=2212
            echo PGDATABASE=crypto
            echo DATABASE_URL=postgresql://postgres:2212@localhost:5432/crypto
            echo.
            echo # Reset database on startup (set to false after first run^)
            echo RESET_DATABASE=true
            echo.
            echo # API Keys
            echo BINANCE_API_KEY=
            echo BINANCE_API_SECRET=
            echo FRED_API_KEY=
            echo ALPHA_VANTAGE_API_KEY=
            echo OPENAI_API_KEY=
        ) > .env
        
        echo Basic .env file created. You may need to update API keys.
    )
)

REM Create .streamlit directory and config if needed
if not exist .streamlit mkdir .streamlit
if not exist .streamlit\config.toml (
    (
        echo [server]
        echo headless = true
        echo address = "0.0.0.0"
        echo port = 5000
    ) > .streamlit\config.toml
    
    echo Created Streamlit configuration
)

echo.
echo Starting Streamlit application in a new window...
start cmd /k "python -m streamlit run app.py --server.port 5000"

echo.
echo Starting process manager directly in a new window...
start cmd /k "python process_manager.py start"

echo.
echo ======================================================
echo CryptoApp has been started!
echo Main application: http://localhost:5000
echo ======================================================
echo.
echo Press any key to close this window...
pause > nul