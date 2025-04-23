# PowerShell script to start the CryptoApp on Windows
Write-Host "======================================================" -ForegroundColor Green
Write-Host "CryptoApp Windows Startup Script" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green
Write-Host ""

# Define variables
$APP_SCRIPT = "app.py"
$PROCESS_MANAGER = "process_manager.py"
$STREAMLIT_PORT = 5000

# Function to check if Python is installed
function Check-Python {
    try {
        $pythonVersion = python --version
        Write-Host "Python found: $pythonVersion" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "Python not found in PATH. Please install Python and ensure it's in your PATH." -ForegroundColor Red
        return $false
    }
}

# Function to check if PostgreSQL is installed
function Check-PostgreSQL {
    try {
        $pgVersion = psql --version
        Write-Host "PostgreSQL found: $pgVersion" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "PostgreSQL not found in PATH." -ForegroundColor Yellow
        Write-Host "If you've installed PostgreSQL, make sure it's in your PATH." -ForegroundColor Yellow
        Write-Host "You may need to run setup_local_database.ps1 first." -ForegroundColor Yellow
        return $false
    }
}

# Function to start the main app
function Start-App {
    Write-Host ""
    Write-Host "Starting Streamlit application..." -ForegroundColor Green
    
    # Start app in a new PowerShell window
    Start-Process powershell -ArgumentList "-Command", "cd '$PWD'; streamlit run $APP_SCRIPT --server.port $STREAMLIT_PORT"
    
    Write-Host "Streamlit application started in a new window." -ForegroundColor Green
    Write-Host "The app will be available at: http://localhost:$STREAMLIT_PORT" -ForegroundColor Cyan
}

# Function to start the process manager
function Start-ProcessManager {
    Write-Host ""
    Write-Host "Starting process manager..." -ForegroundColor Green
    
    # Start directly instead of using manage_processes.py
    Start-Process powershell -ArgumentList "-Command", "cd '$PWD'; python $PROCESS_MANAGER start"
    
    Write-Host "Process manager started in a new window." -ForegroundColor Green
}

# Main execution
if (-not (Check-Python)) {
    Write-Host "Please install Python and try again." -ForegroundColor Red
    exit 1
}

if (-not (Check-PostgreSQL)) {
    $response = Read-Host "Do you want to continue without PostgreSQL? (y/n)"
    if ($response -ne "y") {
        Write-Host "Please install PostgreSQL and run setup_local_database.ps1 first." -ForegroundColor Yellow
        exit 1
    }
}

# Check for environment file
if (-not (Test-Path ".env")) {
    Write-Host "Warning: .env file not found. Running setup_local_database.ps1..." -ForegroundColor Yellow
    
    if (Test-Path "setup_local_database.ps1") {
        & .\setup_local_database.ps1
    }
    else {
        Write-Host "setup_local_database.ps1 not found. Creating basic .env file..." -ForegroundColor Yellow
        
        @"
# Database configuration
PGHOST=localhost
PGPORT=5432
PGUSER=postgres
PGPASSWORD=2212
PGDATABASE=crypto
DATABASE_URL=postgresql://postgres:2212@localhost:5432/crypto

# Reset database on startup (set to false after first run)
RESET_DATABASE=true

# API Keys
BINANCE_API_KEY=
BINANCE_API_SECRET=
FRED_API_KEY=
ALPHA_VANTAGE_API_KEY=
OPENAI_API_KEY=
"@ | Out-File -FilePath ".env" -Encoding utf8
        
        Write-Host "Basic .env file created. You may need to update API keys." -ForegroundColor Yellow
    }
}

# Create .streamlit directory and config if needed
if (-not (Test-Path ".streamlit")) {
    New-Item -ItemType Directory -Path ".streamlit" | Out-Null
    
    @"
[server]
headless = true
address = "0.0.0.0"
port = $STREAMLIT_PORT
"@ | Out-File -FilePath ".streamlit/config.toml" -Encoding utf8
    
    Write-Host "Created Streamlit configuration" -ForegroundColor Green
}

# Start the application
Start-App

# Start the process manager
Start-ProcessManager

Write-Host ""
Write-Host "======================================================" -ForegroundColor Green
Write-Host "CryptoApp has been started!" -ForegroundColor Green
Write-Host "Main application: http://localhost:$STREAMLIT_PORT" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Green