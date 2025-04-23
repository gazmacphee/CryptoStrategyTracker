# Enhanced Windows startup script for Crypto Trading App that fixes encoding issues
# and ensures all required tables are created

# Set UTF-8 encoding for Python to fix Windows terminal issues
$env:PYTHONIOENCODING = "utf-8"

Write-Host "========================================================================"
Write-Host "Starting Crypto Trading App with Enhanced Setup..." -ForegroundColor Green
Write-Host "========================================================================"

# Activate virtual environment if it exists
if (Test-Path ".\venv\Scripts\activate.ps1") {
    Write-Host "[1/5] Activating virtual environment..." -ForegroundColor Cyan
    & .\venv\Scripts\Activate.ps1
} else {
    Write-Host "[1/5] No virtual environment found. Using system Python." -ForegroundColor Yellow
}

# Create required database tables first
Write-Host "[2/5] Ensuring all database tables are created..." -ForegroundColor Cyan
python ensure_tables.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create database tables!" -ForegroundColor Red
    Write-Host "Trying fallback method..." -ForegroundColor Yellow
    
    # Try individual components to ensure tables exist
    python database.py
    python economic_indicators.py
}

# Fix for economic_indicators table specifically
Write-Host "[3/5] Ensuring economic indicators table exists..." -ForegroundColor Cyan
python -c "import economic_indicators; economic_indicators.create_economic_indicator_tables()"

# Start the Process Manager
Write-Host "[4/5] Starting Process Manager..." -ForegroundColor Cyan
Start-Process -FilePath "cmd" -ArgumentList "/k set PYTHONIOENCODING=utf-8 && python process_manager.py run"

# Give process manager time to initialize
Write-Host "Waiting for Process Manager to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Start the main application
Write-Host "[5/5] Starting Streamlit Application..." -ForegroundColor Cyan
Start-Process -FilePath "cmd" -ArgumentList "/k set PYTHONIOENCODING=utf-8 && python app.py"

Write-Host "========================================================================"
Write-Host "All components started successfully!" -ForegroundColor Green
Write-Host "========================================================================"
Write-Host "Application URLs:"
Write-Host "- Main App: http://localhost:8501" -ForegroundColor Cyan
Write-Host "========================================================================" 
Write-Host ""
Write-Host "This window can be closed. The application will continue running in the separate command windows."