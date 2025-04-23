# Start script for CryptoStrategyTracker on Windows
# This script handles the proper startup of all components

# Function to check if Python is installed
function Test-PythonInstalled {
    try {
        $pythonVersion = python --version
        Write-Host "✅ $pythonVersion detected" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "❌ Python not found. Please install Python 3.8 or higher" -ForegroundColor Red
        return $false
    }
}

# Function to check if required packages are installed
function Test-PackagesInstalled {
    Write-Host "Checking for required packages..." -ForegroundColor Cyan
    $packages = @(
        "streamlit",
        "pandas",
        "numpy",
        "psycopg2",  # Use psycopg2 instead of psycopg2-binary
        "sqlalchemy",
        "plotly"
    )
    
    $allInstalled = $true
    foreach ($package in $packages) {
        # Fix: Use a modified Python script that correctly handles module objects
        $pkgName = $package -replace "-", "_" # Replace hyphens with underscores for import
        
        # Create a temporary Python script file for this package check
        $checkScript = @"
try:
    import $pkgName
    print(f'✅ $pkgName is installed')
except Exception as e:
    print(f'❌ $pkgName is not installed: {e}')
"@
        $tempScriptPath = "temp_pkg_check_$pkgName.py"
        $checkScript | Out-File -FilePath $tempScriptPath -Encoding utf8
        
        # Run the script and capture output
        $result = python $tempScriptPath
        
        # Clean up
        Remove-Item $tempScriptPath
        
        if ($result -match "✅|installed") {
            # The package is installed, display in green
            Write-Host $result -ForegroundColor Green
        } else {
            # The package is not installed
            Write-Host $result -ForegroundColor Red
            $allInstalled = $false
        }
    }
    
    if (-not $allInstalled) {
        Write-Host "`nSome packages are missing. Run the following command to install them:" -ForegroundColor Yellow
        Write-Host "pip install -r dependencies.txt" -ForegroundColor Cyan
        return $false
    }
    
    return $true
}

# Function to check database connection
function Test-DatabaseConnection {
    Write-Host "`nTesting database connection..." -ForegroundColor Cyan
    
    try {
        # Create safer Python code that ensures proper error handling
        $pythonScript = @"
try:
    from database import get_db_connection, close_db_connection
    conn = get_db_connection()
    success = conn is not None
    if conn:
        close_db_connection(conn)
    print('✅ Database connection successful' if success else '❌ Database connection failed')
except Exception as e:
    print(f'❌ Database connection error: {e}')
"@
        $pythonScriptPath = "temp_db_check.py"
        $pythonScript | Out-File -FilePath $pythonScriptPath -Encoding utf8
        
        $result = python $pythonScriptPath
        Remove-Item $pythonScriptPath # Clean up
        
        if ($result -match "successful") {
            Write-Host $result -ForegroundColor Green
            return $true
        }
        else {
            Write-Host $result -ForegroundColor Red
            Write-Host "Check your .env file and make sure DATABASE_URL is set correctly" -ForegroundColor Yellow
            return $false
        }
    }
    catch {
        Write-Host "❌ Error testing database connection: $_" -ForegroundColor Red
        return $false
    }
}

# Function to start all processes
function Start-CryptoApp {
    Write-Host "`nStarting CryptoStrategyTracker..." -ForegroundColor Cyan
    
    # First make sure the process manager is stopped
    Write-Host "Ensuring clean start by stopping any running processes..." -ForegroundColor Yellow
    python process_manager.py stop --force
    
    # Then start the application
    Write-Host "Starting application..." -ForegroundColor Green
    python reset_and_start.py
    
    # Print status information
    Write-Host "`n=================================================================================" -ForegroundColor Green
    Write-Host "CryptoStrategyTracker is now running!" -ForegroundColor Green
    Write-Host "=================================================================================" -ForegroundColor Green
    Write-Host "Access the web interface at: http://localhost:5000" -ForegroundColor Cyan
    Write-Host "Check process status with: .\check_processes.ps1" -ForegroundColor Cyan
    Write-Host "Stop all processes with:   python process_manager.py stop" -ForegroundColor Cyan
    Write-Host "=================================================================================" -ForegroundColor Green
}

# Main execution
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host "CryptoStrategyTracker - Windows Startup" -ForegroundColor Cyan
Write-Host "=================================================================================" -ForegroundColor Cyan

$pythonOk = Test-PythonInstalled
if (-not $pythonOk) { exit 1 }

$packagesOk = Test-PackagesInstalled
if (-not $packagesOk) { 
    $response = Read-Host "Would you like to install the missing packages now? (y/n)"
    if ($response -eq "y") {
        Write-Host "Installing required packages..." -ForegroundColor Cyan
        pip install -r dependencies.txt
        $packagesOk = Test-PackagesInstalled
        if (-not $packagesOk) { exit 1 }
    }
    else {
        exit 1
    }
}

$dbOk = Test-DatabaseConnection
if (-not $dbOk) {
    $response = Read-Host "Would you like to set up the database now? (y/n)"
    if ($response -eq "y") {
        Write-Host "Running database setup script..." -ForegroundColor Cyan
        ./setup_local_database.ps1
        $dbOk = Test-DatabaseConnection
        if (-not $dbOk) { exit 1 }
    }
    else {
        exit 1
    }
}

# Start the application
Start-CryptoApp