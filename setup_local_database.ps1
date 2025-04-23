# PowerShell script to set up local PostgreSQL database for CryptoApp
Write-Host "======================================================" -ForegroundColor Green
Write-Host "CryptoApp Local PostgreSQL Database Setup" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green
Write-Host ""

# Check if PostgreSQL is installed
$pgInstalled = $false
$pgPassword = "2212"  # Default password for local development

try {
    # Try to use psql to check version
    $pgVersion = Invoke-Expression "psql --version" -ErrorAction Stop
    $pgInstalled = $true
    Write-Host "✓ PostgreSQL is installed:" -ForegroundColor Green
    Write-Host $pgVersion
} catch {
    Write-Host "❌ PostgreSQL is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please download and install PostgreSQL from:" -ForegroundColor Yellow
    Write-Host "https://www.postgresql.org/download/windows/" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "During installation:" -ForegroundColor Yellow
    Write-Host "1. Set password to: $pgPassword" -ForegroundColor Yellow
    Write-Host "2. Keep the default port (5432)" -ForegroundColor Yellow
    Write-Host "3. After installation, run this script again" -ForegroundColor Yellow
    
    Write-Host "Press any key to exit..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit
}

# Create the database
Write-Host ""
Write-Host "Creating 'crypto' database if it doesn't exist..." -ForegroundColor Yellow

$createDbScript = @"
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'crypto') THEN
    CREATE DATABASE crypto;
  END IF;
END
\$\$;
"@

try {
    # Write the script to a temporary file
    $tempFile = [System.IO.Path]::GetTempFileName()
    Set-Content -Path $tempFile -Value $createDbScript

    # Run the script with psql
    $env:PGPASSWORD = $pgPassword
    psql -U postgres -f $tempFile

    # Remove the temporary file
    Remove-Item -Path $tempFile

    Write-Host "✓ Database 'crypto' is ready" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to create database: $_" -ForegroundColor Red
    Write-Host "Please check your PostgreSQL installation and permissions." -ForegroundColor Yellow
    
    Write-Host "Press any key to exit..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit
}

# Create or update the .env file
Write-Host ""
Write-Host "Creating/updating .env file with local database credentials..." -ForegroundColor Yellow

# Check if the .env file exists and read its content
$envContent = @{}
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        $line = $_.Trim()
        if ($line -and !$line.StartsWith("#")) {
            $parts = $line.Split("=", 2)
            if ($parts.Count -eq 2) {
                $key = $parts[0]
                $value = $parts[1]
                $envContent[$key] = $value
            }
        }
    }
}

# Update database-related settings
$envContent["PGHOST"] = "localhost"
$envContent["PGPORT"] = "5432"
$envContent["PGUSER"] = "postgres"
$envContent["PGPASSWORD"] = $pgPassword
$envContent["PGDATABASE"] = "crypto"
$envContent["DATABASE_URL"] = "postgresql://postgres:$pgPassword@localhost:5432/crypto"
$envContent["RESET_DATABASE"] = "true"

# Preserve other settings that might be in the file
if (-not $envContent.ContainsKey("BINANCE_API_KEY")) { $envContent["BINANCE_API_KEY"] = "" }
if (-not $envContent.ContainsKey("BINANCE_API_SECRET")) { $envContent["BINANCE_API_SECRET"] = "" }
if (-not $envContent.ContainsKey("FRED_API_KEY")) { $envContent["FRED_API_KEY"] = "" }
if (-not $envContent.ContainsKey("ALPHA_VANTAGE_API_KEY")) { $envContent["ALPHA_VANTAGE_API_KEY"] = "" }
if (-not $envContent.ContainsKey("OPENAI_API_KEY")) { $envContent["OPENAI_API_KEY"] = "" }

# Write the updated content to the .env file
$envFileContent = @"
# Database configuration
PGHOST=localhost
PGPORT=5432
PGUSER=postgres
PGPASSWORD=$pgPassword
PGDATABASE=crypto
DATABASE_URL=postgresql://postgres:$pgPassword@localhost:5432/crypto

# Reset database on startup (set to false after first run)
RESET_DATABASE=true

# API Keys
"@

foreach ($key in @("BINANCE_API_KEY", "BINANCE_API_SECRET", "FRED_API_KEY", "ALPHA_VANTAGE_API_KEY", "OPENAI_API_KEY")) {
    $envFileContent += "`n$key=$($envContent[$key])"
}

# Save the file
Set-Content -Path ".env" -Value $envFileContent

Write-Host "✓ .env file updated with local database settings" -ForegroundColor Green

# Create .streamlit directory and config if needed
if (-not (Test-Path ".streamlit")) {
    New-Item -ItemType Directory -Path ".streamlit" | Out-Null
    
    $streamlitConfig = @"
[server]
headless = true
address = "0.0.0.0"
port = 5000
"@

    Set-Content -Path ".streamlit/config.toml" -Value $streamlitConfig
    Write-Host "✓ Created Streamlit configuration" -ForegroundColor Green
}

# Update the process manager timeout
$manageProcessesPath = "manage_processes.py"
if (Test-Path $manageProcessesPath) {
    $content = Get-Content $manageProcessesPath -Raw
    
    # Check if we need to modify the timeout
    if ($content -match "timeout=30\s+#\s+30\s+second\s+timeout") {
        $updatedContent = $content -replace "timeout=30\s+#\s+30\s+second\s+timeout", "timeout=120  # 120 second timeout"
        Set-Content -Path $manageProcessesPath -Value $updatedContent
        Write-Host "✓ Updated process manager timeout to 120 seconds" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "======================================================" -ForegroundColor Green
Write-Host "Setup complete! You can now run the application:" -ForegroundColor Green
Write-Host "1. Start the app: python app.py" -ForegroundColor Cyan
Write-Host "2. Open a new terminal window" -ForegroundColor Cyan
Write-Host "3. Start background processes: python manage_processes.py start" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")