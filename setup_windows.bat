@echo off
echo ======================================================
echo CryptoApp Windows Setup Helper
echo ======================================================
echo.
echo This script will install the required npm package for data downloading.
echo.

REM Check if node.js is installed
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Node.js is not installed or not in PATH. Please install Node.js from https://nodejs.org/
    echo After installing, try running this script again.
    pause
    exit /b 1
)

REM Check npm
where npm >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo npm is not found. Please make sure Node.js is properly installed.
    pause
    exit /b 1
)

echo Node.js is installed. Version:
node --version
echo.
echo npm is installed. Version:
npm --version
echo.

REM Create package.json if it doesn't exist
if not exist package.json (
    echo Creating package.json file...
    echo {"name": "crypto-app", "version": "1.0.0", "private": true} > package.json
    echo Created package.json file.
    echo.
)

echo Installing binance-historical-data package...
call npm install binance-historical-data --no-save
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install binance-historical-data package.
    echo Don't worry - the application will automatically use direct downloads instead.
    echo.
) else (
    echo Successfully installed binance-historical-data package.
    echo.
)

echo Setup completed. You can now run the application.
echo.
echo Press any key to exit...
pause > nul